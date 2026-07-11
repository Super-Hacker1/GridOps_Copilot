"""Deterministic diagnosis workflow orchestration."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.agents.dcrm_agent import diagnose_dcrm
from app.agents.fra_agent import FRAArtifactPredictor, analyze_fra
from app.agents.ingestion_agent import ingest_csv
from app.agents.maintenance_agent import analyze_maintenance_context
from app.agents.scada_agent import analyze_scada_context
from app.schemas.diagnosis import DiagnosisResult, DiagnosticType, RiskLevel
from app.services.risk_scoring import score_risk


@dataclass(frozen=True)
class DiagnosisReferenceData:
    """Validated reference frames used by the deterministic workflow."""

    assets: pd.DataFrame
    scada: pd.DataFrame
    maintenance: pd.DataFrame
    dcrm_baseline: pd.DataFrame
    fra_baseline: pd.DataFrame

    @classmethod
    def from_directory(cls, directory: Path) -> "DiagnosisReferenceData":
        """Load the canonical generated demo files, not the legacy tiny fixtures."""

        directory = Path(directory)

        def load(filename: str, file_type: str) -> pd.DataFrame:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Diagnosis reference file not found: {path}")
            return ingest_csv(path.read_bytes(), declared_type=file_type).frame

        return cls(
            assets=load("assets.csv", "assets"),
            scada=load("scada_events.csv", "scada"),
            maintenance=load("maintenance_logs.csv", "maintenance"),
            dcrm_baseline=load("dcrm_healthy.csv", "dcrm"),
            fra_baseline=load("fra_healthy.csv", "fra"),
        )


_ASSET_ID_COLUMN: dict[DiagnosticType, str] = {
    "dcrm": "breaker_id",
    "fra": "transformer_id",
}

_EXPECTED_ASSET_TYPE: dict[DiagnosticType, str] = {
    "dcrm": "circuit_breaker",
    "fra": "transformer",
}

_DISPLAY_ASSET_TYPE = {
    "circuit_breaker": "Circuit Breaker",
    "transformer": "Transformer",
}


def _select_asset_curve(
    frame: pd.DataFrame,
    *,
    asset_id: str,
    diagnostic_type: DiagnosticType,
    source_name: str,
    not_after: datetime | None = None,
) -> tuple[pd.DataFrame, datetime]:
    id_column = _ASSET_ID_COLUMN[diagnostic_type]
    if id_column not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError(
            f"{source_name.capitalize()} {diagnostic_type.upper()} data has no asset identity"
        )

    asset_mask = frame[id_column].astype(str).str.strip().eq(asset_id)
    selected = frame.loc[asset_mask].copy()
    if selected.empty:
        raise ValueError(f"{source_name.capitalize()} data does not contain asset {asset_id}")

    timestamps = pd.to_datetime(selected["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{source_name.capitalize()} diagnostic timestamps must be valid")
    if not_after is not None:
        cutoff = pd.Timestamp(not_after)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        eligible = timestamps.le(cutoff)
        if not eligible.any():
            raise ValueError(
                f"No {source_name} for asset {asset_id} exists at or before {cutoff.isoformat()}"
            )
        selected = selected.loc[eligible].copy()
        timestamps = timestamps.loc[eligible]
    latest_timestamp = timestamps.max()
    selected = selected.loc[timestamps.eq(latest_timestamp)].copy()

    curve_dimension = "operation_type" if diagnostic_type == "dcrm" else "winding"
    if selected[curve_dimension].astype(str).str.strip().nunique() != 1:
        raise ValueError(
            f"{source_name.capitalize()} data must identify one {curve_dimension} per diagnosis"
        )

    return selected, latest_timestamp.to_pydatetime()


def _recommendation(
    *,
    risk_level: RiskLevel,
    diagnostic_type: DiagnosticType,
) -> str:
    subject = "circuit breaker" if diagnostic_type == "dcrm" else "transformer"
    if risk_level == "Low":
        action = f"Continue routine monitoring of the {subject}"
    elif risk_level == "Medium":
        action = f"Review the {subject} during the next maintenance window"
    elif risk_level == "High":
        action = f"Schedule a prompt diagnostic inspection of the {subject}"
    else:
        action = f"Escalate the {subject} finding promptly to a senior maintenance engineer"
    return f"{action} and confirm the finding with a qualified engineer before action."


class DiagnosisOrchestrator:
    """Combine diagnostics, operational context, history, and risk."""

    def __init__(
        self,
        reference_data: DiagnosisReferenceData,
        *,
        fra_artifact_predictor: FRAArtifactPredictor | None = None,
    ) -> None:
        self.reference_data = reference_data
        self.fra_artifact_predictor = fra_artifact_predictor

    def diagnose(
        self,
        *,
        asset_id: str,
        diagnostic_type: DiagnosticType,
        current: pd.DataFrame,
        data_quality_warnings: tuple[str, ...] = (),
        as_of: datetime | None = None,
    ) -> DiagnosisResult:
        """Run the complete deterministic core for one uploaded asset curve."""

        assets = self.reference_data.assets
        asset_rows = assets.loc[assets["asset_id"].astype(str).str.strip().eq(asset_id)]
        if asset_rows.empty:
            raise ValueError(f"Unknown asset: {asset_id}")
        if len(asset_rows) != 1:
            raise ValueError(f"Asset registry contains duplicate asset: {asset_id}")

        asset = asset_rows.iloc[0]
        asset_type = str(asset["asset_type"]).strip().lower()
        expected_type = _EXPECTED_ASSET_TYPE[diagnostic_type]
        if asset_type != expected_type:
            raise ValueError(
                f"Asset {asset_id} of type {asset_type} does not support "
                f"{diagnostic_type.upper()} diagnosis"
            )

        current_curve, diagnostic_time = _select_asset_curve(
            current,
            asset_id=asset_id,
            diagnostic_type=diagnostic_type,
            source_name="current",
        )
        baseline_frame = (
            self.reference_data.dcrm_baseline
            if diagnostic_type == "dcrm"
            else self.reference_data.fra_baseline
        )
        baseline_curve, _ = _select_asset_curve(
            baseline_frame,
            asset_id=asset_id,
            diagnostic_type=diagnostic_type,
            source_name="baseline",
            not_after=diagnostic_time,
        )
        curve_dimension = "operation_type" if diagnostic_type == "dcrm" else "winding"
        current_configuration = str(current_curve[curve_dimension].iloc[0]).strip().lower()
        baseline_configuration = str(baseline_curve[curve_dimension].iloc[0]).strip().lower()
        if current_configuration != baseline_configuration:
            raise ValueError(
                f"Current {curve_dimension} '{current_configuration}' does not match "
                f"baseline {curve_dimension} '{baseline_configuration}'"
            )

        if diagnostic_type == "dcrm":
            analysis = diagnose_dcrm(current_curve, baseline_curve)
        else:
            analysis = analyze_fra(
                current_curve,
                baseline_curve,
                artifact_predictor=self.fra_artifact_predictor,
            )

        reference_time = as_of or diagnostic_time
        scada = analyze_scada_context(
            self.reference_data.scada,
            asset_id=asset_id,
            as_of=reference_time,
        )
        maintenance = analyze_maintenance_context(
            self.reference_data.maintenance,
            asset_id=asset_id,
            diagnostic_type=diagnostic_type,
            as_of=reference_time.date(),
        )
        risk = score_risk(
            anomaly_score=analysis.anomaly_score,
            criticality=str(asset["criticality"]),
            recent_alarm=scada.has_recent_alarm,
            maintenance_overdue=maintenance.is_overdue,
            historical_recurrence=maintenance.has_recurrence,
            data_quality_warning_count=len(data_quality_warnings),
        )

        warning_penalty = min(0.3, 0.1 * len(data_quality_warnings))
        adjusted_confidence = round(max(0.0, analysis.confidence - warning_penalty), 3)
        warning_evidence = [f"Data quality warning: {warning}" for warning in data_quality_warnings]
        evidence = [
            *analysis.evidence,
            *scada.evidence,
            *maintenance.evidence,
            *warning_evidence,
        ]
        requires_human_review = (
            analysis.requires_human_review
            or analysis.confidence < 0.7
            or risk.level in {"High", "Critical"}
            or bool(data_quality_warnings)
        )

        return DiagnosisResult(
            asset_id=asset_id,
            asset_type=_DISPLAY_ASSET_TYPE[asset_type],
            diagnostic_type=diagnostic_type,
            fault_class=analysis.fault_class,
            confidence=adjusted_confidence,
            anomaly_score=analysis.anomaly_score,
            risk_score=risk.score,
            risk_level=risk.level,
            evidence=evidence,
            recommended_action=_recommendation(
                risk_level=risk.level,
                diagnostic_type=diagnostic_type,
            ),
            requires_human_review=requires_human_review,
            analysis_method=analysis.analysis_method,
            metrics=analysis.metrics,
            risk_factors=risk.factors,
            data_quality_warnings=list(data_quality_warnings),
        )
