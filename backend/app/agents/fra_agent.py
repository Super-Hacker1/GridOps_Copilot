"""FRA diagnostic agent with a deterministic rule fallback."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from app.schemas.diagnosis import DiagnosticAnalysisResult


FRAArtifactPredictor = Callable[[pd.DataFrame], DiagnosticAnalysisResult]

FRA_REQUIRED_COLUMNS = (
    "frequency_hz",
    "magnitude_db",
    "phase_deg",
)

MIN_FRA_RULE_POINTS = 32
MIN_FRA_LOG_COVERAGE_DECADES = 4.0

_BANDS = {
    "low": (1.0, 3.3),
    "mid": (3.3, 4.7),
    "high": (4.7, 6.1),
}

_FAULT_BY_BAND = {
    "low": "core_clamping_issue_suspected",
    "mid": "winding_deformation_suspected",
    "high": "insulation_related_abnormality_suspected",
}

_ALLOWED_FRA_FAULT_CLASSES = {
    "healthy",
    "winding_deformation_suspected",
    "core_clamping_issue_suspected",
    "insulation_related_abnormality_suspected",
    "needs_human_review",
}


def _validated_curve(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [column for column in FRA_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required {name} FRA columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{name.capitalize()} FRA curve requires at least 1 row")

    frequency = frame["frequency_hz"].to_numpy(dtype=float)
    magnitude = frame["magnitude_db"].to_numpy(dtype=float)
    phase = frame["phase_deg"].to_numpy(dtype=float)

    if not (
        np.isfinite(frequency).all() and np.isfinite(magnitude).all() and np.isfinite(phase).all()
    ):
        raise ValueError(f"{name.capitalize()} FRA curve contains non-finite values")
    if (frequency <= 0).any():
        raise ValueError(f"{name.capitalize()} FRA frequencies must be greater than zero")

    order = np.argsort(frequency)
    frequency = frequency[order]
    if (np.diff(frequency) <= 0).any():
        raise ValueError(f"{name.capitalize()} FRA frequencies must be unique")

    unwrapped_phase = np.rad2deg(np.unwrap(np.deg2rad(phase[order])))
    return np.log10(frequency), magnitude[order], unwrapped_phase


def _insufficient_data_result(reason: str, metrics: dict[str, float]) -> DiagnosticAnalysisResult:
    return DiagnosticAnalysisResult(
        fault_class="needs_human_review",
        is_anomalous=False,
        confidence=0.25,
        anomaly_score=0.0,
        evidence=[reason],
        metrics=metrics,
        analysis_method="fra_rule_fallback",
        requires_human_review=True,
        anomaly_types=["insufficient_data"],
        likely_faults=[],
    )


def _rule_based_fra_analysis(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
) -> DiagnosticAnalysisResult:
    current_x, current_magnitude, current_phase = _validated_curve(current, "current")
    baseline_x, baseline_magnitude, baseline_phase = _validated_curve(baseline, "baseline")

    if len(current_x) < MIN_FRA_RULE_POINTS or len(baseline_x) < MIN_FRA_RULE_POINTS:
        return _insufficient_data_result(
            f"FRA fallback requires at least {MIN_FRA_RULE_POINTS} samples in both curves; "
            f"received {len(current_x)} current and {len(baseline_x)} baseline samples.",
            {
                "current_sample_count": float(len(current_x)),
                "baseline_sample_count": float(len(baseline_x)),
            },
        )

    overlap_low = max(float(current_x.min()), float(baseline_x.min()))
    overlap_high = min(float(current_x.max()), float(baseline_x.max()))
    coverage_decades = max(0.0, overlap_high - overlap_low)
    if coverage_decades < MIN_FRA_LOG_COVERAGE_DECADES:
        return _insufficient_data_result(
            f"FRA frequency coverage is {coverage_decades:.2f} decades; fallback rules require "
            f"at least {MIN_FRA_LOG_COVERAGE_DECADES:.2f} decades.",
            {
                "current_sample_count": float(len(current_x)),
                "baseline_sample_count": float(len(baseline_x)),
                "frequency_coverage_decades": coverage_decades,
            },
        )

    grid = np.linspace(overlap_low, overlap_high, 256)
    magnitude_delta = np.abs(
        np.interp(grid, current_x, current_magnitude)
        - np.interp(grid, baseline_x, baseline_magnitude)
    )
    phase_difference = np.interp(grid, current_x, current_phase) - np.interp(
        grid,
        baseline_x,
        baseline_phase,
    )
    phase_delta = np.abs((phase_difference + 180.0) % 360.0 - 180.0)

    scores: dict[str, float] = {}
    metrics: dict[str, float] = {}

    for band, (lower, upper) in _BANDS.items():
        mask = (grid >= max(lower, overlap_low)) & (grid < min(upper, overlap_high + 1e-9))
        if not mask.any():
            scores[band] = 0.0
            metrics[f"{band}_magnitude_mae_db"] = 0.0
            metrics[f"{band}_phase_mae_deg"] = 0.0
            continue

        magnitude_mae = float(magnitude_delta[mask].mean())
        phase_mae = float(phase_delta[mask].mean())
        scores[band] = min(1.0, (0.55 * magnitude_mae / 8.0) + (0.45 * phase_mae / 30.0))
        metrics[f"{band}_magnitude_mae_db"] = round(magnitude_mae, 3)
        metrics[f"{band}_phase_mae_deg"] = round(phase_mae, 3)

    strongest_band = max(scores, key=scores.get)
    anomaly_score = scores[strongest_band]
    is_anomalous = anomaly_score >= 0.18

    if is_anomalous:
        fault_class = _FAULT_BY_BAND[strongest_band]
        confidence = min(0.85, 0.55 + (0.3 * anomaly_score))
        evidence = [
            f"{strongest_band.capitalize()}-frequency FRA deviation score is "
            f"{anomaly_score:.2f} against the synthetic healthy baseline.",
            f"Mean absolute deviation in that band is "
            f"{metrics[f'{strongest_band}_magnitude_mae_db']:.2f} dB magnitude and "
            f"{metrics[f'{strongest_band}_phase_mae_deg']:.2f} degrees phase.",
        ]
    else:
        fault_class = "healthy"
        confidence = max(0.65, 0.9 - (0.5 * anomaly_score))
        evidence = ["FRA magnitude and phase deviations are within fallback rule tolerances."]

    metrics.update({f"{band}_deviation_score": round(score, 3) for band, score in scores.items()})

    return DiagnosticAnalysisResult(
        fault_class=fault_class,
        is_anomalous=is_anomalous,
        confidence=round(confidence, 3),
        anomaly_score=round(anomaly_score, 3),
        evidence=evidence,
        metrics=metrics,
        analysis_method="fra_rule_fallback",
        requires_human_review=True,
        anomaly_types=[] if not is_anomalous else [f"{strongest_band}_frequency_deviation"],
        likely_faults=[] if not is_anomalous else [fault_class],
    )


def analyze_fra(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    artifact_predictor: FRAArtifactPredictor | None = None,
) -> DiagnosticAnalysisResult:
    """Run an optional model hook, otherwise compare against a healthy baseline."""

    if artifact_predictor is not None:
        predictor_input = current.loc[:, list(FRA_REQUIRED_COLUMNS)].copy()
        try:
            prediction = artifact_predictor(predictor_input)
        except (
            Exception
        ):  # Model runtimes are optional; deterministic fallback must remain available.
            fallback = _rule_based_fra_analysis(current, baseline)
            fallback.evidence.append("FRA artifact predictor failed; rule fallback was used.")
            return fallback
        if (
            not isinstance(prediction, DiagnosticAnalysisResult)
            or prediction.fault_class not in _ALLOWED_FRA_FAULT_CLASSES
            or prediction.analysis_method != "fra_model_artifact"
        ):
            fallback = _rule_based_fra_analysis(current, baseline)
            fallback.evidence.append(
                "FRA artifact predictor output was invalid; rule fallback was used."
            )
            return fallback
        if not prediction.requires_human_review:
            prediction = prediction.model_copy(
                update={
                    "requires_human_review": True,
                    "evidence": [
                        *prediction.evidence,
                        "FRA artifact output requires qualified human review before action.",
                    ],
                }
            )
        return prediction

    return _rule_based_fra_analysis(current, baseline)
