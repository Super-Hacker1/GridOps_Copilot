"""Read dashboard summaries from the canonical asset registry."""

import csv
import json
from pathlib import Path

from app.schemas.asset import AssetSummary

CRITICALITY_SCORES = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
ASSET_TYPE_LABELS = {
    "transformer": "Transformer",
    "circuit_breaker": "Circuit Breaker",
}
REQUIRED_COLUMNS = {
    "asset_id",
    "asset_type",
    "voltage_level",
    "criticality",
}


class AssetRegistryError(RuntimeError):
    """Raised when the configured asset registry cannot be served."""


def _risk_level(health_score: int) -> str:
    if health_score < 70:
        return "High"
    if health_score < 85:
        return "Medium"
    return "Low"


def _asset_summary(row: dict[str, str | None]) -> AssetSummary:
    criticality_label = (row.get("criticality") or "").strip().lower()

    try:
        criticality = CRITICALITY_SCORES[criticality_label]
    except KeyError as exc:
        raise AssetRegistryError(
            f"Unsupported asset criticality: {criticality_label or '<empty>'}"
        ) from exc

    asset_type = (row.get("asset_type") or "").strip().lower()
    asset_type_label = ASSET_TYPE_LABELS.get(
        asset_type,
        asset_type.replace("_", " ").title(),
    )
    health_score = 100 - (8 * criticality)

    try:
        return AssetSummary(
            asset_id=(row.get("asset_id") or "").strip(),
            asset_type=asset_type_label,
            voltage_level="".join((row.get("voltage_level") or "").split()),
            criticality=criticality,
            health_score=health_score,
            risk_level=_risk_level(health_score),
        )
    except ValueError as exc:
        raise AssetRegistryError("Asset registry contains an invalid row") from exc


def load_asset_summaries(
    registry_path: Path,
    *,
    diagnosis_directory: Path | None = None,
) -> list[AssetSummary]:
    """Load dashboard summaries and overlay the latest persisted diagnosis."""

    try:
        with registry_path.open(encoding="utf-8", newline="") as registry_file:
            reader = csv.DictReader(registry_file)
            columns = set(reader.fieldnames or ())
            missing_columns = sorted(REQUIRED_COLUMNS - columns)

            if missing_columns:
                missing = ", ".join(missing_columns)
                raise AssetRegistryError(f"Asset registry is missing columns: {missing}")

            summaries = [_asset_summary(row) for row in reader]
    except OSError as exc:
        raise AssetRegistryError("Asset registry could not be read") from exc

    if not summaries:
        raise AssetRegistryError("Asset registry contains no assets")

    if diagnosis_directory is None or not diagnosis_directory.is_dir():
        return summaries

    latest_by_asset: dict[str, tuple[int, int, str]] = {}
    for diagnosis_path in diagnosis_directory.glob("diag_*.json"):
        try:
            payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
            asset_id = str(payload["asset_id"])
            risk_score = int(payload["risk_score"])
            risk_level = str(payload["risk_level"])
            modified_at = diagnosis_path.stat().st_mtime_ns
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
        if not 0 <= risk_score <= 100 or risk_level not in {
            "Low",
            "Medium",
            "High",
            "Critical",
        }:
            continue
        existing = latest_by_asset.get(asset_id)
        if existing is None or modified_at > existing[0]:
            latest_by_asset[asset_id] = (modified_at, risk_score, risk_level)

    return [
        summary.model_copy(
            update={
                "health_score": 100 - latest_by_asset[summary.asset_id][1],
                "risk_level": latest_by_asset[summary.asset_id][2],
            }
        )
        if summary.asset_id in latest_by_asset
        else summary
        for summary in summaries
    ]
