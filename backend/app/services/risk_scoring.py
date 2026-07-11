"""Deterministic asset risk scoring."""

from app.schemas.diagnosis import RiskAssessment, RiskLevel


CRITICALITY_WEIGHTS = {
    "low": 0,
    "medium": 5,
    "high": 10,
    "critical": 15,
}


def risk_level_for_score(score: int) -> RiskLevel:
    """Map the documented inclusive risk bands to a label."""

    if not 0 <= score <= 100:
        raise ValueError("Risk score must be between 0 and 100")
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    if score <= 80:
        return "High"
    return "Critical"


def score_risk(
    *,
    anomaly_score: float,
    criticality: str,
    recent_alarm: bool,
    maintenance_overdue: bool,
    historical_recurrence: bool,
    data_quality_warning_count: int = 0,
) -> RiskAssessment:
    """Combine bounded, explainable inputs into a 0-100 risk score."""

    if not 0.0 <= anomaly_score <= 1.0:
        raise ValueError("Anomaly score must be between 0 and 1")
    if data_quality_warning_count < 0:
        raise ValueError("Data quality warning count cannot be negative")

    normalized_criticality = criticality.strip().lower()
    if normalized_criticality not in CRITICALITY_WEIGHTS:
        raise ValueError(f"Unsupported asset criticality: {criticality}")

    factors = {
        "diagnostic_anomaly": int(round(anomaly_score * 50)),
        "asset_criticality": CRITICALITY_WEIGHTS[normalized_criticality],
        "recent_alarm": 10 if recent_alarm else 0,
        "maintenance_overdue": 10 if maintenance_overdue else 0,
        "historical_recurrence": 10 if historical_recurrence else 0,
        "data_quality_penalty": -min(15, data_quality_warning_count * 5),
    }
    score = max(0, min(100, sum(factors.values())))

    return RiskAssessment(
        score=score,
        level=risk_level_for_score(score),
        factors=factors,
    )
