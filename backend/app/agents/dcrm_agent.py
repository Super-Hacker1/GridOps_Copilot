"""Structured DCRM diagnostic agent."""

import pandas as pd

from app.schemas.diagnosis import DiagnosticAnalysisResult
from app.services.dcrm_analyzer import analyze_dcrm


def diagnose_dcrm(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
) -> DiagnosticAnalysisResult:
    """Adapt the existing DCRM analyzer to the shared diagnosis contract."""

    analysis = analyze_dcrm(current, baseline)
    positive_deviations = (
        max(0.0, analysis.metrics["peak_resistance_deviation_pct"]),
        max(0.0, analysis.metrics["duration_deviation_pct"]),
        max(0.0, analysis.metrics["travel_completion_deviation_pct"]),
        max(0.0, analysis.metrics["travel_midpoint_delay_pct"]),
        analysis.metrics["coil_peak_absolute_deviation_pct"],
    )
    anomaly_score = min(1.0, max(positive_deviations) / 100.0)

    if analysis.likely_faults:
        fault_class = analysis.likely_faults[0]
    else:
        fault_class = "healthy"

    return DiagnosticAnalysisResult(
        fault_class=fault_class,
        is_anomalous=analysis.is_anomalous,
        confidence=analysis.confidence,
        anomaly_score=round(anomaly_score, 3),
        evidence=analysis.evidence,
        metrics=analysis.metrics,
        analysis_method="dcrm_rule_based",
        requires_human_review=analysis.is_anomalous or analysis.confidence < 0.7,
        anomaly_types=analysis.anomaly_types,
        likely_faults=analysis.likely_faults,
    )
