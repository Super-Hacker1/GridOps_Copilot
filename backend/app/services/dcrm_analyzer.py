"""Circuit-breaker DCRM waveform analysis."""

import pandas as pd
from app.schemas.diagnosis import DCRMAnalysisResult

RESISTANCE_SPIKE_THRESHOLD_PCT = 25.0
MECHANISM_DELAY_THRESHOLD_PCT = 20.0


def percentage_deviation(observed: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("Baseline value must be greater than zero")

    return ((observed - baseline) / baseline) * 100.0


def analyze_dcrm(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
) -> DCRMAnalysisResult:
    baseline_peak = float(baseline["resistance_micro_ohm"].max())
    observed_peak = float(current["resistance_micro_ohm"].max())

    if baseline_peak <= 0:
        raise ValueError("Baseline peak resistance must be greater than zero")

    baseline_duration = float(baseline["time_ms"].max() - baseline["time_ms"].min())
    observed_duration = float(current["time_ms"].max() - current["time_ms"].min())

    if baseline_duration <= 0:
        raise ValueError("Baseline waveform duration must be greater than zero")

    resistance_deviation = percentage_deviation(
        observed_peak,
        baseline_peak,
    )
    duration_deviation = percentage_deviation(
        observed_duration,
        baseline_duration,
    )

    anomaly_types: list[str] = []
    likely_faults: list[str] = []
    evidence: list[str] = []

    if resistance_deviation >= RESISTANCE_SPIKE_THRESHOLD_PCT:
        anomaly_types.append("abnormal_resistance_spike")
        likely_faults.append("possible_contact_wear")
        evidence.append(
            f"Resistance peak is {resistance_deviation:.1f}% above baseline "
            f"({observed_peak:.2f} vs {baseline_peak:.2f} micro-ohm)."
        )

    if duration_deviation >= MECHANISM_DELAY_THRESHOLD_PCT:
        anomaly_types.append("mechanism_delay")
        likely_faults.append("possible_operating_mechanism_issue")
        evidence.append(
            f"Operation duration is {duration_deviation:.1f}% above baseline "
            f"({observed_duration:.2f} vs {baseline_duration:.2f} ms)."
        )

    resistance_severity = max(
        resistance_deviation / RESISTANCE_SPIKE_THRESHOLD_PCT,
        0.0,
    )
    duration_severity = max(
        duration_deviation / MECHANISM_DELAY_THRESHOLD_PCT,
        0.0,
    )

    maximum_severity = max(resistance_severity, duration_severity)

    if anomaly_types:
        confidence = min(0.95, 0.5 + (0.2 * maximum_severity))
    else:
        confidence = max(0.6, 0.95 - (0.3 * maximum_severity))
        evidence.append("Peak resistance and operation duration are within configured thresholds")

    return DCRMAnalysisResult(
        is_anomalous=bool(anomaly_types),
        anomaly_types=anomaly_types,
        likely_faults=likely_faults,
        confidence=round(confidence, 3),
        metrics={
            "baseline_peak_resistance_micro_ohm": baseline_peak,
            "observed_peak_resistance_micro_ohm": observed_peak,
            "peak_resistance_deviation_pct": resistance_deviation,
            "baseline_duration_ms": baseline_duration,
            "observed_duration_ms": observed_duration,
            "duration_deviation_pct": duration_deviation,
        },
        evidence=evidence,
    )
