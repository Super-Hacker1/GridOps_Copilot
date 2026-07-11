"""Circuit-breaker DCRM waveform analysis."""

import pandas as pd
from app.schemas.diagnosis import DCRMAnalysisResult

RESISTANCE_SPIKE_THRESHOLD_PCT = 25.0
MECHANISM_DELAY_THRESHOLD_PCT = 20.0
TRAVEL_COMPLETION_THRESHOLD_PCT = 20.0
COIL_CURRENT_DEVIATION_THRESHOLD_PCT = 30.0


def percentage_deviation(observed: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("Baseline value must be greater than zero")

    return ((observed - baseline) / baseline) * 100.0


def _travel_midpoint_elapsed(frame: pd.DataFrame) -> float:
    minimum_travel = float(frame["travel_mm"].min())
    maximum_travel = float(frame["travel_mm"].max())
    midpoint = minimum_travel + ((maximum_travel - minimum_travel) / 2.0)
    crossing = frame.loc[frame["travel_mm"] >= midpoint, "time_ms"]
    if crossing.empty:
        raise ValueError("DCRM travel does not reach its midpoint")
    return float(crossing.iloc[0] - frame["time_ms"].min())


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

    baseline_travel_span = float(baseline["travel_mm"].max() - baseline["travel_mm"].min())
    observed_travel_span = float(current["travel_mm"].max() - current["travel_mm"].min())
    if baseline_travel_span <= 0:
        raise ValueError("Baseline travel span must be greater than zero")
    travel_completion_deviation = max(
        0.0,
        ((baseline_travel_span - observed_travel_span) / baseline_travel_span) * 100.0,
    )
    baseline_travel_midpoint = _travel_midpoint_elapsed(baseline)
    if baseline_travel_midpoint <= 0:
        raise ValueError("Baseline travel midpoint time must be greater than zero")
    if observed_travel_span > 0:
        observed_travel_midpoint = _travel_midpoint_elapsed(current)
        travel_midpoint_delay = percentage_deviation(
            observed_travel_midpoint,
            baseline_travel_midpoint,
        )
    else:
        observed_travel_midpoint = observed_duration
        travel_midpoint_delay = 0.0

    baseline_coil_peak = float(baseline["coil_current_a"].max())
    observed_coil_peak = float(current["coil_current_a"].max())
    if baseline_coil_peak <= 0:
        raise ValueError("Baseline peak coil current must be greater than zero")
    coil_peak_deviation = percentage_deviation(observed_coil_peak, baseline_coil_peak)
    coil_peak_absolute_deviation = abs(coil_peak_deviation)

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

    if travel_completion_deviation >= TRAVEL_COMPLETION_THRESHOLD_PCT:
        if "mechanism_delay" not in anomaly_types:
            anomaly_types.append("mechanism_delay")
        if "possible_operating_mechanism_issue" not in likely_faults:
            likely_faults.append("possible_operating_mechanism_issue")
        evidence.append(
            f"Travel span is {travel_completion_deviation:.1f}% below baseline "
            f"({observed_travel_span:.2f} vs {baseline_travel_span:.2f} mm)."
        )

    if travel_midpoint_delay >= MECHANISM_DELAY_THRESHOLD_PCT:
        if "mechanism_delay" not in anomaly_types:
            anomaly_types.append("mechanism_delay")
        if "possible_operating_mechanism_issue" not in likely_faults:
            likely_faults.append("possible_operating_mechanism_issue")
        if duration_deviation < MECHANISM_DELAY_THRESHOLD_PCT:
            evidence.append(
                f"Travel midpoint is {travel_midpoint_delay:.1f}% later than baseline "
                f"({observed_travel_midpoint:.2f} vs {baseline_travel_midpoint:.2f} ms)."
            )

    if coil_peak_absolute_deviation >= COIL_CURRENT_DEVIATION_THRESHOLD_PCT:
        anomaly_types.append("abnormal_coil_current")
        if "possible_operating_mechanism_issue" not in likely_faults:
            likely_faults.append("possible_operating_mechanism_issue")
        evidence.append(
            f"Peak coil current differs from baseline by {coil_peak_absolute_deviation:.1f}% "
            f"({observed_coil_peak:.2f} vs {baseline_coil_peak:.2f} A)."
        )

    resistance_severity = max(
        resistance_deviation / RESISTANCE_SPIKE_THRESHOLD_PCT,
        0.0,
    )
    duration_severity = max(
        duration_deviation / MECHANISM_DELAY_THRESHOLD_PCT,
        0.0,
    )
    travel_completion_severity = max(
        travel_completion_deviation / TRAVEL_COMPLETION_THRESHOLD_PCT,
        0.0,
    )
    travel_midpoint_severity = max(
        travel_midpoint_delay / MECHANISM_DELAY_THRESHOLD_PCT,
        0.0,
    )
    coil_current_severity = coil_peak_absolute_deviation / COIL_CURRENT_DEVIATION_THRESHOLD_PCT

    maximum_severity = max(
        resistance_severity,
        duration_severity,
        travel_completion_severity,
        travel_midpoint_severity,
        coil_current_severity,
    )

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
            "baseline_travel_span_mm": baseline_travel_span,
            "observed_travel_span_mm": observed_travel_span,
            "travel_completion_deviation_pct": travel_completion_deviation,
            "baseline_travel_midpoint_ms": baseline_travel_midpoint,
            "observed_travel_midpoint_ms": observed_travel_midpoint,
            "travel_midpoint_delay_pct": travel_midpoint_delay,
            "baseline_peak_coil_current_a": baseline_coil_peak,
            "observed_peak_coil_current_a": observed_coil_peak,
            "coil_peak_deviation_pct": coil_peak_deviation,
            "coil_peak_absolute_deviation_pct": coil_peak_absolute_deviation,
        },
        evidence=evidence,
    )
