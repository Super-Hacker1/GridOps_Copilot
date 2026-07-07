"""Tests for DCRM waveform analysis."""

import pandas as pd
import pytest

from app.services import dcrm_analyzer


def make_waveform(
    times: list[float],
    resistance: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_ms": times,
            "resistance_micro_ohm": resistance,
            "travel_mm": [0.0, 5.0, 10.0],
            "coil_current_a": [0.0, 1.5, 0.5],
        }
    )


def test_analyze_dcrm_marks_similar_waveform_healthy() -> None:
    baseline = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 100.0, 100.0],
    )
    current = make_waveform(
        [0.0, 5.0, 10.0],
        [99.0, 102.0, 100.0],
    )

    result = dcrm_analyzer.analyze_dcrm(current, baseline)

    assert result.is_anomalous is False
    assert result.anomaly_types == []
    assert result.likely_faults == []
    assert result.confidence >= 0.9
    assert result.metrics["peak_resistance_deviation_pct"] == pytest.approx(2.0)


def test_analyze_dcrm_detects_abnormal_resistance_spike() -> None:
    baseline = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 100.0, 100.0],
    )
    current = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 140.0, 100.0],
    )

    result = dcrm_analyzer.analyze_dcrm(current, baseline)

    assert result.is_anomalous is True
    assert "abnormal_resistance_spike" in result.anomaly_types
    assert "possible_contact_wear" in result.likely_faults
    assert result.metrics["peak_resistance_deviation_pct"] == pytest.approx(40.0)
    assert result.evidence == [
        "Resistance peak is 40.0% above baseline (140.00 vs 100.00 micro-ohm)."
    ]


def test_analyze_dcrm_detects_mechanism_delay() -> None:
    baseline = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 100.0, 100.0],
    )
    current = make_waveform(
        [0.0, 6.5, 13.0],
        [100.0, 100.0, 100.0],
    )

    result = dcrm_analyzer.analyze_dcrm(current, baseline)

    assert result.is_anomalous is True
    assert "mechanism_delay" in result.anomaly_types
    assert "possible_operating_mechanism_issue" in result.likely_faults
    assert result.metrics["duration_deviation_pct"] == pytest.approx(30.0)
    assert result.evidence == ["Operation duration is 30.0% above baseline (13.00 vs 10.00 ms)."]


def test_analyze_dcrm_can_report_multiple_anomalies() -> None:
    baseline = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 100.0, 100.0],
    )
    current = make_waveform(
        [0.0, 6.5, 13.0],
        [100.0, 140.0, 100.0],
    )

    result = dcrm_analyzer.analyze_dcrm(current, baseline)

    assert result.anomaly_types == [
        "abnormal_resistance_spike",
        "mechanism_delay",
    ]
    assert len(result.evidence) == 2
    assert 0.0 <= result.confidence <= 1.0


def test_analyze_dcrm_rejects_zero_peak_baseline() -> None:
    baseline = make_waveform(
        [0.0, 5.0, 10.0],
        [0.0, 0.0, 0.0],
    )
    current = make_waveform(
        [0.0, 5.0, 10.0],
        [100.0, 110.0, 100.0],
    )

    with pytest.raises(
        ValueError,
        match="Baseline peak resistance must be greater than zero",
    ):
        dcrm_analyzer.analyze_dcrm(current, baseline)
