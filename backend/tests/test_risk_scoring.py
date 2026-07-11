"""Tests for deterministic weighted risk scoring."""

import pytest

from app.services.risk_scoring import risk_level_for_score, score_risk


def test_risk_scoring_combines_diagnostic_and_context_factors() -> None:
    result = score_risk(
        anomaly_score=0.5,
        criticality="critical",
        recent_alarm=True,
        maintenance_overdue=True,
        historical_recurrence=True,
    )

    assert result.score == 70
    assert result.level == "High"
    assert result.factors == {
        "diagnostic_anomaly": 25,
        "asset_criticality": 15,
        "recent_alarm": 10,
        "maintenance_overdue": 10,
        "historical_recurrence": 10,
        "data_quality_penalty": 0,
    }


def test_risk_scoring_applies_bounded_data_quality_penalty() -> None:
    result = score_risk(
        anomaly_score=1.0,
        criticality="critical",
        recent_alarm=True,
        maintenance_overdue=True,
        historical_recurrence=True,
        data_quality_warning_count=20,
    )

    assert result.score == 80
    assert result.factors["data_quality_penalty"] == -15


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "Low"),
        (30, "Low"),
        (31, "Medium"),
        (60, "Medium"),
        (61, "High"),
        (80, "High"),
        (81, "Critical"),
        (100, "Critical"),
    ],
)
def test_risk_level_boundaries(score: int, expected: str) -> None:
    assert risk_level_for_score(score) == expected
