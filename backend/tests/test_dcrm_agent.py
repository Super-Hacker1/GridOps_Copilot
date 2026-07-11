"""Tests for the structured DCRM diagnosis adapter."""

from pathlib import Path

from app.agents.dcrm_agent import diagnose_dcrm
from app.agents.ingestion_agent import ingest_csv


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def load_dcrm(filename: str):
    return ingest_csv(
        (DATA_ROOT / filename).read_bytes(),
        declared_type="dcrm",
    ).frame


def test_dcrm_agent_reuses_analyzer_for_contact_wear_fixture() -> None:
    result = diagnose_dcrm(
        load_dcrm("dcrm_fault_contact_wear.csv"),
        load_dcrm("dcrm_healthy.csv"),
    )

    assert result.fault_class == "possible_contact_wear"
    assert result.is_anomalous is True
    assert result.analysis_method == "dcrm_rule_based"
    assert result.anomaly_score >= 0.4
    assert "abnormal_resistance_spike" in result.anomaly_types
    assert result.requires_human_review is True


def test_dcrm_agent_reports_healthy_fixture_without_fault_label_leakage() -> None:
    current = load_dcrm("dcrm_healthy.csv").copy()
    current["label"] = "contact_wear_suspected"

    result = diagnose_dcrm(current, load_dcrm("dcrm_healthy.csv"))

    assert result.fault_class == "healthy"
    assert result.is_anomalous is False
    assert result.anomaly_score == 0.0


def test_dcrm_agent_scores_coil_current_anomaly() -> None:
    baseline = load_dcrm("dcrm_healthy.csv")
    current = baseline.copy()
    current["coil_current_a"] *= 2.0

    result = diagnose_dcrm(current, baseline)

    assert "abnormal_coil_current" in result.anomaly_types
    assert result.anomaly_score >= 0.9
