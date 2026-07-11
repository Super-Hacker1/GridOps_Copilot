"""End-to-end unit tests for the deterministic diagnosis core."""

from pathlib import Path

import pandas as pd
import pytest

from app.agents.ingestion_agent import ingest_csv
from app.agents.orchestrator import DiagnosisOrchestrator, DiagnosisReferenceData
from app.safety import SAFETY_STATEMENT


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def load_current(filename: str, file_type: str):
    return ingest_csv(
        (DATA_ROOT / filename).read_bytes(),
        declared_type=file_type,
    ).frame


@pytest.fixture
def orchestrator() -> DiagnosisOrchestrator:
    return DiagnosisOrchestrator(DiagnosisReferenceData.from_directory(DATA_ROOT))


def test_orchestrator_runs_cb_402_dcrm_golden_path(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    result = orchestrator.diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=load_current("dcrm_fault_contact_wear.csv", "dcrm"),
    )

    assert result.asset_type == "Circuit Breaker"
    assert result.fault_class == "possible_contact_wear"
    assert 61 <= result.risk_score <= 80
    assert result.risk_level == "High"
    assert result.analysis_method == "dcrm_rule_based"
    assert result.requires_human_review is True
    assert any("CB_TIMING_DEVIATION" in item for item in result.evidence)
    assert any("overdue by 23 days" in item for item in result.evidence)
    assert "qualified engineer" in result.recommended_action


def test_orchestrator_runs_tx_1_fra_fallback_golden_path(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    result = orchestrator.diagnose(
        asset_id="TX-1",
        diagnostic_type="fra",
        current=load_current("fra_fault_winding_shift.csv", "fra"),
    )

    assert result.asset_type == "Transformer"
    assert result.fault_class == "winding_deformation_suspected"
    assert result.analysis_method == "fra_rule_fallback"
    assert result.risk_level in {"Medium", "High"}
    assert any("TX_TEMP_HIGH" in item for item in result.evidence)
    assert result.requires_human_review is True


def test_orchestrator_rejects_diagnostic_type_incompatible_with_asset(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not support DCRM diagnosis",
    ):
        orchestrator.diagnose(
            asset_id="TX-1",
            diagnostic_type="dcrm",
            current=load_current("dcrm_fault_contact_wear.csv", "dcrm"),
        )


def test_orchestrator_rejects_current_curve_for_another_asset(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    current = load_current("dcrm_fault_contact_wear.csv", "dcrm")

    with pytest.raises(ValueError, match="does not contain asset CB-401"):
        orchestrator.diagnose(
            asset_id="CB-401",
            diagnostic_type="dcrm",
            current=current,
        )


def test_orchestrator_does_not_use_future_baseline() -> None:
    references = DiagnosisReferenceData.from_directory(DATA_ROOT)
    current = load_current("dcrm_fault_contact_wear.csv", "dcrm")
    future_baseline = current.copy()
    future_baseline["timestamp"] = "2026-07-09T10:00:00Z"
    references = DiagnosisReferenceData(
        assets=references.assets,
        scada=references.scada,
        maintenance=references.maintenance,
        dcrm_baseline=pd.concat(
            [references.dcrm_baseline, future_baseline],
            ignore_index=True,
        ),
        fra_baseline=references.fra_baseline,
    )

    result = DiagnosisOrchestrator(references).diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=current,
    )

    assert result.fault_class == "possible_contact_wear"


def test_orchestrator_errors_when_only_future_baseline_exists() -> None:
    references = DiagnosisReferenceData.from_directory(DATA_ROOT)
    future_baseline = references.dcrm_baseline.copy()
    future_baseline["timestamp"] = "2026-07-09T10:00:00Z"
    references = DiagnosisReferenceData(
        assets=references.assets,
        scada=references.scada,
        maintenance=references.maintenance,
        dcrm_baseline=future_baseline,
        fra_baseline=references.fra_baseline,
    )

    with pytest.raises(ValueError, match="No baseline.*at or before"):
        DiagnosisOrchestrator(references).diagnose(
            asset_id="CB-402",
            diagnostic_type="dcrm",
            current=load_current("dcrm_fault_contact_wear.csv", "dcrm"),
        )


def test_data_quality_warning_forces_human_review() -> None:
    orchestrator = DiagnosisOrchestrator(DiagnosisReferenceData.from_directory(DATA_ROOT))

    result = orchestrator.diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=load_current("dcrm_healthy.csv", "dcrm"),
        data_quality_warnings=("Only 40 samples were provided.",),
    )

    assert result.requires_human_review is True


def test_data_quality_warning_lowers_reported_confidence() -> None:
    orchestrator = DiagnosisOrchestrator(DiagnosisReferenceData.from_directory(DATA_ROOT))
    current = load_current("dcrm_healthy.csv", "dcrm")

    clean = orchestrator.diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=current,
    )
    warned = orchestrator.diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=current,
        data_quality_warnings=("Only 40 samples were provided.",),
    )

    assert warned.confidence < clean.confidence
    assert any("Data quality warning" in item for item in warned.evidence)


def test_diagnosis_carries_decision_support_safety_notice(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    result = orchestrator.diagnose(
        asset_id="CB-402",
        diagnostic_type="dcrm",
        current=load_current("dcrm_fault_contact_wear.csv", "dcrm"),
    )

    assert result.safety_statement == SAFETY_STATEMENT
    assert "qualified engineer" in result.recommended_action


def test_orchestrator_rejects_diagnostic_curve_with_mismatched_configuration(
    orchestrator: DiagnosisOrchestrator,
) -> None:
    current = load_current("dcrm_fault_contact_wear.csv", "dcrm").copy()
    current["operation_type"] = "open"

    with pytest.raises(ValueError, match="operation_type.*does not match"):
        orchestrator.diagnose(
            asset_id="CB-402",
            diagnostic_type="dcrm",
            current=current,
        )
