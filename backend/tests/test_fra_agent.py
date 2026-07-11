"""Tests for FRA rule fallback and optional artifact inference hook."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.fra_agent import analyze_fra
from app.agents.ingestion_agent import ingest_csv
from app.schemas.diagnosis import DiagnosticAnalysisResult


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def load_fra(filename: str):
    return ingest_csv(
        (DATA_ROOT / filename).read_bytes(),
        declared_type="fra",
    ).frame


@pytest.mark.parametrize(
    ("filename", "expected_fault"),
    [
        ("fra_healthy.csv", "healthy"),
        ("fra_fault_winding_shift.csv", "winding_deformation_suspected"),
        ("fra_fault_core_clamping.csv", "core_clamping_issue_suspected"),
        ("fra_fault_insulation.csv", "insulation_related_abnormality_suspected"),
    ],
)
def test_fra_rule_fallback_classifies_canonical_curves(
    filename: str,
    expected_fault: str,
) -> None:
    result = analyze_fra(
        load_fra(filename),
        load_fra("fra_healthy.csv"),
    )

    assert result.fault_class == expected_fault
    assert result.analysis_method == "fra_rule_fallback"
    assert result.requires_human_review is True
    assert 0.0 <= result.anomaly_score <= 1.0


def test_fra_rule_fallback_does_not_use_uploaded_label() -> None:
    current = load_fra("fra_healthy.csv").copy()
    current["label"] = "winding_deformation_suspected"

    result = analyze_fra(current, load_fra("fra_healthy.csv"))

    assert result.fault_class == "healthy"


def test_fra_agent_uses_optional_artifact_predictor_when_supplied() -> None:
    calls: list[int] = []

    def predictor(current):
        calls.append(len(current))
        return DiagnosticAnalysisResult(
            fault_class="winding_deformation_suspected",
            is_anomalous=True,
            confidence=0.91,
            anomaly_score=0.78,
            evidence=["Artifact prediction."],
            metrics={},
            analysis_method="fra_model_artifact",
            requires_human_review=True,
        )

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert calls == [512]
    assert result.analysis_method == "fra_model_artifact"
    assert result.confidence == 0.91


def test_fra_rule_fallback_routes_too_few_points_to_human_review() -> None:
    sparse = load_fra("fra_healthy.csv").iloc[::100].copy()

    result = analyze_fra(sparse, load_fra("fra_healthy.csv"))

    assert result.fault_class == "needs_human_review"
    assert result.requires_human_review is True
    assert result.confidence <= 0.5
    assert any("samples" in item.lower() for item in result.evidence)


def test_fra_rule_fallback_routes_narrow_frequency_coverage_to_human_review() -> None:
    healthy = load_fra("fra_healthy.csv")
    narrow = healthy.loc[healthy["frequency_hz"].between(10_000.0, 100_000.0)].copy()

    result = analyze_fra(narrow, healthy)

    assert len(narrow) >= 32
    assert result.fault_class == "needs_human_review"
    assert result.confidence <= 0.5
    assert any("coverage" in item.lower() for item in result.evidence)


def test_fra_artifact_predictor_receives_feature_columns_only() -> None:
    received_columns: list[str] = []

    def predictor(current):
        received_columns.extend(current.columns.tolist())
        return DiagnosticAnalysisResult(
            fault_class="healthy",
            is_anomalous=False,
            confidence=0.9,
            anomaly_score=0.05,
            evidence=["Artifact prediction."],
            metrics={},
            analysis_method="fra_model_artifact",
            requires_human_review=False,
        )

    analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert received_columns == ["frequency_hz", "magnitude_db", "phase_deg"]


def test_fra_agent_rejects_unknown_artifact_fault_taxonomy() -> None:
    def predictor(current):
        return DiagnosticAnalysisResult(
            fault_class="certified_failure",
            is_anomalous=True,
            confidence=0.99,
            anomaly_score=0.99,
            evidence=["Unsupported output."],
            metrics={},
            analysis_method="fra_model_artifact",
            requires_human_review=False,
        )

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert result.fault_class == "healthy"
    assert result.analysis_method == "fra_rule_fallback"
    assert any("invalid" in item.lower() for item in result.evidence)


def test_fra_agent_rejects_non_artifact_analysis_method_from_predictor() -> None:
    def predictor(current):
        return DiagnosticAnalysisResult(
            fault_class="healthy",
            is_anomalous=False,
            confidence=0.99,
            anomaly_score=0.01,
            evidence=["Wrong method."],
            metrics={},
            analysis_method="dcrm_rule_based",
            requires_human_review=False,
        )

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert result.analysis_method == "fra_rule_fallback"
    assert any("invalid" in item.lower() for item in result.evidence)


def test_fra_agent_falls_back_when_artifact_predictor_fails() -> None:
    def predictor(current):
        raise RuntimeError("inference unavailable")

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert result.fault_class == "healthy"
    assert result.analysis_method == "fra_rule_fallback"
    assert result.requires_human_review is True
    assert any("predictor failed" in item.lower() for item in result.evidence)


def test_fra_agent_rejects_unstructured_predictor_output() -> None:
    def predictor(current):
        return {"fault_class": "healthy", "analysis_method": "fra_model_artifact"}

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert result.analysis_method == "fra_rule_fallback"
    assert any("invalid" in item.lower() for item in result.evidence)


def test_fra_agent_forces_human_review_for_artifact_output() -> None:
    def predictor(current):
        return DiagnosticAnalysisResult(
            fault_class="healthy",
            is_anomalous=False,
            confidence=0.95,
            anomaly_score=0.02,
            evidence=["Artifact prediction."],
            metrics={},
            analysis_method="fra_model_artifact",
            requires_human_review=False,
        )

    result = analyze_fra(
        load_fra("fra_healthy.csv"),
        load_fra("fra_healthy.csv"),
        artifact_predictor=predictor,
    )

    assert result.analysis_method == "fra_model_artifact"
    assert result.requires_human_review is True


def test_fra_fallback_uses_circular_phase_distance_at_angle_wrap() -> None:
    baseline = load_fra("fra_healthy.csv").copy()
    current = baseline.copy()
    baseline["phase_deg"] = 179.0
    current["phase_deg"] = -179.0

    result = analyze_fra(current, baseline)

    assert result.fault_class == "healthy"
    assert result.metrics["mid_phase_mae_deg"] == pytest.approx(2.0)


def test_fra_fallback_unwraps_phase_before_interpolating_different_grids() -> None:
    baseline_frequency = np.logspace(1.0, 6.0, 128)
    current_frequency = np.logspace(1.01, 5.99, 127)

    def frame(frequency: np.ndarray) -> pd.DataFrame:
        unwrapped_phase = 160.0 + 40.0 * ((np.log10(frequency) - 1.0) / 5.0)
        wrapped_phase = (unwrapped_phase + 180.0) % 360.0 - 180.0
        return pd.DataFrame(
            {
                "frequency_hz": frequency,
                "magnitude_db": np.zeros_like(frequency),
                "phase_deg": wrapped_phase,
            }
        )

    result = analyze_fra(frame(current_frequency), frame(baseline_frequency))

    assert result.fault_class == "healthy"
    assert max(result.metrics[f"{band}_phase_mae_deg"] for band in ("low", "mid", "high")) < 0.5
