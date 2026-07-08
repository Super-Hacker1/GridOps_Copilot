"""Tests for deterministic synthetic demo data generation"""

from pathlib import Path
import pandas as pd
from app.services import synthetic_data

EXPECTED_SCHEMAS = {
    "assets.csv": (
        "asset_id",
        "asset_type",
        "voltage_level",
        "manufacturer",
        "age_years",
        "criticality",
        "bus_group",
        "connected_to",
    ),
    "maintenance_logs.csv": (
        "asset_id",
        "date",
        "issue",
        "action_taken",
        "severity",
        "next_due_date",
    ),
    "scada_events.csv": (
        "asset_id",
        "timestamp",
        "voltage_kv",
        "current_a",
        "temperature_c",
        "status",
        "alarm_code",
    ),
    "fra_healthy.csv": (
        "transformer_id",
        "timestamp",
        "frequency_hz",
        "magnitude_db",
        "phase_deg",
        "winding",
        "label",
    ),
    "fra_fault_winding_shift.csv": (
        "transformer_id",
        "timestamp",
        "frequency_hz",
        "magnitude_db",
        "phase_deg",
        "winding",
        "label",
    ),
    "fra_fault_core_clamping.csv": (
        "transformer_id",
        "timestamp",
        "frequency_hz",
        "magnitude_db",
        "phase_deg",
        "winding",
        "label",
    ),
    "fra_fault_insulation.csv": (
        "transformer_id",
        "timestamp",
        "frequency_hz",
        "magnitude_db",
        "phase_deg",
        "winding",
        "label",
    ),
    "dcrm_healthy.csv": (
        "breaker_id",
        "timestamp",
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_A",
        "operation_type",
        "label",
    ),
    "dcrm_fault_contact_wear.csv": (
        "breaker_id",
        "timestamp",
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_A",
        "operation_type",
        "label",
    ),
    "dcrm_fault_mechanism_delay.csv": (
        "breaker_id",
        "timestamp",
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_A",
        "operation_type",
        "label",
    ),
}


def test_generate_synthetic_data_writes_expected_files(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)

    assert set(generated) == set(EXPECTED_SCHEMAS)
    assert all(path.is_file() for path in generated.values())


def test_generated_files_match_required_schemas(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)

    for filename, expected_columns in EXPECTED_SCHEMAS.items():
        frame = pd.read_csv(generated[filename])

        assert list(frame.columns) == list(expected_columns)
        assert not frame.empty
        assert not frame.isna().any().any()


def test_asset_registry_contains_demo_assets(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)
    assets = pd.read_csv(generated["assets.csv"])

    assert set(assets["asset_id"]) == {
        "TX-1",
        "TX-2",
        "CB-401",
        "CB-402",
        "CB-221",
    }


def test_generated_waveforms_are_ordered_and_labeled(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)
    expected_labels = {
        "fra_healthy.csv": "healthy",
        "fra_fault_winding_shift.csv": "winding_deformation_suspected",
        "fra_fault_core_clamping.csv": "core_clamping_issue_suspected",
        "fra_fault_insulation.csv": "insulation_related_abnormality_suspected",
        "dcrm_healthy.csv": "healthy",
        "dcrm_fault_contact_wear.csv": "contact_wear_suspected",
        "dcrm_fault_mechanism_delay.csv": "mechanism_delay_suspected",
    }

    for filename, expected_label in expected_labels.items():
        frame = pd.read_csv(generated[filename])
        assert set(frame["label"]) == {expected_label}

        if filename.startswith("fra_"):
            assert frame["frequency_hz"].diff().dropna().gt(0).all()
        else:
            assert frame["time_ms"].diff().dropna().gt(0).all()


def test_generation_is_repeatable_for_same_seed(tmp_path: Path) -> None:
    first = synthetic_data.generate_synthetic_data(
        tmp_path / "first",
        seed=42,
    )
    second = synthetic_data.generate_synthetic_data(
        tmp_path / "second",
        seed=42,
    )

    for filename in EXPECTED_SCHEMAS:
        assert first[filename].read_bytes() == second[filename].read_bytes()


def test_dcrm_faults_are_distinguishable_from_healthy_case(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)

    healthy = pd.read_csv(generated["dcrm_healthy.csv"])
    contact_wear = pd.read_csv(generated["dcrm_fault_contact_wear.csv"])
    mechanism_delay = pd.read_csv(generated["dcrm_fault_mechanism_delay.csv"])

    healthy_peak = healthy["resistance_micro_ohm"].max()
    contact_wear_peak = contact_wear["resistance_micro_ohm"].max()

    healthy_duration = healthy["time_ms"].max() - healthy["time_ms"].min()
    delayed_duration = mechanism_delay["time_ms"].max() - mechanism_delay["time_ms"].min()

    assert contact_wear_peak > healthy_peak * 1.25
    assert delayed_duration > healthy_duration * 1.20


def test_fra_faults_deviate_in_expected_frequency_regions(tmp_path: Path) -> None:
    generated = synthetic_data.generate_synthetic_data(tmp_path, seed=42)

    healthy = pd.read_csv(generated["fra_healthy.csv"])
    winding = pd.read_csv(generated["fra_fault_winding_shift.csv"])
    core = pd.read_csv(generated["fra_fault_core_clamping.csv"])
    insulation = pd.read_csv(generated["fra_fault_insulation.csv"])

    frequency = healthy["frequency_hz"]
    mid_frequency = frequency.between(1_000, 100_000)
    low_mid_frequency = frequency.between(100, 10_000)
    high_frequency = frequency.between(100_000, 1_000_000)

    winding_deviation = (winding["magnitude_db"] - healthy["magnitude_db"]).abs()
    core_deviation = (core["magnitude_db"] - healthy["magnitude_db"]).abs()
    insulation_deviation = (insulation["magnitude_db"] - healthy["magnitude_db"]).abs()

    assert winding_deviation[mid_frequency].mean() > 2.0
    assert core_deviation[low_mid_frequency].mean() > 2.0
    assert insulation_deviation[high_frequency].mean() > 8.0
