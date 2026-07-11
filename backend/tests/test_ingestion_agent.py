"""Tests for multi-format CSV ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from app.agents import ingestion_agent
from app.services import synthetic_data


@pytest.fixture
def generated_files(tmp_path: Path) -> dict[str, Path]:
    return synthetic_data.generate_synthetic_data(tmp_path, seed=42)


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_asset_ids"),
    [
        (
            "assets.csv",
            "assets",
            ("CB-221", "CB-401", "CB-402", "TX-1", "TX-2"),
        ),
        (
            "maintenance_logs.csv",
            "maintenance",
            ("CB-221", "CB-401", "CB-402", "TX-1", "TX-2"),
        ),
        (
            "scada_events.csv",
            "scada",
            ("CB-221", "CB-401", "CB-402", "TX-1", "TX-2"),
        ),
        ("fra_healthy.csv", "fra", ("TX-1",)),
        ("dcrm_healthy.csv", "dcrm", ("CB-402",)),
    ],
)
def test_ingest_csv_detects_supported_file_types(
    generated_files: dict[str, Path],
    filename: str,
    expected_type: str,
    expected_asset_ids: tuple[str, ...],
) -> None:
    result = ingestion_agent.ingest_csv(generated_files[filename].read_bytes())

    assert result.file_type == expected_type
    assert result.asset_ids == expected_asset_ids
    assert result.row_count > 0


def test_ingest_csv_normalizes_dcrm_coil_current_column(
    generated_files: dict[str, Path],
) -> None:
    result = ingestion_agent.ingest_csv(generated_files["dcrm_healthy.csv"].read_bytes())

    assert "coil_current_a" in result.frame.columns
    assert "coil_current_A" not in result.frame.columns
    assert "Normalized column 'coil_current_A' to 'coil_current_a'." in result.warnings


def test_ingest_csv_normalizes_legacy_scada_columns() -> None:
    content = (
        b"asset_id,timestamp,voltage,current,temperature,status,alarm_code\n"
        b"CB-402,2026-07-08T09:00:00Z,400,510,42,closed,NONE\n"
    )

    result = ingestion_agent.ingest_csv(content)

    assert result.file_type == "scada"
    assert "voltage_kv" in result.frame.columns
    assert "current_a" in result.frame.columns
    assert "temperature_c" in result.frame.columns
    assert len(result.warnings) == 3


def test_ingest_csv_rejects_missing_required_columns() -> None:
    content = b"asset_id,asset_type\nCB-402,circuit_breaker\n"

    with pytest.raises(
        ValueError,
        match="Missing required assets columns",
    ):
        ingestion_agent.ingest_csv(content, declared_type="assets")


@pytest.mark.parametrize(
    ("resistance", "expected_message"),
    [
        ("", "Required columns contain missing values"),
        ("invalid", "Columns must contain numeric values"),
        ("inf", "Columns contain non-finite values"),
    ],
    ids=["missing", "non-numeric", "non-finite"],
)
def test_ingest_csv_rejects_invalid_numeric_values(
    resistance: str,
    expected_message: str,
) -> None:
    content = (
        "breaker_id,timestamp,time_ms,resistance_micro_ohm,"
        "travel_mm,coil_current_A,operation_type,label\n"
        "CB-402,2026-07-08T10:00:00Z,0,100,0,0,close,healthy\n"
        f"CB-402,2026-07-08T10:00:00Z,1,{resistance},2,1,"
        "close,healthy\n"
    ).encode()

    with pytest.raises(ValueError, match=expected_message):
        ingestion_agent.ingest_csv(content)


def test_ingest_csv_rejects_unknown_schema() -> None:
    content = b"unknown,value\nexample,1\n"

    with pytest.raises(ValueError, match="Unrecognized CSV schema"):
        ingestion_agent.ingest_csv(content)


def test_ingest_csv_rejects_declared_type_mismatch(
    generated_files: dict[str, Path],
) -> None:
    content = generated_files["fra_healthy.csv"].read_bytes()

    with pytest.raises(
        ValueError,
        match="Declared type 'dcrm' does not match detected type 'fra'",
    ):
        ingestion_agent.ingest_csv(content, declared_type="dcrm")


def test_ingest_csv_rejects_unsupported_declared_type() -> None:
    content = b"unknown,value\nexample,1\n"

    with pytest.raises(ValueError, match="Unsupported declared_type"):
        ingestion_agent.ingest_csv(content, declared_type="other")


def test_ingest_csv_rejects_unordered_dcrm_samples() -> None:
    content = (
        b"breaker_id,timestamp,time_ms,resistance_micro_ohm,"
        b"travel_mm,coil_current_A,operation_type,label\n"
        b"CB-402,2026-07-08T10:00:00Z,0,100,0,0,close,healthy\n"
        b"CB-402,2026-07-08T10:00:00Z,2,102,2,1,close,healthy\n"
        b"CB-402,2026-07-08T10:00:00Z,1,101,1,0.5,close,healthy\n"
    )

    with pytest.raises(
        ValueError,
        match="DCRM time_ms must be strictly increasing",
    ):
        ingestion_agent.ingest_csv(content)


def test_ingest_csv_rejects_unordered_fra_samples() -> None:
    content = (
        b"transformer_id,timestamp,frequency_hz,magnitude_db,"
        b"phase_deg,winding,label\n"
        b"TX-1,2026-07-08T11:00:00Z,100,-2,-4,HV,healthy\n"
        b"TX-1,2026-07-08T11:00:00Z,500,-5,-15,HV,healthy\n"
        b"TX-1,2026-07-08T11:00:00Z,400,-4,-12,HV,healthy\n"
    )

    with pytest.raises(
        ValueError,
        match="FRA frequency_hz must be strictly increasing",
    ):
        ingestion_agent.ingest_csv(content)


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        (b"", "CSV is empty"),
        (
            b'asset_id,asset_type\n"CB-402,circuit_breaker\n',
            "CSV could not be parsed",
        ),
    ],
    ids=["empty", "malformed"],
)
def test_ingest_csv_rejects_unreadable_content(
    content: bytes,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        ingestion_agent.ingest_csv(content)


def test_ingestion_result_exposes_validated_frame(
    generated_files: dict[str, Path],
) -> None:
    result = ingestion_agent.ingest_csv(generated_files["fra_healthy.csv"].read_bytes())

    assert isinstance(result.frame, pd.DataFrame)
    assert result.row_count == 512
    assert result.warnings == ()


def test_ingest_csv_rejects_blank_asset_identifier() -> None:
    content = (
        b"asset_id,asset_type,voltage_level,manufacturer,age_years,criticality,"
        b"bus_group,connected_to\n"
        b"   ,transformer,400 kV,Demo,10,critical,400kV Bus,BUS-400\n"
    )

    with pytest.raises(ValueError, match="Asset identifiers must not be blank"):
        ingestion_agent.ingest_csv(content, declared_type="assets")


def test_ingest_csv_rejects_invalid_diagnostic_timestamp() -> None:
    content = (
        b"transformer_id,timestamp,frequency_hz,magnitude_db,phase_deg,winding,label\n"
        b"TX-1,not-a-date,10,-1,-2,HV,healthy\n"
        b"TX-1,not-a-date,100,-2,-3,HV,healthy\n"
    )

    with pytest.raises(ValueError, match="timestamp must contain valid dates"):
        ingestion_agent.ingest_csv(content, declared_type="fra")


def test_ingest_csv_rejects_non_positive_fra_frequency() -> None:
    content = (
        b"transformer_id,timestamp,frequency_hz,magnitude_db,phase_deg,winding,label\n"
        b"TX-1,2026-07-08T10:00:00Z,0,-1,-2,HV,healthy\n"
        b"TX-1,2026-07-08T10:00:00Z,100,-2,-3,HV,healthy\n"
    )

    with pytest.raises(ValueError, match="FRA frequency_hz must be greater than zero"):
        ingestion_agent.ingest_csv(content, declared_type="fra")
