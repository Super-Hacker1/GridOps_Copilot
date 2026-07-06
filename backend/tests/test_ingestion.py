"""Tests for diagnostic data ingestion."""

from app.services import ingestion
import pytest


def test_parse_dcrm_csv_accepts_required_columns() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n0,100,0,0\n1,105,2,1.5\n"

    frame = ingestion.parse_dcrm_csv(content)

    assert list(frame.columns) == [
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_a",
    ]

    assert len(frame) == 2


def test_parse_dcrm_csv_rejects_missing_required_columns() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm\n0,100,0\n1,105,2\n"

    with pytest.raises(
        ValueError,
        match="Missing required DCRM columns: coil_current_a",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_rejects_empty_file() -> None:
    with pytest.raises(
        ValueError,
        match="DCRM CSV is empty",
    ):
        ingestion.parse_dcrm_csv(b"")


def test_parse_dcrm_csv_rejects_file_without_data_rows() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n"

    with pytest.raises(
        ValueError,
        match="DCRM CSV contains no data rows",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_requires_at_least_two_data_rows() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n0,100,0,0\n"

    with pytest.raises(
        ValueError,
        match="DCRM CSV requires at least 2 data rows",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_rejects_missing_values() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n0,100,0,0\n1,,2,1.5\n"

    with pytest.raises(
        ValueError,
        match="DCRM columns contain missing values: resistance_micro_ohm",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_rejects_non_numeric_values() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n0,100,0,0\n1,invalid,2,1.5\n"

    with pytest.raises(
        ValueError,
        match="DCRM columns must contain numeric values: resistance_micro_ohm",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_rejects_non_finite_values() -> None:
    content = b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n0,100,0,0\n1,inf,2,1.5\n"

    with pytest.raises(
        ValueError,
        match="DCRM columns contain non-finite values: resistance_micro_ohm",
    ):
        ingestion.parse_dcrm_csv(content)


@pytest.mark.parametrize(
    "time_values",
    [
        ("1", "0"),
        ("0", "0"),
    ],
    ids=["decreasing", "duplicate"],
)
def test_parse_dcrm_csv_rqeuires_strictly_increasing_time(time_values: tuple[str, str]) -> None:
    first_time, second_time = time_values
    content = (
        "time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n"
        f"{first_time},100,0,0\n"
        f"{second_time},105,2,1.5\n"
    ).encode()

    with pytest.raises(
        ValueError,
        match="DCRM time_ms must be strictly increasing",
    ):
        ingestion.parse_dcrm_csv(content)


def test_parse_dcrm_csv_rejects_malformed_csv() -> None:
    content = (
        b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n"
        b"0,100,0,0\n"
        b"1,105,2,1.5,unexpected\n"
    )

    with pytest.raises(
        ValueError,
        match="DCRM CSV couldn't be parsed",
    ):
        ingestion.parse_dcrm_csv(content)
