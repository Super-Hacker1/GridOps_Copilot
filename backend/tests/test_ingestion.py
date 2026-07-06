"""Tests for diagnostic data ingestion."""

from app.services import ingestion

def test_parse_dcrm_csv_accepts_required_columns() -> None:
    content = (
        b"time_ms,resistance_micro_ohm,travel_mm,coil_current_a\n"
        b"0,100,0,0\n"
        b"1,105,2,1.5\n"
    )
    
    assert hasattr(ingestion, "parse_dcrm_csv"), "DCRM parser has not been implemented"

    frame = ingestion.parse_dcrm_csv(content)

    assert list(frame.columns) == [
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_a",
    ]

    assert len(frame) == 2