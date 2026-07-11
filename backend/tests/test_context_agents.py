"""Tests for deterministic SCADA and maintenance context agents."""

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from app.agents.maintenance_agent import analyze_maintenance_context
from app.agents.scada_agent import analyze_scada_context


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def test_scada_agent_finds_recent_cb_timing_alarm() -> None:
    events = pd.read_csv(DATA_ROOT / "scada_events.csv")

    context = analyze_scada_context(
        events,
        asset_id="CB-402",
        as_of=datetime(2026, 7, 8, 10, tzinfo=timezone.utc),
    )

    assert context.has_recent_alarm is True
    assert context.alarm_codes == ["CB_TIMING_DEVIATION"]
    assert context.max_temperature_c == 68.0
    assert any("CB_TIMING_DEVIATION" in item for item in context.evidence)


def test_scada_agent_returns_empty_context_for_unknown_asset() -> None:
    events = pd.read_csv(DATA_ROOT / "scada_events.csv")

    context = analyze_scada_context(
        events,
        asset_id="CB-999",
        as_of=datetime(2026, 7, 8, 10, tzinfo=timezone.utc),
    )

    assert context.has_recent_alarm is False
    assert context.alarm_codes == []
    assert context.evidence == []


def test_maintenance_agent_finds_cb_recurrence_and_stable_overdue_days() -> None:
    logs = pd.read_csv(DATA_ROOT / "maintenance_logs.csv")

    context = analyze_maintenance_context(
        logs,
        asset_id="CB-402",
        diagnostic_type="dcrm",
        as_of=date(2026, 7, 8),
    )

    assert context.is_overdue is True
    assert context.overdue_days == 23
    assert context.has_recurrence is True
    assert any("23 days" in item for item in context.evidence)


def test_maintenance_agent_does_not_mark_future_tx_due_date_overdue() -> None:
    logs = pd.read_csv(DATA_ROOT / "maintenance_logs.csv")

    context = analyze_maintenance_context(
        logs,
        asset_id="TX-1",
        diagnostic_type="fra",
        as_of=date(2026, 7, 8),
    )

    assert context.is_overdue is False
    assert context.overdue_days == 0
    assert context.has_recurrence is True


def test_maintenance_agent_uses_latest_applicable_due_date() -> None:
    logs = pd.DataFrame(
        [
            {
                "asset_id": "CB-402",
                "date": "2026-01-01",
                "issue": "Elevated contact resistance",
                "action_taken": "Contacts serviced",
                "severity": "high",
                "next_due_date": "2026-06-01",
            },
            {
                "asset_id": "CB-402",
                "date": "2026-07-01",
                "issue": "DCRM follow-up completed",
                "action_taken": "Results reviewed",
                "severity": "low",
                "next_due_date": "2027-07-01",
            },
        ]
    )

    context = analyze_maintenance_context(
        logs,
        asset_id="CB-402",
        diagnostic_type="dcrm",
        as_of=date(2026, 7, 8),
    )

    assert context.is_overdue is False
    assert context.overdue_days == 0


def test_maintenance_agent_excludes_routine_no_abnormality_from_recurrence() -> None:
    logs = pd.DataFrame(
        [
            {
                "asset_id": "CB-401",
                "date": "2026-01-18",
                "issue": "Routine breaker inspection",
                "action_taken": "No abnormal condition found",
                "severity": "low",
                "next_due_date": "2027-01-18",
            }
        ]
    )

    context = analyze_maintenance_context(
        logs,
        asset_id="CB-401",
        diagnostic_type="dcrm",
        as_of=date(2026, 7, 8),
    )

    assert context.has_recurrence is False
    assert context.matching_issues == []


def test_scada_agent_evidences_huge_asset_local_current_deviation() -> None:
    events = pd.DataFrame(
        [
            {
                "asset_id": "TX-9",
                "timestamp": "2026-07-08T08:00:00Z",
                "voltage_kv": 400.0,
                "current_a": 100.0,
                "temperature_c": 50.0,
                "status": "normal",
                "alarm_code": "NONE",
            },
            {
                "asset_id": "TX-9",
                "timestamp": "2026-07-08T09:00:00Z",
                "voltage_kv": 400.0,
                "current_a": 110.0,
                "temperature_c": 50.0,
                "status": "normal",
                "alarm_code": "NONE",
            },
            {
                "asset_id": "TX-9",
                "timestamp": "2026-07-08T10:00:00Z",
                "voltage_kv": 400.0,
                "current_a": 1000.0,
                "temperature_c": 50.0,
                "status": "normal",
                "alarm_code": "NONE",
            },
        ]
    )

    context = analyze_scada_context(
        events,
        asset_id="TX-9",
        as_of=datetime(2026, 7, 8, 10, tzinfo=timezone.utc),
    )

    assert context.current_deviation_pct > 800.0
    assert any("asset-local" in item and "current" in item.lower() for item in context.evidence)
