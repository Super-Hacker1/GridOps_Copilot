"""Deterministic maintenance-history context agent."""

from datetime import date

import pandas as pd

from app.schemas.diagnosis import DiagnosticType, MaintenanceContext


MAINTENANCE_REQUIRED_COLUMNS = (
    "asset_id",
    "date",
    "issue",
    "action_taken",
    "severity",
    "next_due_date",
)

_RECURRENCE_TERMS: dict[DiagnosticType, tuple[str, ...]] = {
    "dcrm": ("dcrm", "contact", "resistance", "mechanism"),
    "fra": ("fra", "winding", "core", "insulation"),
}

_ROUTINE_OR_NORMAL_TERMS = (
    "routine",
    "no abnormal",
    "no fault",
    "normal condition",
)


def analyze_maintenance_context(
    logs: pd.DataFrame,
    *,
    asset_id: str,
    diagnostic_type: DiagnosticType,
    as_of: date,
) -> MaintenanceContext:
    """Find relevant prior issues and overdue maintenance as of the test date."""

    missing = [column for column in MAINTENANCE_REQUIRED_COLUMNS if column not in logs.columns]
    if missing:
        raise ValueError(f"Missing required maintenance columns: {', '.join(missing)}")

    record_dates = pd.to_datetime(logs["date"], errors="coerce").dt.date
    due_dates = pd.to_datetime(logs["next_due_date"], errors="coerce").dt.date
    if record_dates.isna().any() or due_dates.isna().any():
        raise ValueError("Maintenance dates must be valid")

    relevant_mask = logs["asset_id"].astype(str).str.strip().eq(asset_id) & record_dates.le(as_of)
    relevant = logs.loc[relevant_mask].copy()
    relevant["_record_date"] = record_dates.loc[relevant_mask]
    relevant["_due_date"] = due_dates.loc[relevant_mask]

    if relevant.empty:
        return MaintenanceContext(
            is_overdue=False,
            overdue_days=0,
            has_recurrence=False,
            matching_issues=[],
            evidence=[],
        )

    latest_record = relevant.loc[relevant["_record_date"].idxmax()]
    overdue_days = max(0, (as_of - latest_record["_due_date"]).days)

    searchable = relevant["issue"].astype(str).str.lower()
    recurrence_mask = searchable.apply(
        lambda value: (
            not any(term in value for term in _ROUTINE_OR_NORMAL_TERMS)
            and any(term in value for term in _RECURRENCE_TERMS[diagnostic_type])
        )
    )
    matching_issues = relevant.loc[recurrence_mask, "issue"].astype(str).tolist()

    evidence: list[str] = []
    if overdue_days:
        evidence.append(
            f"Maintenance due date {latest_record['_due_date'].isoformat()} is overdue by "
            f"{overdue_days} days."
        )
    if matching_issues:
        evidence.append(f"Relevant maintenance history includes: {matching_issues[-1]}.")

    return MaintenanceContext(
        is_overdue=overdue_days > 0,
        overdue_days=overdue_days,
        has_recurrence=bool(matching_issues),
        matching_issues=matching_issues,
        evidence=evidence,
    )
