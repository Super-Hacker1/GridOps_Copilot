"""Deterministic SCADA context agent."""

from datetime import datetime, timedelta

import pandas as pd

from app.schemas.diagnosis import SCADAContext


SCADA_REQUIRED_COLUMNS = (
    "asset_id",
    "timestamp",
    "current_a",
    "temperature_c",
    "status",
    "alarm_code",
)


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def analyze_scada_context(
    events: pd.DataFrame,
    *,
    asset_id: str,
    as_of: datetime,
    lookback: timedelta = timedelta(days=7),
) -> SCADAContext:
    """Return recent alarms and operational context without inferring a fault."""

    missing = [column for column in SCADA_REQUIRED_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError(f"Missing required SCADA columns: {', '.join(missing)}")
    if lookback <= timedelta(0):
        raise ValueError("SCADA lookback must be greater than zero")

    timestamps = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("SCADA timestamps must be valid")

    reference_time = _utc_timestamp(as_of)
    in_window = events["asset_id"].astype(str).str.strip().eq(asset_id) & timestamps.between(
        reference_time - lookback, reference_time, inclusive="both"
    )
    recent = events.loc[in_window].copy()
    recent["_timestamp"] = timestamps.loc[in_window]
    recent = recent.sort_values("_timestamp")

    if recent.empty:
        return SCADAContext(
            has_recent_alarm=False,
            alarm_codes=[],
            max_temperature_c=None,
            latest_current_a=None,
            baseline_current_a=None,
            current_deviation_pct=None,
            status_changed=False,
            evidence=[],
        )

    normalized_alarms = recent["alarm_code"].astype(str).str.strip()
    alarm_mask = ~normalized_alarms.str.upper().isin({"", "NONE", "NAN"})
    alarm_codes = list(dict.fromkeys(normalized_alarms.loc[alarm_mask].tolist()))
    max_temperature = float(pd.to_numeric(recent["temperature_c"]).max())
    currents = pd.to_numeric(recent["current_a"], errors="coerce")
    if currents.isna().any():
        raise ValueError("SCADA current values must be numeric")
    latest_current = float(currents.iloc[-1])
    prior_currents = currents.iloc[:-1]
    baseline_current = float(prior_currents.median()) if not prior_currents.empty else None
    if baseline_current is not None and baseline_current > 0:
        current_deviation = ((latest_current - baseline_current) / baseline_current) * 100.0
    else:
        current_deviation = None
    statuses = recent["status"].astype(str).str.strip().str.lower()
    status_changed = statuses.nunique() > 1

    evidence: list[str] = []
    for code in alarm_codes:
        alarm_time = recent.loc[normalized_alarms.eq(code), "_timestamp"].iloc[-1]
        evidence.append(f"Recent SCADA alarm {code} was recorded at {alarm_time.isoformat()}.")
    if status_changed:
        evidence.append("SCADA status changed within the seven-day context window.")
    if max_temperature >= 80.0:
        evidence.append(f"Recent maximum SCADA temperature was {max_temperature:.1f} C.")
    if current_deviation is not None and abs(current_deviation) >= 25.0:
        direction = "above" if current_deviation > 0 else "below"
        evidence.append(
            f"Latest SCADA current is {abs(current_deviation):.1f}% {direction} the asset-local "
            f"recent baseline ({latest_current:.1f} vs {baseline_current:.1f} A)."
        )

    return SCADAContext(
        has_recent_alarm=bool(alarm_codes),
        alarm_codes=alarm_codes,
        max_temperature_c=max_temperature,
        latest_current_a=latest_current,
        baseline_current_a=baseline_current,
        current_deviation_pct=None if current_deviation is None else round(current_deviation, 3),
        status_changed=status_changed,
        evidence=evidence,
    )
