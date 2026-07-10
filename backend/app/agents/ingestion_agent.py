"""CSV ingestion and validation agent."""

from dataclasses import dataclass
from io import BytesIO
from typing import Literal, cast

import numpy as np
import pandas as pd


FileType = Literal["fra", "dcrm", "scada", "maintenance", "assets"]


@dataclass(frozen=True)
class IngestionResult:
    """Validated diagnostic data and its ingestion metadata."""

    file_type: FileType
    frame: pd.DataFrame
    asset_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.frame)


FILE_SCHEMAS: dict[FileType, tuple[str, ...]] = {
    "fra": (
        "transformer_id",
        "timestamp",
        "frequency_hz",
        "magnitude_db",
        "phase_deg",
        "winding",
        "label",
    ),
    "dcrm": (
        "breaker_id",
        "timestamp",
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_a",
        "operation_type",
        "label",
    ),
    "scada": (
        "asset_id",
        "timestamp",
        "voltage_kv",
        "current_a",
        "temperature_c",
        "status",
        "alarm_code",
    ),
    "maintenance": (
        "asset_id",
        "date",
        "issue",
        "action_taken",
        "severity",
        "next_due_date",
    ),
    "assets": (
        "asset_id",
        "asset_type",
        "voltage_level",
        "manufacturer",
        "age_years",
        "criticality",
        "bus_group",
        "connected_to",
    ),
}

NUMERIC_COLUMNS: dict[FileType, tuple[str, ...]] = {
    "fra": ("frequency_hz", "magnitude_db", "phase_deg"),
    "dcrm": (
        "time_ms",
        "resistance_micro_ohm",
        "travel_mm",
        "coil_current_a",
    ),
    "scada": ("voltage_kv", "current_a", "temperature_c"),
    "maintenance": (),
    "assets": ("age_years",),
}

ASSET_ID_COLUMNS: dict[FileType, str] = {
    "fra": "transformer_id",
    "dcrm": "breaker_id",
    "scada": "asset_id",
    "maintenance": "asset_id",
    "assets": "asset_id",
}

COLUMN_ALIASES = {
    "voltage": "voltage_kv",
    "current": "current_a",
    "temperature": "temperature_c",
}

MINIMUM_RECOMMENDED_ROWS: dict[FileType, int] = {
    "fra": 128,
    "dcrm": 50,
    "scada": 1,
    "maintenance": 1,
    "assets": 1,
}


def _parse_csv(content: bytes) -> pd.DataFrame:
    if not content.strip():
        raise ValueError("CSV is empty")

    try:
        frame = pd.read_csv(BytesIO(content))
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError("CSV could not be parsed") from exc

    if frame.empty:
        raise ValueError("CSV contains no data rows")

    return frame


def _normalize_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    normalized_columns: list[str] = []
    warnings: list[str] = []

    for column in frame.columns:
        original = str(column)
        lowered = original.strip().lower()
        normalized = COLUMN_ALIASES.get(lowered, lowered)
        normalized_columns.append(normalized)

        if original != normalized:
            warnings.append(f"Normalized column '{original}' to '{normalized}'.")

    duplicates = sorted(
        {column for column in normalized_columns if normalized_columns.count(column) > 1}
    )
    if duplicates:
        columns = ", ".join(duplicates)
        raise ValueError(f"CSV contains duplicate columns after normalization: {columns}")

    normalized_frame = frame.copy()
    normalized_frame.columns = normalized_columns
    return normalized_frame, warnings


def _validate_declared_type(
    declared_type: str | None,
) -> FileType | None:
    if declared_type is None:
        return None

    if declared_type not in FILE_SCHEMAS:
        supported = ", ".join(FILE_SCHEMAS)
        raise ValueError(
            f"Unsupported declared_type '{declared_type}'. Expected one of: {supported}"
        )

    return cast(FileType, declared_type)


def _detect_file_type(
    frame: pd.DataFrame,
    declared_type: FileType | None,
) -> FileType:
    available_columns = set(frame.columns)
    matches: list[FileType] = [
        file_type
        for file_type, required_columns in FILE_SCHEMAS.items()
        if set(required_columns).issubset(available_columns)
    ]

    if not matches:
        if declared_type is not None:
            missing = sorted(set(FILE_SCHEMAS[declared_type]) - available_columns)
            columns = ", ".join(missing)
            raise ValueError(f"Missing required {declared_type} columns: {columns}")

        raise ValueError("Unrecognized CSV schema")

    if len(matches) > 1:
        detected = ", ".join(matches)
        raise ValueError(f"Ambiguous CSV schema; matched: {detected}")

    detected_type = matches[0]
    if declared_type is not None and declared_type != detected_type:
        raise ValueError(
            f"Declared type '{declared_type}' does not match detected type '{detected_type}'"
        )

    return detected_type


def _validate_required_values(
    frame: pd.DataFrame,
    file_type: FileType,
) -> None:
    columns_with_missing_values = [
        column for column in FILE_SCHEMAS[file_type] if frame[column].isna().any()
    ]

    if columns_with_missing_values:
        columns = ", ".join(columns_with_missing_values)
        raise ValueError(f"Required columns contain missing values: {columns}")


def _convert_numeric_columns(
    frame: pd.DataFrame,
    file_type: FileType,
) -> None:
    non_numeric_columns: list[str] = []

    for column in NUMERIC_COLUMNS[file_type]:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError):
            non_numeric_columns.append(column)

    if non_numeric_columns:
        columns = ", ".join(non_numeric_columns)
        raise ValueError(f"Columns must contain numeric values: {columns}")

    non_finite_columns = [
        column
        for column in NUMERIC_COLUMNS[file_type]
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all()
    ]

    if non_finite_columns:
        columns = ", ".join(non_finite_columns)
        raise ValueError(f"Columns contain non-finite values: {columns}")


def _validate_waveform_order(
    frame: pd.DataFrame,
    file_type: FileType,
) -> None:
    if file_type == "dcrm":
        order_column = "time_ms"
        group_columns = ["breaker_id", "timestamp", "operation_type"]
    elif file_type == "fra":
        order_column = "frequency_hz"
        group_columns = ["transformer_id", "timestamp", "winding"]
    else:
        return

    for _, group in frame.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        if len(group) < 2:
            raise ValueError(f"{file_type.upper()} data requires at least 2 rows per curve")

        if not group[order_column].diff().dropna().gt(0).all():
            raise ValueError(
                f"{file_type.upper()} {order_column} must be strictly increasing within each curve"
            )


def _extract_asset_ids(
    frame: pd.DataFrame,
    file_type: FileType,
) -> tuple[str, ...]:
    column = ASSET_ID_COLUMNS[file_type]
    values = frame[column].astype(str).str.strip()
    return tuple(sorted(values.unique()))


def ingest_csv(
    content: bytes,
    *,
    declared_type: str | None = None,
) -> IngestionResult:
    """Parse, identify, normalize, and validate a diagnostic CSV."""

    frame = _parse_csv(content)
    frame, warnings = _normalize_columns(frame)
    normalized_declared_type = _validate_declared_type(declared_type)
    file_type = _detect_file_type(frame, normalized_declared_type)

    _validate_required_values(frame, file_type)
    _convert_numeric_columns(frame, file_type)
    _validate_waveform_order(frame, file_type)

    recommended_rows = MINIMUM_RECOMMENDED_ROWS[file_type]
    if len(frame) < recommended_rows:
        warnings.append(
            f"Only {len(frame)} rows were provided; "
            f"{file_type.upper()} data recommends at least "
            f"{recommended_rows} rows."
        )

    return IngestionResult(
        file_type=file_type,
        frame=frame,
        asset_ids=_extract_asset_ids(frame, file_type),
        warnings=tuple(warnings),
    )
