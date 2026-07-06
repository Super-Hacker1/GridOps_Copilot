"""Uploaded diagnostic data validation and parsing."""

from io import BytesIO
import pandas as pd
import numpy as np


REQUIRED_DCRM_COLUMNS = (
    "time_ms",
    "resistance_micro_ohm",
    "travel_mm",
    "coil_current_a",
)


def parse_dcrm_csv(content: bytes) -> pd.DataFrame:
    if not content.strip():
        raise ValueError("DCRM CSV is empty")

    try:
        frame = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("DCRM CSV couldn't be parsed") from exc

    missing_columns = [column for column in REQUIRED_DCRM_COLUMNS if column not in frame.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required DCRM columns: {missing}")

    if frame.empty:
        raise ValueError("DCRM CSV contains no data rows")

    if len(frame) < 2:
        raise ValueError("DCRM CSV requires at least 2 data rows")

    columns_with_missing_values = [
        column for column in REQUIRED_DCRM_COLUMNS if frame[column].isna().any()
    ]

    if columns_with_missing_values:
        columns = ", ".join(columns_with_missing_values)
        raise ValueError(f"DCRM columns contain missing values: {columns}")

    converted_columns: dict[str, pd.Series] = {}
    non_numeric_columns: list[str] = []

    for column in REQUIRED_DCRM_COLUMNS:
        try:
            converted_columns[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )
        except (TypeError, ValueError):
            non_numeric_columns.append(column)

    if non_numeric_columns:
        columns = ", ".join(non_numeric_columns)
        raise ValueError(f"DCRM columns must contain numeric values: {columns}")

    for column, values in converted_columns.items():
        frame[column] = values

    non_finite_columns = [
        column
        for column in REQUIRED_DCRM_COLUMNS
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all()
    ]

    if non_finite_columns:
        columns = ", ".join(non_finite_columns)
        raise ValueError(f"DCRM columns contain non-finite values: {columns}")

    time_deltas = frame["time_ms"].diff().dropna()

    if (time_deltas <= 0).any():
        raise ValueError("DCRM time_ms must be strictly increasing")

    return frame
