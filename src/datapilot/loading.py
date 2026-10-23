"""Data loading: turn a raw CSV source into a DataFrame with sensible column types.

This is the first stage of the analysis pipeline (the ``load_dataset`` step). It runs
before profiling, so every later stage sees correct column types. It is a fixed
pipeline stage, not a tool the agent can call.

Public API
----------
load_csv                 Read a CSV from a path or file-like object. No type changes.
infer_datetime_columns   Convert text columns that contain dates to datetime.
prepare_dataframe        load_csv followed by infer_datetime_columns. This is the
                         entry point the rest of the package should use.

Scope
-----
* Where the data comes from (local disk, an upload, cloud storage) is the caller's
  concern. Any path or file-like object works.
* Registering the DataFrame for SQL queries belongs to ``tools/sql.py``.
* Column names are never changed, so they match the names used in user questions.

Known limitations
-----------------
* Dates are only recognised in the formats listed in ``DATE_FORMATS``.
* Slash-separated dates that are valid both day-first and month-first
  (for example ``03/04/2024``) are read day-first.
* Values in a date column that do not match the detected format become missing (NaT).
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
)
DATETIME_PARSE_THRESHOLD = 0.9  # share of the sample that must parse to call it a date column
DATETIME_SAMPLE_SIZE = 500  # how many non-missing values to test per column


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError as error:
        raise ValueError("File is not valid UTF-8 text (encoding problem).") from error


def _find_date_format(series: pd.Series) -> str | None:
    """Return the first format in DATE_FORMATS that parses enough of the column, else None."""
    values = series.dropna()
    if len(values) == 0:
        return None

    sample = values.sample(min(DATETIME_SAMPLE_SIZE, len(values)), random_state=0)
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() >= DATETIME_PARSE_THRESHOLD:
            return fmt
    return None


def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        series = result[column]
        if not pd.api.types.is_string_dtype(series):
            continue 

        fmt = _find_date_format(series)
        if fmt is not None:
            result[column] = pd.to_datetime(series, format=fmt, errors="coerce")
    return result


def prepare_dataframe(source: str | Path | BinaryIO) -> pd.DataFrame:
    return infer_datetime_columns(load_csv(source))