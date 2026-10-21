"""Data loading: raw source in, clean DataFrame out.

This is a fixed pipeline stage (the `load_dataset` node of the graph), NOT an
agent tool. It runs before profiling so every later stage sees sensible types.

Responsibilities
----------------
* load_csv                -> bytes become a DataFrame, nothing else
* infer_datetime_columns  -> text columns that are really dates become datetimes
* prepare_dataframe       -> the single entry point the rest of the project calls

Not responsibilities (keep them out of this file)
-------------------------------------------------
* Where the bytes come from (local disk now, S3 later: a separate storage module
  will hand this module a file-like object).
* Registering the DataFrame in DuckDB (that belongs to tools/sql.py).
* Renaming or cleaning columns (users refer to the original column names).
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