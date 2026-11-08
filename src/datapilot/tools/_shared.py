"""Helpers shared by the tools that compare two groups of rows.

compare.py uses these today, and the statistical tests will too. They hold the input
checks and row selection that every two-group tool needs, so each tool only contains
its own statistic. The leading underscore marks the module as internal to the
``tools`` package: tools import it, nothing else should.

All functions raise ValueError with a message naming the problem, and none of them
modify the DataFrame they are given.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any of ``columns`` is not a column of ``df``."""
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found, columns are {list(df.columns)}")


def apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Return only the rows where every filter column equals its value.

    ``None`` or an empty dict returns all rows. A filter that matches nothing returns
    an empty DataFrame; it is up to the caller to decide whether that is an error.
    """
    rows = df
    for column, value in (filters or {}).items():
        rows = rows[rows[column] == value]
    return rows


def select_group(rows: pd.DataFrame, group_col: str, group: Any) -> pd.DataFrame:
    """Return the rows where ``group_col`` equals ``group``; raise if there are none."""
    selected = rows[rows[group_col] == group]
    if len(selected) == 0:
        available = list(rows[group_col].dropna().unique())
        raise ValueError(
            f"Group {group!r} not found in column {group_col!r}. Available groups: {available}"
        )
    return selected


def require_numeric(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if ``column`` is not numeric."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"{column!r} must be numeric, but its type is {df[column].dtype}")


def is_binary_metric(series: pd.Series) -> bool:
    """True if the column has at least one value and every non-missing value is 0 or 1.

    The average of a binary column is a rate, so tools label its values as proportions.
    """
    present = series.dropna()
    return len(present) > 0 and bool(present.isin([0, 1]).all())


def prepare_two_groups(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    filters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the inputs of a two-group comparison and return (rows_a, rows_b).

    Checks that the columns exist and that the groups differ, applies the filters, then
    selects each group. Whether the metric must be numeric depends on the tool, so each
    tool calls ``require_numeric`` itself.
    """
    validate_columns(df, [metric, group_col] + list(filters or {}))
    if group_a == group_b:
        raise ValueError(f"group_a and group_b are both {group_a!r}; they must be different")

    rows = apply_filters(df, filters)
    return select_group(rows, group_col, group_a), select_group(rows, group_col, group_b)