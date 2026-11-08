"""compare_group_means: compare the average of a metric between two groups of rows.

Part of the deterministic toolbox. A tool is a plain function: a DataFrame and typed
arguments go in, a ``Finding`` comes out. It has no hidden state and no knowledge of
LangChain or LangGraph (the agent layer wraps tools later). Input checks and row
selection live in ``_shared.py``; this module only holds what is specific to averages.

Conventions
-----------
* A "group" is the set of rows where ``group_col`` equals a given value.
* Direction: the change goes FROM group_a TO group_b.
      difference      = value_b - value_a
      relative_change = (value_b - value_a) / value_a * 100
* A binary metric is a numeric column whose non-missing values are all 0 or 1.
  Its group value is a rate:
      value_a, value_b   Unit.PROPORTION          e.g. 0.80
      difference         Unit.PERCENTAGE_POINTS   (value_b - value_a) * 100, e.g. -15.0
      relative_change    Unit.PERCENT
* Any other numeric metric is compared by its average:
      value_a, value_b   Unit.NUMBER
      difference         Unit.NUMBER
      relative_change    Unit.PERCENT
* n_a and n_b (Unit.COUNT) are the rows actually used, after dropping missing metric values.
* Keys of Finding.values, which the verifier looks up by name:
      n_a, n_b, value_a, value_b, difference, relative_change
* Nothing is rounded.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from datapilot.models import Finding, Measure, Unit
from datapilot.tools._shared import (
    MIN_GROUP_SIZE,
    is_binary_metric,
    prepare_two_groups,
    require_numeric,
)


def _summarise_group(rows: pd.DataFrame, metric: str) -> tuple[int, float | None]:
    """Return (number of rows with a metric value, their average or None if there are none)."""
    values = rows[metric].dropna()
    if len(values) == 0:
        return 0, None
    return len(values), float(values.mean())


def compare_group_means(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    finding_id: str,
    filters: dict[str, Any] | None = None,
) -> Finding:
    """Compare the average of ``metric`` between two groups defined by ``group_col``.

    Args:
        df: the prepared DataFrame.
        metric: numeric column to average (a 0/1 column gives a rate).
        group_col: column that defines the groups, e.g. "signup_quarter".
        group_a: the baseline group.
        group_b: the group compared to the baseline.
        finding_id: id for the returned Finding, supplied by the caller.
        filters: optional {column: value} pairs; only rows matching all of them are used.

    Raises:
        ValueError: for an unknown column, identical groups, a non-numeric metric, or a
            group with no rows.
    """
    rows_a, rows_b = prepare_two_groups(df, metric, group_col, group_a, group_b, filters)
    require_numeric(df, metric)
    filters = filters or {}

    n_a, value_a = _summarise_group(rows_a, metric)
    n_b, value_b = _summarise_group(rows_b, metric)

    # Look at the whole column so both groups are treated the same way.
    is_binary = is_binary_metric(df[metric])
    if is_binary:
        value_unit, difference_unit = Unit.PROPORTION, Unit.PERCENTAGE_POINTS
    else:
        value_unit, difference_unit = Unit.NUMBER, Unit.NUMBER

    caveats: list[str] = []
    for group, n, value in ((group_a, n_a, value_a), (group_b, n_b, value_b)):
        if value is None:
            caveats.append(f"Group {group!r} has no non-missing values, so its average is undefined")
        if n < MIN_GROUP_SIZE:
            caveats.append(f"Small sample: group {group!r} has only {n} rows")

    difference = None
    relative_change = None
    if value_a is not None and value_b is not None:
        difference = value_b - value_a
        if is_binary:
            difference = difference * 100  # proportions -> percentage points
        if value_a == 0:
            caveats.append("The baseline (group_a) is zero, so the relative change is undefined")
        else:
            relative_change = (value_b - value_a) / value_a * 100

    description = (
        f"Average of {metric} for {group_col} = {group_a!r} (group A) "
        f"compared with {group_col} = {group_b!r} (group B)"
    )
    if filters:
        description += f", restricted to rows where {filters}"

    return Finding(
        id=finding_id,
        tool="compare_group_means",
        description=description,
        values={
            "n_a": Measure(value=n_a, unit=Unit.COUNT),
            "n_b": Measure(value=n_b, unit=Unit.COUNT),
            "value_a": Measure(value=value_a, unit=value_unit),
            "value_b": Measure(value=value_b, unit=value_unit),
            "difference": Measure(value=difference, unit=difference_unit),
            "relative_change": Measure(value=relative_change, unit=Unit.PERCENT),
        },
        arguments={
            "metric": metric,
            "group_col": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "filters": filters,
        },
        caveats=caveats,
    )
