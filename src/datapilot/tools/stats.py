"""Significance tests for the difference between two groups of rows.

``compare.py`` says how big a difference is. These tools say whether it is larger than
random variation would plausibly produce. Both use the same inputs as the comparison
tool and return a ``Finding``.

* chi_square_test    for a 0/1 metric (a rate), e.g. retention in Q2 versus Q3
* mann_whitney_test  for any numeric metric, e.g. revenue in two segments. It compares
                     the whole distributions and does not assume they are bell-shaped.

Conventions
-----------
* Direction matches the comparison tool: from group_a to group_b.
* p_value (Unit.P_VALUE) is the probability of seeing a difference at least this large if
  the groups were really the same. A small p-value is evidence of a difference; it says
  nothing about why, and a large p-value is not proof of "no difference".
* The significance level used to judge a p-value is recorded in ``arguments["alpha"]``,
  so a claim of "significant" can be checked against the same threshold.
* Keys of Finding.values:
      chi_square_test    n_a, n_b, value_a, value_b, statistic, p_value, effect_size
      mann_whitney_test  n_a, n_b, median_a, median_b, statistic, p_value, effect_size
* effect_size (Unit.NUMBER) is how big the difference is, independent of sample size:
      chi_square_test    phi coefficient, 0 (no link) to 1 (perfect link). Always >= 0.
      mann_whitney_test  rank-biserial correlation, -1 to 1: the chance that a random
                         group_b value is higher than a random group_a value, minus the
                         chance that it is lower. Positive means group_b tends to be higher.
* If the test cannot be computed (every row has the same outcome), statistic, p_value
  and effect_size are None and a caveat says why.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

from datapilot.models import Finding, Measure, Unit
from datapilot.tools._shared import (
    MIN_GROUP_SIZE,
    is_binary_metric,
    prepare_two_groups,
    require_numeric,
)

SIGNIFICANCE_LEVEL = 0.05  # recorded in each finding's arguments as "alpha"
MIN_EXPECTED_COUNT = 5  # chi-square is unreliable if any expected cell count is below this


def _usable_values(rows: pd.DataFrame, metric: str, group: Any) -> pd.Series:
    """Return the group's non-missing metric values; raise if there are none."""
    values = rows[metric].dropna()
    if len(values) == 0:
        raise ValueError(f"Group {group!r} has no non-missing values of {metric!r}")
    return values


def _standard_caveats(group_a: Any, n_a: int, group_b: Any, n_b: int) -> list[str]:
    caveats = []
    for group, n in ((group_a, n_a), (group_b, n_b)):
        if n < MIN_GROUP_SIZE:
            caveats.append(f"Small sample: group {group!r} has only {n} rows")
    caveats.append(
        "This is an association between the groups and the outcome; it does not establish a cause"
    )
    return caveats


def _arguments(metric, group_col, group_a, group_b, filters) -> dict[str, Any]:
    return {
        "metric": metric,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "filters": filters,
        "alpha": SIGNIFICANCE_LEVEL,
    }


def chi_square_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    finding_id: str,
    filters: dict[str, Any] | None = None,
) -> Finding:
    """Test whether the rate of a 0/1 ``metric`` differs between two groups.

    Args:
        df: the prepared DataFrame.
        metric: a column containing only 0 and 1 (for example retained_90d).
        group_col: column that defines the groups, e.g. "signup_quarter".
        group_a: the baseline group.
        group_b: the group compared to the baseline.
        finding_id: id for the returned Finding, supplied by the caller.
        filters: optional {column: value} pairs; only rows matching all of them are used.

    Raises:
        ValueError: for an unknown column, identical groups, a metric that is not a 0/1
            column, a group with no rows, or a group with no non-missing metric values.
    """
    rows_a, rows_b = prepare_two_groups(df, metric, group_col, group_a, group_b, filters)
    require_numeric(df, metric)
    if not is_binary_metric(df[metric]):
        raise ValueError(f"{metric!r} must contain only 0 and 1 for a chi-square test")
    filters = filters or {}

    values_a = _usable_values(rows_a, metric, group_a)
    values_b = _usable_values(rows_b, metric, group_b)
    n_a, n_b = len(values_a), len(values_b)

    # 2x2 table of counts: one row per group, columns are outcome 0 and outcome 1.
    table = [
        [int((values_a == 0).sum()), int((values_a == 1).sum())],
        [int((values_b == 0).sum()), int((values_b == 1).sum())],
    ]
    column_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]]

    caveats = _standard_caveats(group_a, n_a, group_b, n_b)
    statistic = p_value = effect_size = None
    if 0 in column_totals:
        caveats.append("Every row has the same outcome, so the chi-square test is undefined")
    else:
        chi2, p, _, expected = scipy_stats.chi2_contingency(table)
        statistic, p_value = float(chi2), float(p)
        effect_size = math.sqrt(statistic / (n_a + n_b))
        if (expected < MIN_EXPECTED_COUNT).any():
            caveats.append(
                f"Some expected cell counts are below {MIN_EXPECTED_COUNT}, "
                "so the chi-square result may be unreliable"
            )

    description = (
        f"Chi-square test of whether the rate of {metric} differs between "
        f"{group_col} = {group_a!r} and {group_col} = {group_b!r}"
    )
    if filters:
        description += f", restricted to rows where {filters}"

    return Finding(
        id=finding_id,
        tool="chi_square_test",
        description=description,
        values={
            "n_a": Measure(value=n_a, unit=Unit.COUNT),
            "n_b": Measure(value=n_b, unit=Unit.COUNT),
            "value_a": Measure(value=float(values_a.mean()), unit=Unit.PROPORTION),
            "value_b": Measure(value=float(values_b.mean()), unit=Unit.PROPORTION),
            "statistic": Measure(value=statistic, unit=Unit.NUMBER),
            "p_value": Measure(value=p_value, unit=Unit.P_VALUE),
            "effect_size": Measure(value=effect_size, unit=Unit.NUMBER),
        },
        arguments=_arguments(metric, group_col, group_a, group_b, filters),
        caveats=caveats,
    )


def mann_whitney_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    finding_id: str,
    filters: dict[str, Any] | None = None,
) -> Finding:
    """Test whether a numeric ``metric`` tends to be higher in one group than the other.

    Uses the two-sided Mann-Whitney U test. The reported statistic is U for group_a.

    Args and Raises: as for ``chi_square_test``, except the metric only has to be numeric.
    """
    rows_a, rows_b = prepare_two_groups(df, metric, group_col, group_a, group_b, filters)
    require_numeric(df, metric)
    filters = filters or {}

    values_a = _usable_values(rows_a, metric, group_a)
    values_b = _usable_values(rows_b, metric, group_b)
    n_a, n_b = len(values_a), len(values_b)

    result = scipy_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    u_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    effect_size = 1 - 2 * u_statistic / (n_a * n_b)  # positive: group_b tends to be higher

    description = (
        f"Mann-Whitney test of whether {metric} differs between "
        f"{group_col} = {group_a!r} and {group_col} = {group_b!r}"
    )
    if filters:
        description += f", restricted to rows where {filters}"

    return Finding(
        id=finding_id,
        tool="mann_whitney_test",
        description=description,
        values={
            "n_a": Measure(value=n_a, unit=Unit.COUNT),
            "n_b": Measure(value=n_b, unit=Unit.COUNT),
            "median_a": Measure(value=float(values_a.median()), unit=Unit.NUMBER),
            "median_b": Measure(value=float(values_b.median()), unit=Unit.NUMBER),
            "statistic": Measure(value=u_statistic, unit=Unit.NUMBER),
            "p_value": Measure(value=p_value, unit=Unit.P_VALUE),
            "effect_size": Measure(value=effect_size, unit=Unit.NUMBER),
        },
        arguments=_arguments(metric, group_col, group_a, group_b, filters),
        caveats=_standard_caveats(group_a, n_a, group_b, n_b),
    )
