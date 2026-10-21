import pandas as pd

from datapilot.models import (
    BinaryStats,
    CategoricalStats,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    DatetimeStats,
    NumericStats,
    ValueCount,
)

# --- Tunable thresholds (named constants so tests and docs can refer to them) ---
IDENTIFIER_UNIQUE_RATIO = 0.95  # unique/non-missing >= this -> likely an identifier
MAX_CATEGORICAL_UNIQUE = 50  # more distinct text values than this -> not categorical
TOP_VALUES_LIMIT = 10  # how many top categories to report
OUTLIER_IQR_MULTIPLIER = 1.5  # IQR rule for flagging outliers
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)  # quantiles reported for numeric columns

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:

    column_profiles = []
    for column in df.columns:
        column_profiles.append(_profile_column(df[column]))

    return DatasetProfile(
        memory_size=float(df.memory_usage(deep=True).sum()),
        row_count=int(df.shape[0]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_cells=int(df.isna().sum().sum()),
        column_profiles=column_profiles,
    )

def _profile_column(series: pd.Series) -> ColumnProfile:
    kind = _classify_column(series)
    missing_count = int(series.isna().sum())
    length = len(series)

    blocks = {}
    if kind == ColumnKind.DATETIME:
        blocks["datetime"] = _datetime_stats(series)
    elif kind == ColumnKind.BINARY:
        blocks["binary"] = _binary_stats(series)
    elif kind == ColumnKind.NUMERIC:
        blocks["numeric"] = _numeric_stats(series)
    elif kind == ColumnKind.CATEGORICAL:
        blocks["categorical"] = _categorical_stats(series)


    return ColumnProfile(
        name=str(series.name),
        kind=kind,
        missing_count=missing_count,
        missing_fraction=missing_count / length if length > 0 else 0.0,
        unique_count=int(series.nunique()),
        **blocks,
    )

def _classify_column(series: pd.Series) -> ColumnKind:
    values = series.dropna()

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME

    # An empty or all-missing column has nothing to measure. Treat it as categorical.
    if len(values) == 0:
        return ColumnKind.CATEGORICAL

    n_unique = values.nunique()
    if n_unique == 2:
        return ColumnKind.BINARY

    # Identifiers: integers or text where nearly every value is different.
    # Floats are excluded, because continuous measurements are often all-unique too.
    is_float = pd.api.types.is_float_dtype(series)
    if not is_float and n_unique / len(values) >= IDENTIFIER_UNIQUE_RATIO:
        return ColumnKind.IDENTIFIER

    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.NUMERIC

    return ColumnKind.CATEGORICAL

def _to_float(value) -> float | None:
    """Convert a NumPy number to a plain float, and NaN or None to None."""
    if value is None or pd.isna(value):
        return None
    return float(value)

def _numeric_stats(series: pd.Series) -> NumericStats:
    values = series.dropna()

    quantiles = {}
    outlier_count = 0
    if len(values) > 0:
        quantiles = {
            f"p{round(q * 100):02d}": float(v)
            for q, v in values.quantile(list(QUANTILES)).items()
        }
        iqr = quantiles["p75"] - quantiles["p25"]
        lower = quantiles["p25"] - OUTLIER_IQR_MULTIPLIER * iqr
        upper = quantiles["p75"] + OUTLIER_IQR_MULTIPLIER * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())

    return NumericStats(
        mean=_to_float(values.mean()),
        median=_to_float(values.median()),
        std=_to_float(values.std()),
        min_value=_to_float(values.min()),
        max_value=_to_float(values.max()),
        quantiles=quantiles,
        outlier_count=outlier_count,
    )


def _categorical_stats(series: pd.Series) -> CategoricalStats:
    values = series.dropna()
    total = len(values)

    top_values = []
    for value, count in values.value_counts().head(TOP_VALUES_LIMIT).items():
        top_values.append(
            ValueCount(
                value=str(value),
                count=int(count),
                share=float(count / total),
            )
        )
    return CategoricalStats(top_values=top_values)


def _datetime_stats(series: pd.Series) -> DatetimeStats:
    values = series.dropna()
    if len(values) == 0:
        return DatetimeStats()  # all fields default to None

    earliest = values.min()
    latest = values.max()
    return DatetimeStats(
        earliest=earliest.to_pydatetime(),
        latest=latest.to_pydatetime(),
        span_days=(latest - earliest).total_seconds() / 86400,
    )

def _to_python(value):
    """Convert a NumPy scalar (np.int64 etc.) to a plain Python value."""
    return value.item() if hasattr(value, "item") else value

def _binary_stats(series: pd.Series) -> BinaryStats:
    values = series.dropna()
    unique_values = sorted(values.unique())
    if len(unique_values) != 2:
        raise ValueError(
            f"_binary_stats needs exactly 2 distinct values, got {len(unique_values)}"
        )
    negative, positive = unique_values  # sorted, so the larger value is "positive"

    return BinaryStats(
        positive_rate=float((values == positive).mean()),
        positive_value=_to_python(positive),
        negative_value=_to_python(negative),
    )