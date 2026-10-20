from pydantic import BaseModel
from enum import Enum

class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BINARY = "binary"
    IDENTIFIER = "identifier"

class NumericStats(BaseModel):
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    quantiles: dict[str, float] = {}
    outlier_count: int = 0

from datetime import datetime


class ValueCount(BaseModel):
    value: str
    count: int
    share: float  # fraction of non-missing rows, 0 to 1


class CategoricalStats(BaseModel):
    top_values: list[ValueCount] = []


class DatetimeStats(BaseModel):
    earliest: datetime | None = None
    latest: datetime | None = None
    span_days: float | None = None


class BinaryStats(BaseModel):
    positive_rate: float | None = None
    positive_value: int | float | str
    negative_value: int | float | str


class ColumnProfile(BaseModel):
    # every column
    name: str
    kind: ColumnKind
    missing_count: int
    missing_fraction: float
    unique_count: int

    # kind-specific blocks
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    datetime: DatetimeStats | None = None
    binary: BinaryStats | None = None
    
class DatasetProfile(BaseModel):
    name:str
    memory_size:float
    row_count:int
    column_profiles:list[ColumnProfile]
    
