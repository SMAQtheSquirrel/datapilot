"""run_sql: run a read-only SQL query on the dataset and return the result as a Finding.

The planner uses this for questions the fixed tools do not cover, such as "average
revenue by region". The DataFrame is exposed to the query as a table called ``data``
(``TABLE_NAME``).

Safety layers (each one limits the damage if another fails)
-----------------------------------------------------------
1. Parsing: ``validate_query`` uses sqlglot and accepts exactly one SELECT-type statement
   (SELECT, WITH ... SELECT, UNION / EXCEPT / INTERSECT). Everything else (DROP, INSERT,
   COPY, ATTACH, PRAGMA, several statements at once) is rejected before DuckDB sees it.
2. A fresh in-memory DuckDB database per query that holds nothing but the DataFrame.
   There is nothing else to read or change.
3. External access is switched off (``enable_external_access = false``), so functions
   such as read_csv or read_parquet cannot touch files or the network, and the
   configuration is then locked so the query cannot switch it back on.
4. Resource limits: a memory limit, a row limit on the result, and a query timeout.

Threat model, honestly: these settings are defence in depth. DuckDB's own security
documentation says they are not a substitute for sandboxing and that untrusted SQL should
run in an isolated environment. Running the whole service in a restricted container is a
later milestone. Until then, treat the SQL as coming from the LLM planner, not from
untrusted end users.

Conventions
-----------
* Finding.table holds the result rows, one dict per row, keyed by column name.
* Finding.values holds n_rows (Unit.COUNT): the number of rows in the table.
* If the result has more than MAX_ROWS rows, only the first MAX_ROWS are kept and a
  caveat says so.
* Decimals become floats and NaN becomes None, so the table is JSON-safe.
* Every result column needs a distinct name; use AS to rename.
"""

from __future__ import annotations

import math
import threading
from decimal import Decimal
from typing import Any

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from datapilot.models import Finding, Measure, Unit

TABLE_NAME = "data"  # the name the DataFrame has inside queries
MAX_ROWS = 1000  # result rows kept; more than this are cut off
QUERY_TIMEOUT_SECONDS = 10.0
MEMORY_LIMIT = "1GB"

# The statement types that only read data.
ALLOWED_STATEMENTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)


def validate_query(query: str) -> None:
    """Raise ValueError unless ``query`` is a single read-only SELECT-type statement."""
    if not query or not query.strip():
        raise ValueError("The SQL query is empty")

    try:
        parsed = sqlglot.parse(query, read="duckdb")
    except SqlglotError as error:
        raise ValueError(f"The SQL could not be parsed: {error}") from error

    # A trailing semicolon can produce an empty entry, so drop those.
    statements = [statement for statement in parsed if statement is not None]
    if len(statements) != 1:
        raise ValueError(f"Exactly one SQL statement is allowed, got {len(statements)}")
    if not isinstance(statements[0], ALLOWED_STATEMENTS):
        raise ValueError("Only read-only SELECT queries are allowed")


def _clean_cell(value: Any) -> Any:
    """Make a result value JSON-safe: Decimal -> float, NaN -> None."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run_sql(df: pd.DataFrame, query: str, finding_id: str) -> Finding:
    """Run a read-only SQL ``query`` against ``df`` (available as the table ``data``).

    Args:
        df: the prepared DataFrame.
        query: a single SELECT, WITH ... SELECT, or set-operation query.
        finding_id: id for the returned Finding, supplied by the caller.

    Raises:
        ValueError: if the query is not a single read-only statement, fails to run,
            exceeds the timeout, or returns two columns with the same name.
    """
    validate_query(query)

    connection = duckdb.connect(":memory:")
    timed_out = threading.Event()

    def stop_query() -> None:
        timed_out.set()
        connection.interrupt()

    timer = threading.Timer(QUERY_TIMEOUT_SECONDS, stop_query)
    try:
        connection.register(TABLE_NAME, df)
        connection.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        connection.execute("SET enable_external_access = false")
        connection.execute("SET lock_configuration = true")

        timer.start()
        cursor = connection.execute(query)
        columns = [column[0] for column in cursor.description]
        fetched = cursor.fetchmany(MAX_ROWS + 1)  # one extra row tells us if we cut anything
    except Exception as error:
        if timed_out.is_set():
            raise ValueError(
                f"The query took longer than {QUERY_TIMEOUT_SECONDS} seconds and was stopped"
            ) from error
        raise ValueError(f"The SQL query failed: {error}") from error
    finally:
        timer.cancel()
        connection.close()

    if len(set(columns)) != len(columns):
        raise ValueError(
            f"The result has duplicate column names {columns}; give each column its own alias with AS"
        )

    truncated = len(fetched) > MAX_ROWS
    table = [
        {column: _clean_cell(value) for column, value in zip(columns, row)}
        for row in fetched[:MAX_ROWS]
    ]

    caveats = []
    if truncated:
        caveats.append(f"The result was cut off: only the first {MAX_ROWS} rows are shown")

    return Finding(
        id=finding_id,
        tool="run_sql",
        description=f"Result of a SQL query on the dataset ({len(table)} rows)",
        values={"n_rows": Measure(value=len(table), unit=Unit.COUNT)},
        table=table,
        arguments={"query": query, "max_rows": MAX_ROWS},
        caveats=caveats,
    )
