# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB connector for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

This is the reference implementation of
:class:`~gsf.connectors.base.SQLDatabase`.

Example
-------
::

    from duckdb_connector import DuckDB  # run from tabular-dev-tools/

    conn = DuckDB("./spider2.duckdb")
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations


import logging
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional

from gsf.catalog.constants import TableTypes
from gsf.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)


class DuckDBDatabase(SQLDatabase):
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: True).  Multiple
        processes can hold a read-only connection simultaneously; set to
        ``False`` only when you need to write to the file.
    """

    def __init__(self, connection_string: str, *, read_only: bool = True) -> None:
        db_path = connection_string
        if db_path.startswith("duckdb://"):
            db_path = db_path[len("duckdb://") :]
        self.conn = duckdb.connect(database=db_path, read_only=read_only)
        self._database_name: str = self.execute("SELECT current_database()").iloc[0, 0]
        logger.debug(
            "DuckDB connected (database=%r, read_only=%s).",
            self._database_name,
            read_only,
        )

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "duckdb"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return a pandas DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters.
        """
        logger.debug("DuckDB executing (→ DataFrame): %s", sql[:200])
        if parameters:
            rel = self.conn.execute(sql, parameters)
        else:
            rel = self.conn.execute(sql)
        return rel.df()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Return base tables and views with their canonical catalog type."""
        return self.execute(
            f"""
            SELECT
                table_schema,
                table_name,
                CASE table_type
                    WHEN 'VIEW' THEN '{TableTypes.VIEW}'
                    WHEN 'MATERIALIZED VIEW' THEN '{TableTypes.MATERIALIZED_VIEW}'
                    ELSE '{TableTypes.BASE_TABLE}'
                END AS table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """
        )

    def get_columns(self) -> pd.DataFrame:
        """Return all columns from information_schema as a DataFrame."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable = 'YES' AS is_nullable,
                ordinal_position
            FROM information_schema.columns
            ORDER BY table_schema, table_name, ordinal_position
        """
        )

    def get_queries(self) -> pd.DataFrame:
        """DuckDB has no built-in query history — loads sample queries from a CSV."""
        csv_path = (
            Path(__file__).parent
            / "benchmarks"
            / self._database_name
            / "sample_queries.csv"
        )
        if not csv_path.exists():
            logger.warning(
                "No sample queries CSV found at %s; returning empty DataFrame.",
                csv_path,
            )
            return pd.DataFrame(columns=["query_text", "end_time"])
        df = pd.read_csv(csv_path)
        df["end_time"] = datetime.today()
        return df

    def get_views(self) -> pd.DataFrame:
        """Return all views from information_schema."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name,
                view_definition
            FROM information_schema.views
            ORDER BY table_catalog, table_schema, table_name
        """
        )

    def get_pks(self) -> pd.DataFrame:
        """Return physical keys plus reviewed identities for governed views."""

        physical = self.execute(
            """
            SELECT
                constraints.schema_name AS table_schema,
                constraints.table_name,
                key_column.column_name,
                key_column.ordinal_position
            FROM duckdb_constraints() AS constraints,
                 UNNEST(constraints.constraint_column_names) WITH ORDINALITY
                    AS key_column(column_name, ordinal_position)
            WHERE constraints.constraint_type = 'PRIMARY KEY'
            ORDER BY constraints.schema_name,
                     constraints.table_name,
                     key_column.ordinal_position
            """
        )
        governed = self._get_governed_view_pks()
        if governed.empty:
            return physical
        return (
            pd.concat([physical, governed], ignore_index=True)
            .drop_duplicates(
                subset=["table_schema", "table_name", "column_name"],
                keep="first",
            )
            .sort_values(
                ["table_schema", "table_name", "ordinal_position"],
                ignore_index=True,
            )
        )

    def _get_governed_view_pks(self) -> pd.DataFrame:
        """Materialize contract-owned view identities as ingestion PK rows."""

        from gsf.retrieval.kumo.graph_contract import GraphContractError
        from gsf.retrieval.kumo.graph_contract import load_graph_contracts

        fields = ["table_schema", "table_name", "column_name", "ordinal_position"]
        contract = next(
            (
                item
                for item in load_graph_contracts()
                if item.database_name.casefold() == self.database_name.casefold()
            ),
            None,
        )
        if contract is None:
            return pd.DataFrame(columns=fields)

        tables = {
            (str(row.table_schema).casefold(), str(row.table_name).casefold()): str(
                row.table_type
            )
            for row in self.get_tables().itertuples(index=False)
        }
        columns = {
            (
                str(row.table_schema).casefold(),
                str(row.table_name).casefold(),
                str(row.column_name).casefold(),
            )
            for row in self.get_columns().itertuples(index=False)
        }
        rows: list[dict[str, object]] = []
        for table in contract.tables:
            path = (table.schema_name.casefold(), table.name.casefold())
            if tables.get(path, "").casefold() != TableTypes.VIEW.casefold():
                raise GraphContractError(
                    "governed DuckDB object "
                    f"{contract.database_name}.{table.schema_name}.{table.name} "
                    "must exist as a VIEW before ingestion"
                )
            for ordinal, column_name in enumerate(table.primary_key, start=1):
                if (*path, column_name.casefold()) not in columns:
                    raise GraphContractError(
                        "governed DuckDB view "
                        f"{contract.database_name}.{table.schema_name}.{table.name} "
                        f"has no primary-key column {column_name!r}"
                    )
                rows.append(
                    {
                        "table_schema": table.schema_name,
                        "table_name": table.name,
                        "column_name": column_name,
                        "ordinal_position": ordinal,
                    }
                )
        return pd.DataFrame(rows, columns=fields)

    def get_fks(self) -> pd.DataFrame:
        """Return physical foreign-key pairs, including composite keys."""

        return self.execute(
            """
            SELECT
                constraints.schema_name AS table_schema,
                constraints.table_name,
                source_column.column_name,
                constraints.schema_name AS referenced_schema,
                constraints.referenced_table,
                target_column.column_name AS referenced_column
            FROM duckdb_constraints() AS constraints,
                 UNNEST(constraints.constraint_column_names) WITH ORDINALITY
                    AS source_column(column_name, ordinal_position),
                 UNNEST(constraints.referenced_column_names) WITH ORDINALITY
                    AS target_column(column_name, ordinal_position)
            WHERE constraints.constraint_type = 'FOREIGN KEY'
              AND source_column.ordinal_position = target_column.ordinal_position
            ORDER BY constraints.schema_name,
                     constraints.table_name,
                     source_column.ordinal_position
            """
        )

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
