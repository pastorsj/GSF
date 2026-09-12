# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static containment checks for model-authored SQL."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gsf.retrieval.text_to_sql.agents.sql_parse_validation import (
    SQLValidationAgent,
    generated_sql_safety_error,
)


@dataclass
class _Schema:
    tables: set[str]

    def table_exists(self, name: str) -> bool:
        return name.casefold() in self.tables


SCHEMAS = {"main": _Schema({"orders"})}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_csv_auto('/tmp/outside.csv')",
        "SELECT * FROM glob('/query-claw-active/documents/*')",
        (
            "SELECT orders.id, files.content FROM orders "
            "JOIN read_text('/etc/passwd') AS files ON TRUE"
        ),
        (
            "SELECT orders.id, files.content FROM orders, "
            "LATERAL read_text('/etc/passwd') AS files"
        ),
    ],
)
def test_rejects_external_table_functions(sql: str) -> None:
    result = SQLValidationAgent._sql_parse_validation(SCHEMAS, sql, ["duckdb"])

    assert "table-valued functions or external relations" in result["error"]
    assert result["another_try"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM '/etc/passwd'",
        "SELECT orders.id FROM orders JOIN '/tmp/outside.csv' AS files ON TRUE",
    ],
)
def test_rejects_duckdb_replacement_scans(sql: str) -> None:
    error = generated_sql_safety_error(sql, SCHEMAS, ["duckdb"])

    assert "not a table in the governed catalog" in error


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders; SELECT * FROM read_text('/etc/passwd')",
        "COPY (SELECT * FROM orders) TO '/tmp/outside.csv'",
        "ATTACH '/tmp/outside.duckdb' AS outside",
    ],
)
def test_rejects_multiple_or_non_query_statements(sql: str) -> None:
    error = generated_sql_safety_error(sql, SCHEMAS, ["duckdb"])

    assert "exactly one read-only query" in error


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders",
        "WITH recent AS (SELECT * FROM orders) SELECT COUNT(*) FROM recent",
        "SELECT orders.id FROM orders, UNNEST([1, 2]) AS item(value)",
    ],
)
def test_accepts_queries_derived_only_from_governed_tables(sql: str) -> None:
    assert generated_sql_safety_error(sql, SCHEMAS, ["duckdb"]) == ""
