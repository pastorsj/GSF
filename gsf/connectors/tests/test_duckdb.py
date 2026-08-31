# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import duckdb
import pytest
from gsf.connectors.duckdb import DuckDBDatabase
from gsf.retrieval.kumo.graph_contract import GraphContractError
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes


def _write_graph_contract(path, database_name: str, *, table_name: str = "events") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "database_name": database_name,
                "database_sha256": "0" * 64,
                "source": {
                    "dataset_id": "duckdb-test",
                    "dataset_version": "1",
                    "revision": "test",
                    "manifest_sha256": "1" * 64,
                    "prediction_manifest_sha256": "2" * 64,
                },
                "object_count": 1,
                "row_count": 0,
                "tables": [
                    {
                        "name": table_name,
                        "schema_name": "prediction",
                        "rows": 0,
                        "primary_key": ["event_id"],
                        "time_column": None,
                    }
                ],
                "relationships": [],
                "time_columns": {table_name: None},
                "forbidden_tables": [],
            }
        ),
        encoding="utf-8",
    )


def test_tables_enumerate_governed_views_with_catalog_type(tmp_path) -> None:
    path = tmp_path / "views.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE SCHEMA raw")
    connection.execute("CREATE TABLE raw.events (event_id INTEGER)")
    connection.execute("CREATE SCHEMA prediction")
    connection.execute("CREATE VIEW prediction.events AS SELECT event_id FROM raw.events")
    connection.close()

    database = DuckDBDatabase(str(path))
    try:
        tables = database.get_tables().to_dict(orient="records")
    finally:
        database.close()

    assert tables == [
        {
            "table_schema": "prediction",
            "table_name": "events",
            "table_type": TableTypes.VIEW,
        },
        {
            "table_schema": "raw",
            "table_name": "events",
            "table_type": TableTypes.BASE_TABLE,
        },
    ]


def test_primary_and_foreign_keys_include_composite_column_pairs(tmp_path) -> None:
    path = tmp_path / "catalog.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE parents (account_id INTEGER, region VARCHAR, PRIMARY KEY (account_id, region))")
    connection.execute(
        "CREATE TABLE events (event_id INTEGER PRIMARY KEY, account_id INTEGER, "
        "region VARCHAR, FOREIGN KEY (account_id, region) "
        "REFERENCES parents(account_id, region))"
    )
    connection.close()

    database = DuckDBDatabase(str(path))
    try:
        pks = database.get_pks().to_dict(orient="records")
        fks = database.get_fks().to_dict(orient="records")
    finally:
        database.close()

    assert pks == [
        {
            "table_schema": "main",
            "table_name": "events",
            "column_name": "event_id",
            "ordinal_position": 1,
        },
        {
            "table_schema": "main",
            "table_name": "parents",
            "column_name": "account_id",
            "ordinal_position": 1,
        },
        {
            "table_schema": "main",
            "table_name": "parents",
            "column_name": "region",
            "ordinal_position": 2,
        },
    ]
    assert fks == [
        {
            "table_schema": "main",
            "table_name": "events",
            "column_name": "account_id",
            "referenced_schema": "main",
            "referenced_table": "parents",
            "referenced_column": "account_id",
        },
        {
            "table_schema": "main",
            "table_name": "events",
            "column_name": "region",
            "referenced_schema": "main",
            "referenced_table": "parents",
            "referenced_column": "region",
        },
    ]


def test_primary_keys_combine_physical_constraints_and_reviewed_view_identity(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "catalog.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE raw_events (event_id INTEGER PRIMARY KEY)")
    connection.execute("CREATE SCHEMA prediction")
    connection.execute("CREATE VIEW prediction.events AS SELECT event_id FROM raw_events WHERE false")
    connection.close()
    contract_path = tmp_path / "prediction.graph.json"
    _write_graph_contract(contract_path, "catalog")
    monkeypatch.setenv("KUMO_GRAPH_CONTRACTS_FILE", str(contract_path))

    database = DuckDBDatabase(str(path))
    try:
        pks = database.get_pks().to_dict(orient="records")
    finally:
        database.close()

    assert pks == [
        {
            "table_schema": "main",
            "table_name": "raw_events",
            "column_name": "event_id",
            "ordinal_position": 1,
        },
        {
            "table_schema": "prediction",
            "table_name": "events",
            "column_name": "event_id",
            "ordinal_position": 1,
        },
    ]


def test_reviewed_view_primary_key_fails_closed_when_contract_targets_table(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "catalog.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE SCHEMA prediction")
    connection.execute("CREATE TABLE prediction.events (event_id INTEGER PRIMARY KEY)")
    connection.close()
    contract_path = tmp_path / "prediction.graph.json"
    _write_graph_contract(contract_path, "catalog")
    monkeypatch.setenv("KUMO_GRAPH_CONTRACTS_FILE", str(contract_path))

    database = DuckDBDatabase(str(path))
    try:
        with pytest.raises(GraphContractError, match="must exist as a VIEW"):
            database.get_pks()
    finally:
        database.close()
