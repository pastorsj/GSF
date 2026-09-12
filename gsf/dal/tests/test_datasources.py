# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from gsf.dal.datasources import fetch_table_by_name


@patch("gsf.dal.datasources.store")
def test_fetch_table_by_name_passes_database_and_schema_scope(
    mock_store,
) -> None:
    row = {
        "id": "table-id",
        "name": "events",
        "database_name": "prediction_db",
        "schema_name": "main",
        "table_type": "VIEW",
        "description": "Governed prediction events",
        "pk": ["event_id"],
    }
    mock_store.return_value.query_read.return_value = [row]

    assert fetch_table_by_name("events", database_name="prediction_db", schema_name="main") == row
    statement = str(mock_store.return_value.query_read.call_args.args[0])
    assert "catalog_database.name" in statement
    assert "catalog_schema.name" in statement
    assert "LIMIT" in statement


@patch("gsf.dal.datasources.store")
def test_scoped_table_lookup_rejects_ambiguity(mock_store) -> None:
    mock_store.return_value.query_read.return_value = [
        {"id": "one"},
        {"id": "two"},
    ]

    assert fetch_table_by_name("events", database_name="prediction_db") is None
