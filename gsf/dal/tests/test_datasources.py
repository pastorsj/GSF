# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

from gsf.dal.datasources import fetch_table_by_name


@patch("gsf.dal.datasources.store")
def test_fetch_table_by_name_passes_database_and_schema_scope(
    mock_store: MagicMock,
) -> None:
    row = {
        "id": "table-id",
        "name": "events",
        "database_name": "prediction_db",
        "schema_name": "main",
    }
    mock_store.return_value.query_read.return_value = [row]

    assert (
        fetch_table_by_name("events", database_name="prediction_db", schema_name="main")
        == row
    )
    statement = mock_store.return_value.query_read.call_args.args[0]
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "catalog_database.name = 'prediction_db'" in sql
    assert "catalog_schema.name = 'main'" in sql
    assert "catalog_table.name = 'events'" in sql


@patch("gsf.dal.datasources.store")
def test_scoped_table_lookup_rejects_ambiguity(mock_store: MagicMock) -> None:
    mock_store.return_value.query_read.return_value = [
        {"id": "one"},
        {"id": "two"},
    ]

    assert fetch_table_by_name("events", database_name="prediction_db") is None


@patch("gsf.dal.datasources.store")
def test_unscoped_table_lookup_preserves_its_response_shape(
    mock_store: MagicMock,
) -> None:
    mock_store.return_value.query_read.return_value = [
        {
            "id": "table-id",
            "name": "events",
            "database_name": "prediction_db",
            "schema_name": "main",
            "table_type": "view",
            "description": "Events",
            "pk": ["event_id"],
        }
    ]

    assert fetch_table_by_name("events") == {
        "id": "table-id",
        "name": "events",
        "schema_name": "main",
        "description": "Events",
        "pk": ["event_id"],
    }
