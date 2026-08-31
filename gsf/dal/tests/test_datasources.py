# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

from gsf.dal.datasources import fetch_table_by_name


@patch("gsf.dal.datasources.graph")
def test_fetch_table_by_name_passes_database_and_schema_scope(
    mock_graph: MagicMock,
) -> None:
    row = {
        "id": "table-id",
        "name": "events",
        "database_name": "prediction_db",
        "schema_name": "main",
    }
    mock_graph.return_value.query_read.return_value = [row]

    assert fetch_table_by_name("events", database_name="prediction_db", schema_name="main") == row
    assert mock_graph.return_value.query_read.call_args.args[1] == {
        "name": "events",
        "database_name": "prediction_db",
        "schema_name": "main",
    }


@patch("gsf.dal.datasources.graph")
def test_scoped_table_lookup_rejects_ambiguity(mock_graph: MagicMock) -> None:
    mock_graph.return_value.query_read.return_value = [
        {"id": "one"},
        {"id": "two"},
    ]

    assert fetch_table_by_name("events", database_name="prediction_db") is None
