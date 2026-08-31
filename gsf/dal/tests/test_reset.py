# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from gsf.dal.reset import retire_database_alias


@patch("gsf.dal.reset.get_semantic_vdb")
@patch("gsf.dal.reset.get_data_vdb")
@patch("gsf.dal.reset.get_neo4j_conn")
def test_retire_database_alias_deletes_only_proven_shared_alias(
    mock_conn: MagicMock,
    mock_data_vdb: MagicMock,
    mock_semantic_vdb: MagicMock,
) -> None:
    graph = mock_conn.return_value
    graph.query_read.side_effect = [[{"id": "legacy-db"}], []]
    mock_data_vdb.return_value.delete_by_database.return_value = ["data-1"]
    mock_semantic_vdb.return_value.delete_by_database.return_value = ["semantic-1"]

    result = retire_database_alias(
        "ai_factory_prediction",
        successor_database_name="ai_factory",
    )

    assert result.catalog_nodes == 1
    assert result.data_rows == 1
    assert result.semantic_rows == 1
    delete_call = graph.query_write.call_args
    assert "DETACH DELETE db" in delete_call.kwargs["query"]
    assert delete_call.kwargs["parameters"] == {"database_name": "ai_factory_prediction"}
    mock_data_vdb.return_value.delete_by_database.assert_called_once_with("ai_factory_prediction")
    mock_semantic_vdb.return_value.delete_by_database.assert_called_once_with("ai_factory_prediction")


@patch("gsf.dal.reset.get_semantic_vdb")
@patch("gsf.dal.reset.get_data_vdb")
@patch("gsf.dal.reset.get_neo4j_conn")
def test_retire_database_alias_fails_before_deleting_unmigrated_schema(
    mock_conn: MagicMock,
    mock_data_vdb: MagicMock,
    mock_semantic_vdb: MagicMock,
) -> None:
    graph = mock_conn.return_value
    graph.query_read.side_effect = [
        [{"id": "legacy-db"}],
        [{"id": "legacy-schema", "name": "main"}],
    ]

    with pytest.raises(RuntimeError, match="have not migrated.*main"):
        retire_database_alias(
            "ai_factory_prediction",
            successor_database_name="ai_factory",
        )

    graph.query_write.assert_not_called()
    mock_data_vdb.assert_not_called()
    mock_semantic_vdb.assert_not_called()


@patch("gsf.dal.reset.get_semantic_vdb")
@patch("gsf.dal.reset.get_data_vdb")
@patch("gsf.dal.reset.get_neo4j_conn")
def test_retire_database_alias_cleans_stale_vectors_when_catalog_alias_is_absent(
    mock_conn: MagicMock,
    mock_data_vdb: MagicMock,
    mock_semantic_vdb: MagicMock,
) -> None:
    mock_conn.return_value.query_read.return_value = []
    mock_data_vdb.return_value.delete_by_database.return_value = ["data-1"]
    mock_semantic_vdb.return_value.delete_by_database.return_value = []

    result = retire_database_alias(
        "ai_factory_prediction",
        successor_database_name="ai_factory",
    )

    assert result.catalog_nodes == 0
    assert result.data_rows == 1
    assert result.semantic_rows == 0
    mock_conn.return_value.query_write.assert_not_called()
