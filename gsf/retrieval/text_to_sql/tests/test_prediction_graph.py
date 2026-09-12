# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import GraphContractTable
from gsf.retrieval.text_to_sql.agents.prediction_graph import PredictionGraphAgent
from gsf.retrieval.text_to_sql.agents.prediction_graph import _contract_relevant_tables
from gsf.retrieval.text_to_sql.agents.prediction_graph import _enrich_relevant_tables
from gsf.catalog.constants import TableTypes


@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.fetch_table_by_name")
def test_example_table_enrichment_is_database_scoped(mock_fetch: MagicMock) -> None:
    def table(name: str, **_kwargs):
        return {
            "id": f"{name}-id",
            "name": name,
            "database_name": "prediction_db",
            "schema_name": "main",
            "pk": [f"{name[:-1]}_id"],
        }

    mock_fetch.side_effect = table

    result = _enrich_relevant_tables(
        [],
        [{"query": "PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH entities.id"}],
        database_name="prediction_db",
    )

    assert [item["name"] for item in result] == ["entities", "events"]
    assert all(item["database_name"] == "prediction_db" for item in result)
    assert mock_fetch.call_count == 2
    assert all(
        call.kwargs["database_name"] == "prediction_db"
        for call in mock_fetch.call_args_list
    )


@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.fetch_table_by_name")
def test_contract_resolution_uses_contract_primary_key_when_view_has_none(
    mock_fetch: MagicMock,
) -> None:
    mock_fetch.return_value = {
        "id": "entities-id",
        "name": "entities",
        "database_name": "prediction_db",
        "schema_name": "prediction",
        "table_type": TableTypes.VIEW,
        "pk": [],
    }
    contract = GraphContract(
        database_name="prediction_db",
        tables=(GraphContractTable("entities", "prediction", ("entity_id",), None, 2),),
        edges=(),
        revision="sha256:" + "a" * 64,
    )

    result = _contract_relevant_tables(contract)

    assert result[0]["pk"] == ["entity_id"]

    mock_fetch.assert_called_once_with(
        "entities", database_name="prediction_db", schema_name="prediction"
    )


@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.fetch_table_by_name")
def test_contract_resolution_rejects_conflicting_catalog_primary_key(
    mock_fetch: MagicMock,
) -> None:
    mock_fetch.return_value = {
        "id": "entities-id",
        "name": "entities",
        "database_name": "prediction_db",
        "schema_name": "prediction",
        "table_type": TableTypes.VIEW,
        "pk": ["wrong_id"],
    }
    contract = GraphContract(
        database_name="prediction_db",
        tables=(GraphContractTable("entities", "prediction", ("entity_id",), None, 2),),
        edges=(),
        revision="sha256:" + "a" * 64,
    )

    with pytest.raises(ValueError, match="does not match the GSF catalog"):
        _contract_relevant_tables(contract)


@patch(
    "gsf.retrieval.text_to_sql.agents.prediction_graph._contract_relevant_tables",
    side_effect=ValueError("catalog mismatch"),
)
@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.fetch_pql_examples")
@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.load_graph_contract")
def test_contract_resolution_failure_terminates_prediction_gracefully(
    mock_load_contract: MagicMock,
    _mock_fetch_examples: MagicMock,
    _mock_resolve_tables: MagicMock,
) -> None:
    mock_load_contract.return_value = MagicMock()
    state = {
        "initial_question": "Will this entity have an event?",
        "messages": [],
        "connectors": [MagicMock(database_name="prediction_db")],
        "path_state": {"target_db": "prediction_db"},
        "semantic_retriever": MagicMock(),
    }

    result = PredictionGraphAgent().execute(state)

    assert result["decision"] == "predict_failed"
    assert result["path_state"]["final_response"]["sql_code"] == ""
    assert "catalog mismatch" in result["path_state"]["formatted_response"]
    assert len(result["messages"]) == 1


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "id": "entities-id",
                "name": "entities",
                "database_name": "prediction_db",
                "schema_name": "raw",
                "table_type": TableTypes.VIEW,
                "pk": ["entity_id"],
            },
            "outside its governed catalog path",
        ),
        (
            {
                "id": "entities-id",
                "name": "entities",
                "database_name": "prediction_db",
                "schema_name": "prediction",
                "table_type": TableTypes.BASE_TABLE,
                "pk": ["entity_id"],
            },
            "must be a catalog VIEW",
        ),
    ],
)
@patch("gsf.retrieval.text_to_sql.agents.prediction_graph.fetch_table_by_name")
def test_contract_resolution_rejects_raw_or_out_of_schema_objects(
    mock_fetch: MagicMock,
    row: dict,
    message: str,
) -> None:
    mock_fetch.return_value = row
    contract = GraphContract(
        database_name="prediction_db",
        tables=(GraphContractTable("entities", "prediction", ("entity_id",), None, 2),),
        edges=(),
        revision="sha256:" + "a" * 64,
    )

    with pytest.raises(ValueError, match=message):
        _contract_relevant_tables(contract)
