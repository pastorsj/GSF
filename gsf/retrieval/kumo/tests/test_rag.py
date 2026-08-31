# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

from gsf.retrieval.kumo.rag import fetch_pql_examples


@patch("gsf.retrieval.kumo.rag.fetch_pql_analyses_by_ids")
@patch("gsf.retrieval.kumo.rag.search_semantic_index")
def test_pql_examples_are_filtered_and_refetched_in_database_scope(
    mock_search: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    mock_search.return_value = [{"id": "same-db"}, {"id": "other-db"}]
    mock_fetch.return_value = {
        "same-db": {
            "id": "same-db",
            "database_name": "prediction_db",
            "name": "Predict delay",
            "description": "Reviewed example",
            "pql": "PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH entities.entity_id",
        }
    }

    examples = fetch_pql_examples(object(), "Which entities will be delayed?", "prediction_db", k=3)

    assert mock_search.call_args.kwargs["database_name"] == "prediction_db"
    mock_fetch.assert_called_once_with(["same-db", "other-db"], database_name="prediction_db")
    assert examples == [
        {
            "id": "same-db",
            "database_name": "prediction_db",
            "question": "Predict delay",
            "query": "PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH entities.entity_id",
            "reasoning": "Reviewed example",
        }
    ]
