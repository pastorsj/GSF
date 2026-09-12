# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

from gsf.server.pql_analyses.service import create_pql_analysis


@patch("gsf.server.pql_analyses.service.uuid.uuid4", return_value="analysis-id")
@patch("gsf.server.pql_analyses.service._embed")
@patch("gsf.server.pql_analyses.service.upsert_pql_analysis_node")
@patch("gsf.server.pql_analyses.service.find_pql_analysis_by_pql")
@patch("gsf.server.pql_analyses.service.find_pql_analysis_by_name")
def test_create_scopes_conflicts_storage_and_response_to_database(
    mock_name: MagicMock,
    mock_pql: MagicMock,
    mock_upsert: MagicMock,
    mock_embed: MagicMock,
    _mock_uuid: MagicMock,
) -> None:
    mock_name.return_value = None
    mock_pql.return_value = None

    result = create_pql_analysis(
        "prediction_db",
        "Predict delay",
        "Reviewed",
        "PREDICT delay FOR entities",
    )

    mock_name.assert_called_once_with("Predict delay", exclude_id=None, database_name="prediction_db")
    mock_pql.assert_called_once_with(
        "PREDICT delay FOR entities",
        exclude_id=None,
        database_name="prediction_db",
    )
    mock_upsert.assert_called_once_with(
        "analysis-id",
        "Predict delay",
        "Reviewed",
        "PREDICT delay FOR entities",
        database_name="prediction_db",
    )
    mock_embed.assert_called_once_with("analysis-id", "prediction_db")
    assert result["database_name"] == "prediction_db"
