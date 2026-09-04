from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
from gsf.retrieval.kumo.canary import run_canary
from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import GraphContractTable
from gsf.retrieval.kumo.predictor import PredictionContext
from gsf.retrieval.kumo.provider import KumoProviderReadiness


def _context(model: MagicMock) -> PredictionContext:
    return PredictionContext(
        kumo_model=model,
        connector=MagicMock(),
        graph_ddl="entities(entity_id ID)",
        graph_edges=[],
        graph_col_stypes={},
        column_reference="",
        time_columns={"entities": None},
        table_names={"entities": '"prediction"."entities"'},
        entity_ids={"entities": [53, 54]},
        examples=[],
        database_name="ai_factory",
        graph_receipt={
            "mode": "explicit",
            "database_name": "ai_factory",
            "contract_revision": "sha256:" + "a" * 64,
            "graph_revision": "sha256:" + "b" * 64,
            "tables": [
                {
                    "name": "entities",
                    "schema_name": "prediction",
                    "primary_key": ["entity_id"],
                    "time_column": None,
                    "loaded_rows": 2,
                }
            ],
            "edges": [],
        },
    )


def test_direct_canary_repeats_fixed_population_and_emits_only_typed_proof() -> None:
    anchor = pd.Timestamp("2026-06-06T00:00:00Z")
    model = MagicMock()
    model.predict.return_value = pd.DataFrame(
        {
            "ENTITY": [53, 54],
            "ANCHOR_TIMESTAMP": [anchor, anchor],
            "FALSE_PROB": [0.8, 0.2],
            "TRUE_PROB": [0.2, 0.8],
        }
    )
    contract = GraphContract(
        database_name="ai_factory",
        tables=(GraphContractTable("entities", "prediction", ("entity_id",), None, 2),),
        edges=(),
        revision="sha256:" + "a" * 64,
    )
    connector = MagicMock(database_name="ai_factory")
    readiness = KumoProviderReadiness(
        status="ready",
        ready=True,
        expected_model="kumo-relational",
        advertised_models=("kumo-relational",),
        nvidia_sdfm_version="0.3.0",
        kumorfm_version="2.29.0",
    )
    with (
        patch("gsf.retrieval.kumo.canary.load_graph_contract", return_value=contract),
        patch("gsf.retrieval.kumo.canary.DuckDBDatabase", return_value=connector),
        patch("gsf.retrieval.kumo.canary.build_prediction_context", return_value=_context(model)),
        patch("gsf.retrieval.kumo.canary.current_provider_readiness", return_value=readiness),
    ):
        receipt = run_canary(
            database_path="/booth-data/ai_factory.duckdb",
            database_name="ai_factory",
            graph_contracts_path="/booth-config/prediction-graphs.bundle.json",
            pql="PREDICT COUNT(events.*, 0, 60, days) > 0 FOR EACH entities.entity_id",
            entities=[53, 54],
            anchor="2026-06-06T00:00:00Z",
            score_column="TRUE_PROB",
            repetitions=3,
        )

    assert receipt["status"] == "passed"
    assert receipt["population"]["count"] == 2
    assert receipt["population"]["type_counts"] == {"int": 2}
    assert len(receipt["repetitions"]) == 3
    assert all(attempt["passed"] for attempt in receipt["repetitions"])
    assert model.predict.call_count == 3
    model.validate_pql.assert_called_once()
    connector.close.assert_called_once()
    rendered = str(receipt)
    assert "private" not in rendered
    assert "'value': 53" not in rendered
    assert "'value': 54" not in rendered
