from dataclasses import dataclass
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import GraphContractEdge
from gsf.retrieval.kumo.graph_contract import GraphContractTable
from gsf.retrieval.kumo.predictor import PredictionContext
from gsf.retrieval.kumo.predictor import _build_graph_receipt
from gsf.retrieval.kumo.predictor import _categorical_value_reference
from gsf.retrieval.kumo.predictor import _contract_view_sources
from gsf.retrieval.kumo.predictor import _deduplicate_inferred_links
from gsf.retrieval.kumo.predictor import _load_relevant_frames
from gsf.retrieval.kumo.predictor import _validate_connector_contract_views
from gsf.retrieval.kumo.predictor import build_prediction_context
from gsf.retrieval.kumo.provider import KumoProviderReadiness
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes


@dataclass(frozen=True)
class _Column:
    name: str


@dataclass(frozen=True)
class _Table:
    primary_key: _Column


@dataclass(frozen=True)
class _Edge:
    src_table: str
    fkey: str
    dst_table: str


class _Graph:
    def __init__(self) -> None:
        self.edges = [
            _Edge("JOB_OUTCOMES", "restart_of_job_id", "JOBS"),
            _Edge("JOB_OUTCOMES", "job_id", "JOBS"),
            _Edge("JOBS", "project_id", "PROJECTS"),
        ]
        self.tables = {
            "JOBS": _Table(_Column("job_id")),
            "PROJECTS": _Table(_Column("project_id")),
        }

    def __getitem__(self, table: str) -> _Table:
        return self.tables[table]

    def unlink(self, src_table: str, fkey: str, dst_table: str) -> None:
        self.edges.remove(_Edge(src_table, fkey, dst_table))


def test_client_initialization_uses_compatibility_registry_only_when_readiness_selects_it() -> None:
    from gsf.retrieval.kumo import predictor

    readiness = KumoProviderReadiness(
        status="ready",
        ready=True,
        expected_model="kumo-rfm",
        wire_model="kumo-relational",
        compatibility_adapter="nvidia-sdfm-0.2.1-kumorfm-2.28.0-relational-model",
        advertised_models=("kumo-relational",),
        nvidia_sdfm_version="0.2.1",
        kumorfm_version="2.28.0",
    )
    registry = object()
    client = object()
    with (
        patch.object(predictor, "_client", None),
        patch.object(predictor, "_provider_failure", None),
        patch.object(predictor, "_provider_readiness", None),
        patch.dict(
            "os.environ",
            {"KUMO_RFM_API_URL": "https://provider.example.test", "KUMO_RFM_API_KEY": "private-test-key"},
        ),
        patch("gsf.retrieval.kumo.predictor.require_kumo_provider_ready", return_value=readiness),
        patch("gsf.retrieval.kumo.compatibility.compatibility_registry", return_value=registry),
        patch("nvidia_sdfm.SDFMClient", return_value=client) as client_type,
    ):
        assert predictor._ensure_init() is client

    client_type.assert_called_once_with(
        "https://provider.example.test",
        api_key="private-test-key",
        registry=registry,
    )


def test_deduplicate_inferred_links_prefers_destination_primary_key_name() -> None:
    graph = _Graph()

    assert _deduplicate_inferred_links(graph) == 1
    assert graph.edges == [
        _Edge("JOB_OUTCOMES", "job_id", "JOBS"),
        _Edge("JOBS", "project_id", "PROJECTS"),
    ]


def test_categorical_value_reference_is_exact_and_bounded_to_complete_low_cardinality_sets() -> None:
    frames = {
        "entities": pd.DataFrame(
            {
                "entity_id": ["entity-1", "entity-2", "entity-3"],
                "lifecycle_status": ["Active", "Partially Active", "Active"],
                "free_text": [f"description-{index}" for index in range(3)],
            }
        ),
        "events": pd.DataFrame(
            {
                "event_id": list(range(13)),
                "category": [f"category-{index}" for index in range(13)],
            }
        ),
    }
    stypes = {
        "entities": {
            "entity_id": "ID",
            "lifecycle_status": "categorical",
            "free_text": "text",
        },
        "events": {"event_id": "ID", "category": "categorical"},
    }

    reference = _categorical_value_reference(frames, stypes)

    assert 'column="lifecycle_status"' in reference
    assert 'exact values=["Active", "Partially Active"]' in reference
    assert "entity-1" not in reference
    assert "description-0" not in reference
    # A partial list would teach the model that omitted values do not exist, so
    # a categorical set above the bound is omitted instead of truncated.
    assert "category-0" not in reference


class _Connector:
    def __init__(
        self,
        database_name: str,
        frames: dict[str, pd.DataFrame],
        *,
        schema_name: str = "prediction",
        table_type: str = TableTypes.VIEW,
    ) -> None:
        self.database_name = database_name
        self.frames = frames
        self.schema_name = schema_name
        self.table_type = table_type
        self.sql: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        table = next(name for name in self.frames if f'"{name}"' in sql)
        return self.frames[table].copy()

    def get_tables(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "table_schema": self.schema_name,
                    "table_name": name,
                    "table_type": self.table_type,
                }
                for name in self.frames
            ]
        )


def _view_source(table: GraphContractTable, database_name: str) -> dict:
    return {
        "name": table.name,
        "schema_name": table.schema_name,
        "database_name": database_name,
        "table_type": TableTypes.VIEW,
        "pk": list(table.primary_key),
        "expected_rows": table.rows,
    }


def test_explicit_contract_refuses_a_configured_partial_row_cap() -> None:
    connector = MagicMock(database_name="prediction_db")
    table = {
        "name": "events",
        "schema_name": "main",
        "database_name": "prediction_db",
        "expected_rows": 2,
    }

    with (
        patch("gsf.retrieval.kumo.predictor._MAX_ROWS_PER_TABLE", 1),
        pytest.raises(ValueError, match="refusing a partial graph"),
    ):
        _load_relevant_frames([connector], [table], database_name="prediction_db", strict=True)

    connector.execute.assert_not_called()


def test_explicit_contract_checks_full_count_when_cap_exceeds_table() -> None:
    connector = MagicMock(database_name="prediction_db")
    connector.execute.return_value = pd.DataFrame({"event_id": [1]})
    table = {
        "name": "events",
        "schema_name": "main",
        "database_name": "prediction_db",
        "expected_rows": 2,
    }

    with (
        patch("gsf.retrieval.kumo.predictor._MAX_ROWS_PER_TABLE", 10),
        pytest.raises(ValueError, match="expected 2 rows but loaded 1"),
    ):
        _load_relevant_frames([connector], [table], database_name="prediction_db", strict=True)


def test_explicit_contract_uses_exact_connector_metadata_edges_and_receipt() -> None:
    frames = {
        "entities": pd.DataFrame({"entity_id": [1, 2], "name": ["one", "two"]}),
        "events": pd.DataFrame(
            {
                "event_id": [10, 11],
                "entity_id": [1, 2],
                "recorded_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "other_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            }
        ),
    }
    wrong = _Connector("wrong_db", {})
    selected = _Connector("prediction_db", frames)
    contract = GraphContract(
        database_name="prediction_db",
        tables=(
            GraphContractTable("entities", "prediction", ("entity_id",), None, 2),
            GraphContractTable("events", "prediction", ("event_id",), "recorded_at", 2),
        ),
        edges=(GraphContractEdge("events", ("entity_id",), "entities", ("entity_id",)),),
        revision="sha256:" + "a" * 64,
    )
    relevant = [_view_source(table, "prediction_db") for table in contract.tables]
    client = MagicMock()
    client.kumorfm.return_value = object()

    with (
        patch("gsf.retrieval.kumo.predictor._ensure_init", return_value=client),
        patch("kumorfm.rfm.Graph.infer_links") as infer_links,
    ):
        context = build_prediction_context(
            [wrong, selected],
            relevant,
            database_name="prediction_db",
            graph_contract=contract,
            join_paths=[{"path": [{"source_table": "wrong"}]}],
            examples=[
                {
                    "id": "example-1",
                    "question": "Predict events",
                    "query": "PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH entities.entity_id",
                }
            ],
        )

    assert isinstance(context, PredictionContext)
    assert context.connector is selected
    assert selected.sql == [
        'SELECT * FROM "prediction"."entities"',
        'SELECT * FROM "prediction"."events"',
    ]
    assert infer_links.call_count == 0
    assert context.time_columns == {"entities": None, "events": "recorded_at"}
    assert [(src, dst) for src, _fkey, dst in context.graph_edges] == [("events", "entities")]
    assert context.graph_receipt == {
        "schema_version": 1,
        "database_name": "prediction_db",
        "mode": "explicit",
        "contract_revision": "sha256:" + "a" * 64,
        "tables": [
            {
                "name": "entities",
                "schema_name": "prediction",
                "primary_key": ["entity_id"],
                "time_column": None,
                "loaded_rows": 2,
            },
            {
                "name": "events",
                "schema_name": "prediction",
                "primary_key": ["event_id"],
                "time_column": "recorded_at",
                "loaded_rows": 2,
            },
        ],
        "edges": [
            {
                "source_table": "events",
                "source_columns": ["entity_id"],
                "target_table": "entities",
                "target_columns": ["entity_id"],
            }
        ],
        "pql_example_ids": ["example-1"],
        "pql_example_count": 1,
        "graph_revision": context.graph_receipt["graph_revision"],
    }
    assert context.graph_receipt["graph_revision"].startswith("sha256:")

    changed_examples = _build_graph_receipt(
        database_name="prediction_db",
        graph=context.kumo_model.graph,
        frames=frames,
        examples=[{"id": "fresh-deployment-uuid"}],
        contract=contract,
    )
    assert changed_examples["pql_example_ids"] == ["fresh-deployment-uuid"]
    assert changed_examples["graph_revision"] == context.graph_receipt["graph_revision"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source.update({"database_name": "raw_db"}),
            "outside the governed view contract",
        ),
        (
            lambda source: source.update({"schema_name": "raw"}),
            "outside the governed view contract",
        ),
        (
            lambda source: source.update({"table_type": TableTypes.BASE_TABLE}),
            "must be a catalog VIEW",
        ),
    ],
)
def test_contract_view_boundary_rejects_cross_database_schema_and_raw_sources(
    mutate,
    message: str,
) -> None:
    table = GraphContractTable("events", "prediction", ("event_id",), "recorded_at", 2)
    contract = GraphContract(
        database_name="prediction_db",
        tables=(table,),
        edges=(),
        revision="sha256:" + "a" * 64,
    )
    source = _view_source(table, "prediction_db")
    mutate(source)

    with pytest.raises(ValueError, match=message):
        _contract_view_sources([source], contract)


def test_contract_view_boundary_rejects_extra_raw_graph_source() -> None:
    table = GraphContractTable("events", "prediction", ("event_id",), "recorded_at", 2)
    contract = GraphContract(
        database_name="prediction_db",
        tables=(table,),
        edges=(),
        revision="sha256:" + "a" * 64,
    )
    raw = {
        **_view_source(table, "prediction_db"),
        "name": "raw_events",
        "schema_name": "main",
        "table_type": TableTypes.BASE_TABLE,
    }

    with pytest.raises(ValueError, match="outside the governed view contract"):
        _contract_view_sources([_view_source(table, "prediction_db"), raw], contract)


def test_connector_boundary_rejects_raw_table_at_contracted_view_path() -> None:
    table = GraphContractTable("events", "prediction", ("event_id",), "recorded_at", 2)
    contract = GraphContract(
        database_name="prediction_db",
        tables=(table,),
        edges=(),
        revision="sha256:" + "a" * 64,
    )
    connector = _Connector(
        "prediction_db",
        {"events": pd.DataFrame({"event_id": [1, 2]})},
        table_type=TableTypes.BASE_TABLE,
    )

    with pytest.raises(ValueError, match="missing contracted VIEW"):
        _validate_connector_contract_views(connector, contract)
