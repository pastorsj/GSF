from types import SimpleNamespace

import pandas as pd
import pytest
from gsf.retrieval.kumo import pql_gen
from gsf.retrieval.kumo.graph_contract import GraphContractPredictionScope
from gsf.retrieval.kumo.pql_gen import PqlEntitySelectionError
from gsf.retrieval.kumo.pql_gen import PqlPredictionScopeError
from gsf.retrieval.kumo.pql_gen import _effective_entity_cap
from gsf.retrieval.kumo.pql_gen import _effective_entity_ids
from gsf.retrieval.kumo.pql_gen import _effective_forecast_anchor
from gsf.retrieval.kumo.pql_gen import _enforce_prediction_entity_boundary
from gsf.retrieval.kumo.pql_gen import _forecast_anchor
from gsf.retrieval.kumo.pql_gen import _is_context_capacity_error
from gsf.retrieval.kumo.pql_gen import _is_context_size_limit_error
from gsf.retrieval.kumo.pql_gen import _is_transient_exec_error
from gsf.retrieval.kumo.pql_gen import _NeighbourhoodMemo
from gsf.retrieval.kumo.pql_gen import _predict_resilient
from gsf.retrieval.kumo.pql_gen import _resolve_indices
from gsf.retrieval.kumo.pql_gen import _retry_at_full_neighbourhood
from gsf.retrieval.kumo.pql_gen import _validate_prediction_scope_ids
from gsf.retrieval.kumo.pql_gen import canonicalize_pql_identifiers
from gsf.retrieval.kumo.pql_gen import extract_pql
from gsf.retrieval.kumo.pql_gen import generate_pql
from gsf.retrieval.kumo.pql_gen import parse_entity
from gsf.retrieval.kumo.pql_gen import predict_all
from gsf.retrieval.kumo.provider import KumoProviderCompatibilityError
from gsf.retrieval.kumo.provider import KumoProviderReadiness
from gsf.retrieval.kumo.provider import KumoProviderUnavailableError

# The SDK's client-side per-table row cap (kumorfm.rfm.payload.validate_payload_table_rows).
_ROW_LIMIT_ERROR = (
    "Request batch 0 table 'context.related_tables.GPU_ALLOCATIONS' contains "
    "32,000 rows, exceeding the 10,000-row limit"
)


class _Connector:
    def __init__(self) -> None:
        self.sql = ""

    def execute(self, sql: str) -> pd.DataFrame:
        self.sql = sql
        return pd.DataFrame({"JOB_ID": ["job-1"]})


class _UnexpectedConnector:
    def execute(self, _sql: str) -> pd.DataFrame:
        raise AssertionError("source database should not be queried")


class _EmptyConnector:
    def execute(self, _sql: str) -> pd.DataFrame:
        return pd.DataFrame({"ENTITY_ID": []})


class _MaxTimestampConnector:
    def __init__(self, value: object) -> None:
        self.value = value

    def execute(self, _sql: str) -> pd.DataFrame:
        return pd.DataFrame({"m": [self.value]})


def _prediction_scope(
    *,
    entity_table: str = "entities",
    entity_column: str = "entity_id",
    population_rows: int = 2,
) -> GraphContractPredictionScope:
    return GraphContractPredictionScope(
        anchor_time="2026-08-12T00:00:00+00:00",
        entity_table=entity_table,
        entity_column=entity_column,
        population_view="reviewed_entities",
        population_column=entity_column,
        population_rows=population_rows,
    )


def test_extract_pql_ignores_predict_in_explanatory_prose() -> None:
    response = """We need to predict the outcome from the available data.
PREDICT jobs.status FOR EACH jobs.job_id
This query scores every job.
"""

    assert extract_pql(response) == "PREDICT jobs.status FOR EACH jobs.job_id"


def test_extract_pql_rejects_response_without_statement() -> None:
    assert extract_pql("We should predict the likely outcome.") == ""


def test_extract_pql_preserves_fenced_statement() -> None:
    response = """```pql
PREDICT events.status
FOR EACH jobs.job_id
```"""

    assert extract_pql(response) == ("PREDICT events.status\nFOR EACH jobs.job_id")


@pytest.mark.parametrize(
    "aggregation",
    [
        "COUNT(events.*)",
        "COUNT(events.*, 0, 30, days) > 0",
        "SUM(events.value)",
    ],
)
def test_scalar_prediction_drops_link_only_rank(aggregation: str) -> None:
    pql = f"PREDICT {aggregation} RANK TOP 100 FOR EACH entities.entity_id"

    assert pql_gen._drop_scalar_rank(pql) == (
        f"PREDICT {aggregation} FOR EACH entities.entity_id"
    )


def test_link_prediction_keeps_rank() -> None:
    pql = (
        "PREDICT LIST_DISTINCT(events.item_id) RANK TOP 10 FOR EACH entities.entity_id"
    )

    assert pql_gen._drop_scalar_rank(pql) == pql


def test_link_prediction_keeps_rank_when_assumption_contains_scalar_aggregation() -> (
    None
):
    pql = (
        "PREDICT LIST_DISTINCT(events.item_id, 0, 7, days) "
        "RANK TOP 10 FOR EACH entities.entity_id "
        "ASSUMING COUNT(events.*, 0, 7, days) > 0"
    )

    assert pql_gen._drop_scalar_rank(pql) == pql


def test_canonicalize_pql_identifiers_uses_graph_casing() -> None:
    graph_ddl = (
        "JOBS(JOB_ID primary_key, PRIORITY_TIER categorical)  -- PRIMARY KEY (JOB_ID)"
    )

    assert (
        canonicalize_pql_identifiers(
            "PREDICT jobs.priority_tier FOR EACH jobs.job_id",
            graph_ddl,
        )
        == "PREDICT JOBS.PRIORITY_TIER FOR EACH JOBS.JOB_ID"
    )


def test_resolve_indices_queries_canonical_snowflake_identifier() -> None:
    connector = _Connector()
    graph_ddl = (
        "JOBS(JOB_ID primary_key, PRIORITY_TIER categorical)  -- PRIMARY KEY (JOB_ID)"
    )
    pql = canonicalize_pql_identifiers(
        "PREDICT jobs.priority_tier FOR EACH jobs.job_id",
        graph_ddl,
    )

    assert _resolve_indices(
        pql,
        None,
        connector,
        10,
        {"JOBS": '"GPU_FLEET"."JOBS"'},
    ) == ["job-1"]
    assert connector.sql == (
        'SELECT DISTINCT "JOB_ID" FROM "GPU_FLEET"."JOBS" WHERE "JOB_ID" IS NOT NULL LIMIT 10'
    )


def test_resolve_indices_uses_ids_loaded_into_graph() -> None:
    assert _resolve_indices(
        "PREDICT JOBS.PRIORITY_TIER FOR EACH JOBS.JOB_ID",
        None,
        _UnexpectedConnector(),
        2,
        available_entity_ids={
            "jobs": ["job-in-graph-1", "job-in-graph-2", "job-in-graph-3"]
        },
    ) == ["job-in-graph-1", "job-in-graph-2"]


def test_resolve_indices_does_not_turn_an_empty_explicit_scope_into_predict_all() -> (
    None
):
    with pytest.raises(PqlEntitySelectionError, match="matched zero graph-backed rows"):
        _resolve_indices(
            "PREDICT events.outcome FOR EACH entities.entity_id",
            "SELECT entity_id FROM entities WHERE lifecycle_status = 'active'",
            _EmptyConnector(),
            10,
            available_entity_ids={"entities": ["entity-1", "entity-2"]},
        )


def test_resolve_indices_deduplicates_explicit_sql_in_first_seen_order() -> None:
    class Connector:
        def execute(self, _sql: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "entity_id": [
                        "entity-3",
                        "entity-1",
                        "entity-3",
                        "entity-2",
                        "entity-1",
                    ]
                }
            )

    assert _resolve_indices(
        "PREDICT entities.status FOR EACH entities.entity_id",
        "SELECT entity_id FROM entities",
        Connector(),
        10,
        available_entity_ids={"entities": ["entity-1", "entity-3"]},
    ) == ["entity-3", "entity-1"]


@pytest.mark.parametrize(
    ("prediction_scope_ids", "available_entity_ids", "message"),
    [
        ((), {"entities": ["entity-1", "entity-2", "entity-3"]}, "missing or empty"),
        (
            ("entity-1",),
            {"entities": ["entity-1", "entity-2", "entity-3"]},
            "count does not match",
        ),
        (
            ("entity-1", "entity-1"),
            {"entities": ["entity-1", "entity-2", "entity-3"]},
            "duplicates",
        ),
        (
            ("entity-1", None),
            {"entities": ["entity-1", "entity-2", "entity-3"]},
            "missing value",
        ),
        (
            ("entity-1", "outside"),
            {"entities": ["entity-1", "entity-2", "entity-3"]},
            "outside the loaded graph",
        ),
        (("entity-1", "entity-3"), None, "requires loaded graph"),
        (
            ("entity-1", "entity-3"),
            {"entities": []},
            "missing or empty in the loaded graph",
        ),
        (
            ("entity-1", "entity-3"),
            {"entities": ["entity-1", "entity-1", "entity-3"]},
            "Loaded graph entity identifiers contain duplicates",
        ),
    ],
)
def test_prediction_scope_handoff_rejects_invalid_identifiers(
    prediction_scope_ids: tuple[object, ...],
    available_entity_ids: dict[str, list[object]] | None,
    message: str,
) -> None:
    with pytest.raises(PqlPredictionScopeError, match=message):
        _validate_prediction_scope_ids(
            _prediction_scope(),
            prediction_scope_ids,
            available_entity_ids,
        )


def test_predict_all_rejects_empty_scope_before_provider_execution() -> None:
    class Model:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("provider must not be called")

    with pytest.raises(PqlPredictionScopeError, match="missing or empty"):
        predict_all(
            "PREDICT entities.status FOR EACH entities.entity_id",
            kumo_model=Model(),
            connector=_UnexpectedConnector(),
            available_entity_ids={"entities": ["entity-1", "entity-2"]},
            prediction_scope=_prediction_scope(),
            prediction_scope_ids=(),
        )


def test_generate_pql_rejects_out_of_graph_scope_before_llm_or_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pql_gen,
        "invoke_text",
        lambda *_args: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    with pytest.raises(PqlPredictionScopeError, match="outside the loaded graph"):
        generate_pql(
            "Predict entity status",
            llm=object(),
            kumo_model=object(),
            connector=_UnexpectedConnector(),
            graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
            available_entity_ids={"entities": ["entity-1", "entity-2"]},
            prediction_scope=_prediction_scope(),
            prediction_scope_ids=("entity-1", "outside"),
        )


def test_predict_all_keeps_legacy_unscoped_entity_resolution() -> None:
    class Connector:
        def execute(self, _sql: str) -> pd.DataFrame:
            return pd.DataFrame({"entity_id": ["entity-1", "entity-2"]})

    class Model:
        def __init__(self) -> None:
            self.indices = None

        def predict(self, _query: str, *, indices=None, **_kwargs) -> pd.DataFrame:
            self.indices = indices
            return pd.DataFrame({"ENTITY": indices, "STATUS_PRED": [0.4, 0.6]})

    model = Model()
    predict_all(
        "PREDICT entities.status FOR EACH entities.entity_id",
        kumo_model=model,
        connector=Connector(),
    )

    assert model.indices == ["entity-1", "entity-2"]


def test_predict_all_executes_full_governed_scope_with_existing_batching() -> None:
    entity_ids = list(range(2_001))

    class Model:
        def __init__(self) -> None:
            self.batches: list[list[int]] = []

        def predict(self, _query: str, *, indices=None, **_kwargs) -> pd.DataFrame:
            assert indices is not None
            self.batches.append(indices)
            return pd.DataFrame(
                {"ENTITY": indices, "STATUS_PRED": [0.5] * len(indices)}
            )

    model = Model()
    result = predict_all(
        "PREDICT entities.status FOR EACH entities.entity_id",
        kumo_model=model,
        connector=_UnexpectedConnector(),
        max_entities=10,
        available_entity_ids={"entities": entity_ids},
        prediction_scope=_prediction_scope(population_rows=len(entity_ids)),
        prediction_scope_ids=tuple(entity_ids),
    )

    assert [len(batch) for batch in model.batches] == [1_000, 1_000, 1]
    assert len(result) == len(entity_ids)


def test_predict_all_rejects_provider_entity_outside_requested_scope() -> None:
    class Model:
        def predict(self, _query: str, *, indices=None, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                {"ENTITY": [indices[0], "outside"], "STATUS_PRED": [0.5, 0.9]}
            )

    with pytest.raises(
        PqlPredictionScopeError, match="outside the requested graph-backed scope"
    ):
        predict_all(
            "PREDICT entities.status FOR EACH entities.entity_id",
            kumo_model=Model(),
            connector=_UnexpectedConnector(),
            available_entity_ids={"entities": ["entity-1", "entity-2"]},
            prediction_scope=_prediction_scope(),
            prediction_scope_ids=("entity-1", "entity-2"),
        )


@pytest.mark.parametrize(
    ("pql", "frame"),
    [
        (
            "PREDICT SUM(events.value, 0, 7, days) FORECAST 2 TIMEFRAMES FOR EACH entities.entity_id",
            pd.DataFrame(
                {
                    "ENTITY": ["entity-1", "entity-1"],
                    "TIME": ["2026-08-19", "2026-08-26"],
                    "TARGET_PRED": [1.0, 2.0],
                }
            ),
        ),
        (
            "PREDICT LIST_DISTINCT(events.item_id, 0, 7, days) RANK TOP 2 FOR EACH entities.entity_id",
            pd.DataFrame(
                {
                    "ENTITY": ["entity-1", "entity-1", "entity-2", "entity-2"],
                    "CLASS": ["item-1", "item-2", "item-2", "item-3"],
                    "PROBABILITY": [0.8, 0.7, 0.9, 0.6],
                }
            ),
        ),
    ],
)
def test_provider_boundary_allows_repeated_forecast_and_link_rows(
    pql: str, frame: pd.DataFrame
) -> None:
    _enforce_prediction_entity_boundary(frame, ["entity-1", "entity-2"], pql=pql)


def test_provider_boundary_matches_sdk_composite_rendering_but_keeps_single_keys_typed() -> (
    None
):
    composite = pd.DataFrame(
        {
            "account_id": ["1", "true"],
            "region": ["West", "East"],
            "TRUE_PROB": [0.8, 0.7],
        }
    )
    _enforce_prediction_entity_boundary(
        composite,
        [(1, "West"), (True, "East")],
        pql="PREDICT accounts.risk FOR EACH accounts.account_id",
    )

    with pytest.raises(
        PqlPredictionScopeError, match="outside the requested graph-backed scope"
    ):
        _enforce_prediction_entity_boundary(
            pd.DataFrame({"ENTITY": ["1"], "TRUE_PROB": [0.8]}),
            [1],
            pql="PREDICT accounts.risk FOR EACH accounts.account_id",
        )


def test_matching_governed_scope_above_graph_cap_fails_closed() -> None:
    with pytest.raises(
        PqlPredictionScopeError, match="above the 10000-entity graph execution limit"
    ):
        _effective_entity_cap(
            "PREDICT entities.status FOR EACH entities.entity_id",
            10,
            _prediction_scope(population_rows=10_001),
        )


def test_scoped_nonmatching_entity_cannot_fall_back_to_source_database() -> None:
    with pytest.raises(PqlPredictionScopeError, match="inventory is missing or empty"):
        predict_all(
            "PREDICT other_entities.status FOR EACH other_entities.other_id",
            kumo_model=object(),
            connector=_UnexpectedConnector(),
            available_entity_ids={"entities": ["entity-1", "entity-2"]},
            prediction_scope=_prediction_scope(),
            prediction_scope_ids=("entity-1", "entity-2"),
        )


def test_generate_pql_repairs_empty_entity_sql_without_calling_predict_with_none(
    monkeypatch,
) -> None:
    responses = iter(
        [
            """```pql
PREDICT COUNT(events.* WHERE events.outcome = 'Failed', 0, 30, days) > 0 FOR EACH entities.entity_id
```
```sql
SELECT entity_id FROM entities WHERE lifecycle_status = 'active'
```""",
            """```pql
PREDICT COUNT(events.* WHERE events.outcome = 'Failed', 0, 30, days) > 0 FOR EACH entities.entity_id
```
```sql
SELECT entity_id FROM entities WHERE lifecycle_status = 'Active'
```""",
        ]
    )
    prompts: list[str] = []

    def invoke(_llm, prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    class Connector:
        def execute(self, sql: str) -> pd.DataFrame:
            if "'active'" in sql:
                return pd.DataFrame({"entity_id": []})
            return pd.DataFrame({"entity_id": ["entity-1"]})

    class Model:
        def __init__(self) -> None:
            self.indices: list[list[str] | None] = []

        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, indices=None, **_kwargs) -> pd.DataFrame:
            self.indices.append(indices)
            return pd.DataFrame({"ENTITY": indices, "TRUE_PROB": [0.75]})

    model = Model()
    monkeypatch.setattr(pql_gen, "invoke_text", invoke)

    result = generate_pql(
        "Which active entities are likely to fail?",
        llm=object(),
        kumo_model=model,
        connector=Connector(),
        graph_ddl=(
            "entities(entity_id ID, lifecycle_status categorical)  -- PRIMARY KEY (entity_id)\n"
            "events(event_id ID, entity_id ID, outcome categorical)  -- PRIMARY KEY (event_id)\n"
            "FOREIGN KEY events.entity_id -> entities.<pk>"
        ),
        graph_edges=[("events", "entity_id", "entities")],
        column_reference=(
            '- table="entities", column="lifecycle_status", type=categorical, exact values=["Active", "Inactive"]'
        ),
        available_entity_ids={"entities": ["entity-1"]},
        max_tries=2,
    )

    assert result.success
    assert result.attempts == 2
    assert model.indices == [["entity-1"]]
    assert 'exact values=["Active", "Inactive"]' in prompts[0]
    assert "entity-selection SQL matched zero graph-backed rows" in prompts[1]


def test_generate_pql_explain_requires_entity_correlated_prediction_frame(
    monkeypatch,
) -> None:
    pql = "PREDICT entities.status FOR EACH entities.entity_id"
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: pql)

    class Model:
        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, *, indices=None, **_kwargs):
            assert indices == ["entity-1"]
            return SimpleNamespace(
                summary="Narrative without a prediction frame", prediction=None
            )

    result = generate_pql(
        "Explain the entity's predicted status",
        llm=object(),
        kumo_model=Model(),
        connector=_UnexpectedConnector(),
        graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
        available_entity_ids={"entities": ["entity-1"]},
        explain=True,
        explain_entity="entity-1",
        max_tries=1,
    )

    assert not result.success
    assert result.error == "Prediction provider returned an invalid result shape."


def test_generate_pql_explain_accepts_requested_entity_prediction_frame(
    monkeypatch,
) -> None:
    pql = "PREDICT entities.status FOR EACH entities.entity_id"
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: pql)

    class Model:
        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, *, indices=None, **_kwargs):
            assert indices == ["entity-1"]
            return SimpleNamespace(
                summary="Entity-correlated explanation",
                prediction=pd.DataFrame(
                    {"ENTITY": ["entity-1"], "STATUS_PRED": [0.75]}
                ),
            )

    result = generate_pql(
        "Explain the entity's predicted status",
        llm=object(),
        kumo_model=Model(),
        connector=_UnexpectedConnector(),
        graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
        available_entity_ids={"entities": ["entity-1"]},
        explain=True,
        explain_entity="entity-1",
        max_tries=1,
    )

    assert result.success
    assert result.explanation == "Entity-correlated explanation"
    assert result.rows == [{"ENTITY": "entity-1", "STATUS_PRED": 0.75}]


def test_generate_pql_explain_rejects_empty_entity_prediction_frame(
    monkeypatch,
) -> None:
    pql = "PREDICT entities.status FOR EACH entities.entity_id"
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: pql)

    class Model:
        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, *, indices=None, **_kwargs):
            assert indices == ["entity-1"]
            return SimpleNamespace(
                summary="Narrative without a correlated result row",
                prediction=pd.DataFrame(columns=["ENTITY", "STATUS_PRED"]),
            )

    result = generate_pql(
        "Explain the entity's predicted status",
        llm=object(),
        kumo_model=Model(),
        connector=_UnexpectedConnector(),
        graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
        available_entity_ids={"entities": ["entity-1"]},
        explain=True,
        explain_entity="entity-1",
        max_tries=1,
    )

    assert not result.success
    assert (
        result.error
        == "Prediction provider returned no entity-correlated explanation rows."
    )


def test_forecast_anchor_inherits_timezone_from_aware_source_column() -> None:
    anchor = _forecast_anchor(
        "PREDICT COUNT(events.*, 0, 60, days) > 0 FOR EACH entities.entity_id",
        _MaxTimestampConnector(pd.Timestamp("2026-12-31T00:00:00Z")),
        {"events": "recorded_at"},
        now="2026-08-31",
    )

    assert anchor == pd.Timestamp("2026-08-31T00:00:00Z")
    assert anchor.tz is not None


def test_forecast_anchor_remains_naive_for_naive_source_column() -> None:
    anchor = _forecast_anchor(
        "PREDICT COUNT(events.*, 0, 60, days) > 0 FOR EACH entities.entity_id",
        _MaxTimestampConnector(pd.Timestamp("2026-12-31")),
        {"events": "recorded_at"},
        now="2026-08-31T00:00:00-04:00",
    )

    assert anchor == pd.Timestamp("2026-08-31")
    assert anchor.tz is None


def test_forecast_anchor_converts_aware_now_to_source_timezone() -> None:
    anchor = _forecast_anchor(
        "PREDICT COUNT(events.*, 0, 60, days) > 0 FOR EACH entities.entity_id",
        _MaxTimestampConnector(pd.Timestamp("2026-12-31T00:00:00Z")),
        {"events": "recorded_at"},
        now="2026-08-31T02:00:00+02:00",
    )

    assert anchor == pd.Timestamp("2026-08-31T00:00:00Z")


def test_matching_graph_scope_replaces_data_max_anchor_and_constrains_entities(
    monkeypatch,
) -> None:
    pql = "PREDICT COUNT(events.*, 0, 30, minutes) > 0 FOR EACH entities.entity_id"
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: pql)

    class Model:
        def __init__(self) -> None:
            self.call: dict = {}

        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, **kwargs) -> pd.DataFrame:
            self.call = kwargs
            return pd.DataFrame({"ENTITY": kwargs["indices"], "TRUE_PROB": [0.7, 0.8]})

    model = Model()
    result = generate_pql(
        "Which entities are likely to have events?",
        llm=object(),
        kumo_model=model,
        connector=_UnexpectedConnector(),
        graph_ddl=(
            "entities(entity_id ID)  -- PRIMARY KEY (entity_id)\n"
            "events(event_id ID, entity_id ID, observed_at timestamp)  -- PRIMARY KEY (event_id)"
        ),
        graph_edges=[("events", "entity_id", "entities")],
        time_columns={"events": "observed_at"},
        available_entity_ids={"entities": ["entity-1", "entity-2", "entity-3"]},
        prediction_scope=_prediction_scope(),
        prediction_scope_ids=("entity-1", "entity-3"),
    )

    assert result.success
    assert model.call["indices"] == ["entity-1", "entity-3"]
    assert model.call["anchor_time"] == pd.Timestamp("2026-08-12T00:00:00Z")


def test_graph_scope_never_broadens_an_explicit_entity_selection() -> None:
    pql = "PREDICT events.outcome FOR EACH entities.entity_id"
    available = _effective_entity_ids(
        pql,
        {"entities": ["entity-1", "entity-2", "entity-3"]},
        _prediction_scope(),
        ("entity-1", "entity-3"),
    )

    class Connector:
        def execute(self, _sql: str) -> pd.DataFrame:
            return pd.DataFrame({"entity_id": ["entity-1", "entity-2"]})

    assert _resolve_indices(
        pql,
        "SELECT entity_id FROM entities",
        Connector(),
        10,
        available_entity_ids=available,
    ) == ["entity-1"]


def test_group_by_preserves_entity_sql_and_intersects_governed_population(
    monkeypatch,
) -> None:
    response = """```pql
PREDICT entities.risk_score FOR EACH entities.entity_id
```
```sql
SELECT entity_id FROM entities WHERE review_tier = 'priority'
```"""
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: response)

    class Connector:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, sql: str) -> pd.DataFrame:
            self.queries.append(sql)
            if "review_tier" in sql:
                return pd.DataFrame({"entity_id": ["entity-1", "entity-2"]})
            if '"region"' in sql:
                return pd.DataFrame(
                    {
                        "entity_id": ["entity-1", "entity-2", "entity-3"],
                        "region": ["east", "west", "east"],
                    }
                )
            raise AssertionError(f"unexpected SQL: {sql}")

    class Model:
        def __init__(self) -> None:
            self.indices = None

        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, *, indices=None, **_kwargs) -> pd.DataFrame:
            self.indices = indices
            return pd.DataFrame({"ENTITY": indices, "RISK_PRED": [0.8]})

    connector = Connector()
    model = Model()
    result = generate_pql(
        "Show priority entity risk by region",
        llm=object(),
        kumo_model=model,
        connector=connector,
        graph_ddl=(
            "entities(entity_id ID, risk_score float, review_tier categorical, region categorical)"
            "  -- PRIMARY KEY (entity_id)"
        ),
        group_by="region",
        available_entity_ids={"entities": ["entity-1", "entity-2", "entity-3"]},
        prediction_scope=_prediction_scope(),
        prediction_scope_ids=("entity-1", "entity-3"),
    )

    assert result.success
    assert (
        result.entity_sql
        == "SELECT entity_id FROM entities WHERE review_tier = 'priority'"
    )
    assert result.note is None
    assert model.indices == ["entity-1"]
    assert result.num_entities == 1
    assert result.rows == [
        {"region": "east", "total": 0.8, "average": 0.8, "n_entities": 1}
    ]
    assert "review_tier" in connector.queries[0]


def test_group_by_rejects_provider_entity_outside_intersected_population(
    monkeypatch,
) -> None:
    response = """```pql
PREDICT entities.risk_score FOR EACH entities.entity_id
```
```sql
SELECT entity_id FROM entities WHERE review_tier = 'priority'
```"""
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: response)

    class Connector:
        def execute(self, sql: str) -> pd.DataFrame:
            if "review_tier" in sql:
                return pd.DataFrame({"entity_id": ["entity-1", "entity-2"]})
            raise AssertionError(
                "group lookup must not run after an out-of-scope provider result"
            )

    class Model:
        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, *, indices=None, **_kwargs) -> pd.DataFrame:
            assert indices == ["entity-1"]
            return pd.DataFrame({"ENTITY": ["entity-2"], "RISK_PRED": [0.8]})

    result = generate_pql(
        "Show priority entity risk by region",
        llm=object(),
        kumo_model=Model(),
        connector=Connector(),
        graph_ddl=(
            "entities(entity_id ID, risk_score float, review_tier categorical, region categorical)"
            "  -- PRIMARY KEY (entity_id)"
        ),
        group_by="region",
        available_entity_ids={"entities": ["entity-1", "entity-2", "entity-3"]},
        prediction_scope=_prediction_scope(),
        prediction_scope_ids=("entity-1", "entity-3"),
    )

    assert not result.success
    assert result.error is not None
    assert "outside the requested graph-backed scope" in result.error


@pytest.mark.parametrize(
    ("pql", "available_entity_ids", "time_columns"),
    [
        (
            "PREDICT COUNT(publication_events.*, 0, 182, days) FOR EACH author_entities.author_id",
            {"author_entities": ["author-1", "author-2", "author-3"]},
            {"publication_events": "published_at"},
        ),
        (
            "PREDICT COUNT(tool_excursion_events.*, 0, 14, days) > 0 FOR EACH tool_entities.tool_id",
            {"tool_entities": ["tool-1", "tool-2", "tool-3"]},
            {"tool_excursion_events": "event_at"},
        ),
    ],
)
def test_graph_anchor_applies_to_a_different_entity_without_scoping_its_population(
    pql: str,
    available_entity_ids: dict[str, list[str]],
    time_columns: dict[str, str],
) -> None:
    anchor = _effective_forecast_anchor(
        pql,
        _UnexpectedConnector(),
        time_columns,
        table_names=None,
        prediction_scope=_prediction_scope(),
    )

    assert anchor == pd.Timestamp(_prediction_scope().anchor_time)
    assert (
        _effective_entity_ids(
            pql,
            available_entity_ids,
            _prediction_scope(),
            ("entity-1", "entity-3"),
        )
        == available_entity_ids
    )


def test_graph_anchor_applies_to_nonmatching_entity_in_generate_pql(
    monkeypatch,
) -> None:
    pql = "PREDICT COUNT(publication_events.*, 0, 182, days) FOR EACH author_entities.author_id"
    monkeypatch.setattr(pql_gen, "invoke_text", lambda *_args: pql)

    class Model:
        def __init__(self) -> None:
            self.call: dict = {}

        def validate_pql(self, _query: str) -> None:
            return None

        def predict(self, _query: str, **kwargs) -> pd.DataFrame:
            self.call = kwargs
            return pd.DataFrame(
                {"ENTITY": kwargs["indices"], "COUNT_PRED": [1.0, 2.0, 3.0]}
            )

    model = Model()
    paper_scope = _prediction_scope(
        entity_table="paper_entities",
        entity_column="paper_id",
    )
    result = generate_pql(
        "How many papers is each reviewed author likely to publish?",
        llm=object(),
        kumo_model=model,
        connector=_UnexpectedConnector(),
        graph_ddl=(
            "paper_entities(paper_id ID)  -- PRIMARY KEY (paper_id)\n"
            "author_entities(author_id ID)  -- PRIMARY KEY (author_id)\n"
            "publication_events(publication_id ID, author_id ID, published_at timestamp)"
            "  -- PRIMARY KEY (publication_id)"
        ),
        graph_edges=[("publication_events", "author_id", "author_entities")],
        time_columns={"publication_events": "published_at"},
        available_entity_ids={
            "paper_entities": ["paper-1", "paper-2", "paper-3"],
            "author_entities": ["author-1", "author-2", "author-3"],
        },
        prediction_scope=paper_scope,
        prediction_scope_ids=("paper-1", "paper-3"),
    )

    assert result.success
    assert model.call["anchor_time"] == pd.Timestamp(paper_scope.anchor_time)
    assert model.call["indices"] == ["author-1", "author-2", "author-3"]


def test_graph_anchor_is_not_attached_to_a_non_temporal_prediction() -> None:
    assert (
        _effective_forecast_anchor(
            "PREDICT entities.status FOR EACH entities.entity_id",
            _UnexpectedConnector(),
            {"events": "observed_at"},
            table_names=None,
            prediction_scope=_prediction_scope(),
        )
        is None
    )


def test_row_limit_rejection_is_a_context_capacity_error() -> None:
    """It must reach the neighbourhood backoff, not escape into the PQL regenerate loop."""
    assert _is_context_capacity_error(_ROW_LIMIT_ERROR)


def test_row_limit_rejection_steps_down_instead_of_retrying_at_full() -> None:
    """It is deterministic — the same neighbourhood rebuilds the same oversize table every time."""
    assert _is_context_size_limit_error(_ROW_LIMIT_ERROR)
    assert not _retry_at_full_neighbourhood(_ROW_LIMIT_ERROR)


def test_unrelated_errors_are_not_classified_as_row_limit() -> None:
    assert not _is_context_size_limit_error("Failed to parse query")
    assert not _is_context_capacity_error("Failed to parse query")


def test_provider_contract_failure_does_not_regenerate_pql(monkeypatch) -> None:
    calls = 0

    def invoke(_llm, _prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "PREDICT entities.status FOR EACH entities.entity_id"

    class Model:
        def validate_pql(self, _query: str) -> None:
            raise KumoProviderCompatibilityError(
                KumoProviderReadiness(
                    status="incompatible",
                    ready=False,
                    expected_model="kumo-rfm",
                    advertised_models=("kumo-relational",),
                    nvidia_sdfm_version="0.2.1",
                    kumorfm_version="2.28.0",
                    error_code="KUMO_PROVIDER_MODEL_INCOMPATIBLE",
                )
            )

    monkeypatch.setattr(pql_gen, "invoke_text", invoke)
    result = generate_pql(
        "Predict entity status",
        llm=object(),
        kumo_model=Model(),
        connector=_UnexpectedConnector(),
        graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
        available_entity_ids={"entities": ["entity-1"]},
        max_tries=5,
    )

    assert result.success is False
    assert result.attempts == 1
    assert calls == 1
    assert "not a forecast" in str(result.error)


@pytest.mark.parametrize(("retryable", "expected_attempts"), [(False, 1), (True, 3)])
def test_typed_provider_unavailability_repairs_only_when_retryable(
    monkeypatch,
    retryable: bool,
    expected_attempts: int,
) -> None:
    calls = 0

    def invoke(_llm, _prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "PREDICT entities.status FOR EACH entities.entity_id"

    readiness = KumoProviderReadiness(
        status="unavailable",
        ready=False,
        expected_model="kumo-rfm",
        advertised_models=(),
        nvidia_sdfm_version="0.2.1",
        kumorfm_version="2.28.0",
        error_code="KUMO_PROVIDER_NOT_READY",
        retryable=retryable,
    )

    class Model:
        def validate_pql(self, _query: str) -> None:
            raise KumoProviderUnavailableError(readiness)

    monkeypatch.setattr(pql_gen, "invoke_text", invoke)
    result = generate_pql(
        "Predict entity status",
        llm=object(),
        kumo_model=Model(),
        connector=_UnexpectedConnector(),
        graph_ddl="entities(entity_id ID, status categorical)  -- PRIMARY KEY (entity_id)",
        available_entity_ids={"entities": ["entity-1"]},
        max_tries=3,
    )

    assert result.success is False
    assert result.attempts == expected_attempts
    assert calls == expected_attempts
    if not retryable:
        assert "not a forecast" in str(result.error)


_NIM_500_ERROR = (
    "An unexpected exception occurred. Please create an issue at "
    "'https://github.com/kumo-ai/kumo-rfm'. Unexpected server error."
)


def test_nim_internal_error_is_classified_as_infra_not_a_bad_query() -> None:
    """A bare HTTP 500 from the NIM must stop the repair loop, not trigger 5 rewrites."""
    assert _is_transient_exec_error(_NIM_500_ERROR)
    # It must NOT look like a capacity problem, or it would walk the neighbourhood ladder.
    assert not _is_context_capacity_error(_NIM_500_ERROR)


def test_neighbourhood_memo_skips_rungs_already_proven_too_big() -> None:
    calls: list[list[int] | None] = []

    def _fail_until_smallest(num_neighbors: list[int] | None) -> str:
        calls.append(num_neighbors)
        if num_neighbors != [8, 8]:
            raise RuntimeError(_ROW_LIMIT_ERROR)
        return "ok"

    memo = _NeighbourhoodMemo()
    assert _predict_resilient(_fail_until_smallest, memo=memo) == "ok"
    first_pass = list(calls)
    assert first_pass == [None, [24, 24], [16, 16], [8, 8]]

    # Second attempt in the same run must not re-pay the three rejected rungs.
    calls.clear()
    assert _predict_resilient(_fail_until_smallest, memo=memo) == "ok"
    assert calls == [[8, 8]]


def test_neighbourhood_memo_is_not_advanced_by_intermittent_gpu_faults() -> None:
    """CUDA/OOM faults are intermittent — the next attempt must still start at full accuracy."""
    state = {"fail": True}

    def _flaky(num_neighbors: list[int] | None) -> str:
        if state["fail"] and num_neighbors is None:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return "ok"

    memo = _NeighbourhoodMemo()
    assert _predict_resilient(_flaky, memo=memo) == "ok"
    assert memo.floor == 0

    state["fail"] = False
    calls: list[list[int] | None] = []

    def _record(num_neighbors: list[int] | None) -> str:
        calls.append(num_neighbors)
        return "ok"

    assert _predict_resilient(_record, memo=memo) == "ok"
    assert calls == [None]


_QUOTED_PQL = (
    "PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH `My People`.`Customer ID`"
)


def test_parse_entity_reads_a_quoted_name_as_the_data_spells_it() -> None:
    """Entity resolution feeds warehouse SQL and the entity-id map, so the
    backticks must not survive parsing."""
    assert parse_entity(_QUOTED_PQL) == ("My People", "Customer ID")


def test_parse_entity_still_reads_a_bare_name() -> None:
    assert parse_entity(
        "PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH PEOPLE.CUSTOMER_ID"
    ) == (
        "PEOPLE",
        "CUSTOMER_ID",
    )


def test_canonicalisation_keeps_the_quoting_a_name_needs() -> None:
    ddl = "My People(Customer ID ID, TIER categorical)"
    assert canonicalize_pql_identifiers(_QUOTED_PQL, ddl) == _QUOTED_PQL


def test_static_lint_reads_through_quotes() -> None:
    """The aggregated table is compared against the entity table, so a quoted
    name has to resolve or every quoted query would look cross-table."""
    pql_gen.validate_pql_static(_QUOTED_PQL)

    with pytest.raises(pql_gen.PqlStaticError, match="can only filter columns"):
        pql_gen.validate_pql_static(
            "PREDICT COUNT(ORDERS.* WHERE `My People`.TIER = 'pro', 0, 30, days) > 0 FOR EACH `My People`.`Customer ID`"
        )


def test_static_lint_accepts_a_quoted_event_table_that_links_to_the_entity() -> None:
    """Graph edges carry bare names, so a backticked aggregation table has to be
    unquoted before the direct-foreign-key check compares the two."""
    pql_gen.validate_pql_static(
        "PREDICT COUNT(`Sales Orders`.*, 0, 30, days) FOR EACH `Store Locations`.StoreID",
        edges=[("Sales Orders", "_StoreID", "Store Locations")],
    )

    with pytest.raises(pql_gen.PqlStaticError, match="no direct foreign key"):
        pql_gen.validate_pql_static(
            "PREDICT COUNT(`Line Items`.*, 0, 30, days) FOR EACH `Store Locations`.StoreID",
            edges=[("Sales Orders", "_StoreID", "Store Locations")],
        )


def test_unquote_and_quote_round_trip() -> None:
    assert pql_gen.unquote_name("`Customer ID`") == "Customer ID"
    assert pql_gen.unquote_name("CUSTOMER_ID") == "CUSTOMER_ID"
    assert pql_gen.quote_name("Customer ID") == "`Customer ID`"
    assert pql_gen.quote_name("CUSTOMER_ID") == "CUSTOMER_ID"
    assert pql_gen.quote_name("*") == "*"
