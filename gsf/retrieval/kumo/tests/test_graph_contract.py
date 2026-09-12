# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest
from gsf.retrieval.kumo.graph_contract import GraphContractError
from gsf.retrieval.kumo.graph_contract import __all__ as graph_contract_exports
from gsf.retrieval.kumo.graph_contract import load_graph_contract


def _document() -> dict:
    return {
        "schema_version": 1,
        "database_name": "prediction_db",
        "database_sha256": "a" * 64,
        "source": {
            "dataset_id": "ai-factory",
            "dataset_version": "1.0.0",
            "revision": "fixture-revision",
            "manifest_sha256": "b" * 64,
            "prediction_manifest_sha256": "c" * 64,
        },
        "object_count": 2,
        "row_count": 5,
        "tables": [
            {
                "name": "entities",
                "schema_name": "prediction",
                "rows": 2,
                "primary_key": ["entity_id"],
                "time_column": None,
            },
            {
                "name": "events",
                "schema_name": "prediction",
                "rows": 3,
                "primary_key": ["event_id"],
                "time_column": "recorded_at",
            },
        ],
        "relationships": [
            {
                "source_table": "events",
                "source_column": "entity_id",
                "target_table": "entities",
                "target_column": "entity_id",
            }
        ],
        "time_columns": {"entities": None, "events": "recorded_at"},
        "forbidden_tables": ["future_labels"],
    }


def _scope() -> dict:
    return {
        "anchor_time": "2026-08-12T00:00:00-04:00",
        "entity_table": "entities",
        "entity_column": "entity_id",
        "population_view": "reviewed_entities",
        "population_column": "entity_id",
        "population_rows": 2,
    }


def test_prediction_scope_type_is_publicly_exported() -> None:
    assert "GraphContractPredictionScope" in graph_contract_exports


def test_unconfigured_contract_file_disables_contract_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KUMO_GRAPH_CONTRACTS_FILE", raising=False)

    assert load_graph_contract("prediction_db") is None


def test_loads_exact_database_contract_with_stable_revision(tmp_path) -> None:
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    first = load_graph_contract("PREDICTION_DB", path=path)
    second = load_graph_contract("prediction_db", path=path)

    assert first is not None
    assert first.database_name == "prediction_db"
    assert first.schema_name == "prediction"
    assert [table.name for table in first.tables] == ["entities", "events"]
    assert first.tables[0].time_column is None
    assert first.tables[1].time_column == "recorded_at"
    assert first.edges[0].source_columns == ("entity_id",)
    assert (
        first.revision
        == "sha256:cba9a65c9e4d8c54743e56954671d2f56b07ca44a7d5884b864be3c2f55f8f06"
    )
    assert first.revision == second.revision
    assert first.prediction_scope is None


def test_optional_prediction_scope_is_strict_canonical_and_revision_bound(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(_document()), encoding="utf-8")
    document = _document()
    document["prediction_scope"] = _scope()
    scoped_path = tmp_path / "scoped.json"
    scoped_path.write_text(json.dumps(document), encoding="utf-8")

    legacy = load_graph_contract("prediction_db", path=legacy_path)
    scoped = load_graph_contract("prediction_db", path=scoped_path)

    assert legacy is not None
    assert scoped is not None
    assert scoped.revision != legacy.revision
    assert scoped.prediction_scope is not None
    assert scoped.prediction_scope.anchor_time == "2026-08-12T04:00:00+00:00"
    assert scoped.prediction_scope.entity_table == "entities"
    assert scoped.prediction_scope.population_view == "reviewed_entities"


def test_valid_file_can_omit_selected_database(tmp_path) -> None:
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    assert load_graph_contract("another_db", path=path) is None


def test_bundle_selects_each_database_without_cross_database_leakage(tmp_path) -> None:
    first = _document()
    second = deepcopy(first)
    second["database_name"] = "supply_prediction_db"
    second["source"]["dataset_id"] = "supply-chain"
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps({"schema_version": 1, "contracts": [first, second]}),
        encoding="utf-8",
    )

    factory = load_graph_contract("PREDICTION_DB", path=path)
    supply = load_graph_contract("SUPPLY_PREDICTION_DB", path=path)

    assert factory is not None
    assert supply is not None
    assert factory.database_name == "prediction_db"
    assert supply.database_name == "supply_prediction_db"
    assert factory.schema_name == supply.schema_name == "prediction"


def test_bundle_fails_closed_when_database_is_not_configured(tmp_path) -> None:
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps({"schema_version": 1, "contracts": [_document()]}),
        encoding="utf-8",
    )

    with pytest.raises(GraphContractError, match="has no contract"):
        load_graph_contract("raw_database", path=path)


def test_bundle_rejects_duplicate_database_names(tmp_path) -> None:
    duplicate = deepcopy(_document())
    duplicate["database_name"] = "PREDICTION_DB"
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps({"schema_version": 1, "contracts": [_document(), duplicate]}),
        encoding="utf-8",
    )

    with pytest.raises(GraphContractError, match="duplicate database"):
        load_graph_contract("prediction_db", path=path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update({"unknown": True}),
        lambda doc: doc.update({"schema_version": 2}),
        lambda doc: doc["tables"][0].update({"unknown": True}),
        lambda doc: doc["tables"][0].pop("schema_name"),
        lambda doc: doc["tables"][0].update({"schema_name": "raw"}),
        lambda doc: doc["relationships"][0].update(
            {"target_column": "not_the_primary_key"}
        ),
        lambda doc: doc["relationships"][0].update({"source_table": "missing"}),
        lambda doc: doc.update({"row_count": 6}),
        lambda doc: doc["time_columns"].update({"events": "wrong_time"}),
    ],
)
def test_invalid_contracts_fail_closed(tmp_path, mutate) -> None:
    document = _document()
    mutate(document)
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GraphContractError):
        load_graph_contract("prediction_db", path=path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda scope: scope.update({"unknown": True}),
        lambda scope: scope.update({"anchor_time": "2026-08-12T00:00:00"}),
        lambda scope: scope.update({"entity_table": "missing"}),
        lambda scope: scope.update({"entity_column": "wrong_id"}),
        lambda scope: scope.update({"population_view": "entities"}),
        lambda scope: scope.update({"population_view": "future_labels"}),
        lambda scope: scope.update({"population_rows": 0}),
    ],
)
def test_invalid_prediction_scopes_fail_closed(tmp_path, mutate) -> None:
    document = _document()
    document["prediction_scope"] = _scope()
    mutate(document["prediction_scope"])
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GraphContractError):
        load_graph_contract("prediction_db", path=path)


def test_null_prediction_scope_fails_closed(tmp_path) -> None:
    document = _document()
    document["prediction_scope"] = None
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GraphContractError):
        load_graph_contract("prediction_db", path=path)


def test_unreadable_configured_contract_fails_closed(tmp_path) -> None:
    with pytest.raises(GraphContractError):
        load_graph_contract("prediction_db", path=tmp_path / "missing.json")
