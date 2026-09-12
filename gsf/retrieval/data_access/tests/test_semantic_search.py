# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from gsf.retrieval.data_access import semantic_search
from gsf.catalog.constants import Labels


class _Retriever:
    def __init__(self, hits: list[dict[str, Any]], *, fmt: str = "sql") -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []
        self.vdb_kwargs = {"vdb": type("_Vdb", (), {"metadata_filter_format": fmt})()}

    def query(self, entity: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"entity": entity, **kwargs})
        return list(self.hits[: kwargs.get("top_k")])


def _table_hit(identifier: str, schema_name: str, distance: float) -> dict:
    return {
        "text": "events",
        "_distance": distance,
        "metadata": {
            "id": identifier,
            "name": "events",
            "label": Labels.TABLE,
            "database_name": "analytics",
            "schema_name": schema_name,
        },
    }


def test_ordinary_search_excludes_same_named_governed_prediction_view(
    monkeypatch,
) -> None:
    retriever = _Retriever(
        [
            _table_hit("prediction-events", "prediction", 0.01),
            _table_hit("main-events", "main", 0.02),
        ]
    )
    monkeypatch.setattr(
        semantic_search,
        "_configured_governed_view_paths",
        lambda _database: {("analytics", "prediction")},
    )

    rows = semantic_search.search_semantic_index(
        retriever,
        "events",
        label_filter=[Labels.TABLE],
        per_label_k=2,
        database_name="analytics",
    )

    assert [row["id"] for row in rows] == ["main-events"]
    where = retriever.calls[0]["vdb_kwargs"]["where"]
    assert '"database_name":"analytics"' in where
    assert '"schema_name":"prediction"' in where
    assert "NOT (" in where


def test_explicit_schema_search_can_inspect_governed_view(monkeypatch) -> None:
    retriever = _Retriever([_table_hit("prediction-events", "prediction", 0.01)])

    def unexpected(_database: str | None) -> set[tuple[str, str]]:
        raise AssertionError(
            "explicit schema search must not apply ordinary exclusions"
        )

    monkeypatch.setattr(semantic_search, "_configured_governed_view_paths", unexpected)

    rows = semantic_search.search_semantic_index(
        retriever,
        "events",
        label_filter=[Labels.TABLE],
        database_name="analytics",
        schema_name="prediction",
    )

    assert [row["id"] for row in rows] == ["prediction-events"]


def test_postgres_filter_excludes_governed_views_before_vector_limit(
    monkeypatch,
) -> None:
    """The dict backend must not depend on a guessed over-fetch multiplier."""

    retriever = _Retriever(
        [
            _table_hit("prediction-events-1", "prediction", 0.01),
            _table_hit("prediction-events-2", "prediction", 0.02),
            _table_hit("main-events", "main", 0.03),
        ],
        fmt="dict",
    )
    monkeypatch.setattr(
        semantic_search,
        "_configured_governed_view_paths",
        lambda _database: {("analytics", "prediction")},
    )

    # A real Postgres VDB applies this filter before LIMIT.  Model that server
    # behavior here so the regression asserts both the filter shape and k.
    original_query = retriever.query

    def query_without_governed_hits(entity: str, **kwargs: Any) -> list[dict[str, Any]]:
        retriever.hits = [
            hit
            for hit in retriever.hits
            if hit["metadata"]["schema_name"] != "prediction"
        ]
        return original_query(entity, **kwargs)

    retriever.query = query_without_governed_hits  # type: ignore[method-assign]

    rows = semantic_search.search_semantic_index(
        retriever,
        "events",
        label_filter=[Labels.TABLE],
        per_label_k=1,
        database_name="analytics",
    )

    assert [row["id"] for row in rows] == ["main-events"]
    assert retriever.calls[0]["top_k"] == 1
    assert retriever.calls[0]["vdb_kwargs"]["where"] == {
        "$and": [
            {"label": Labels.TABLE},
            {"database_name": "analytics"},
            {
                "$not": {
                    "$and": [
                        {"database_name": "analytics"},
                        {"schema_name": "prediction"},
                    ]
                }
            },
        ]
    }
