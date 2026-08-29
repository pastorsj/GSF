# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gsf.server.metadata import router
from gsf.server.metadata import service

_DATABASE_ID = "42d62c22-b1c4-4ed7-8173-6aec9e2bd432"


@pytest.fixture
def catalog_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        router.datasources_dal,
        "fetch_databases",
        lambda zone_ids=None: [
            {
                "id": _DATABASE_ID,
                "name": "ai_factory",
            }
        ],
    )


def test_resolves_catalog_database_uuid_to_name(catalog_databases: None) -> None:
    assert router._resolve_coverage_target_db(_DATABASE_ID) == "ai_factory"


def test_resolves_database_name_case_insensitively(
    catalog_databases: None,
) -> None:
    assert router._resolve_coverage_target_db("AI_FACTORY") == "ai_factory"


def test_rejects_unknown_target_database(catalog_databases: None) -> None:
    with pytest.raises(HTTPException) as exc_info:
        router._resolve_coverage_target_db("unknown")

    assert exc_info.value.status_code == 422
    assert "does not match a catalog database UUID or name" in exc_info.value.detail


def test_route_forwards_canonical_target_database(
    catalog_databases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_entity_coverage(
        question: str,
        max_distance: float,
        target_db: str | None,
    ) -> dict:
        received.update(
            question=question,
            max_distance=max_distance,
            target_db=target_db,
        )
        return {"coverage": 1.0, "candidates": []}

    monkeypatch.setattr(router.dal, "entity_coverage", fake_entity_coverage)

    response = router.entity_coverage(
        router.EntityCoverageRequest(
            question="Which deployments are at risk?",
            target_db=_DATABASE_ID,
        )
    )

    assert response == {"data": {"coverage": 1.0, "candidates": []}}
    assert received["target_db"] == "ai_factory"


def _patch_payload_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service,
        "get_connectors",
        lambda: [SimpleNamespace(database_name="ai_factory")],
    )
    monkeypatch.setattr(service, "get_data_objects_retriever", lambda: object())
    monkeypatch.setattr(service, "get_semantic_objects_retriever", lambda: object())
    monkeypatch.setattr(service, "fetch_acronyms", lambda: [])
    monkeypatch.setattr(service, "fetch_custom_prompts", lambda: "")


def test_payload_scopes_coverage_to_configured_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_payload_dependencies(monkeypatch)

    payload = service._build_coverage_payload(
        "Which deployments are at risk?",
        target_db="AI_FACTORY",
    )

    assert payload["target_db"] == "ai_factory"


def test_payload_rejects_database_without_a_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_payload_dependencies(monkeypatch)

    with pytest.raises(service.PredictionFlowError) as exc_info:
        service._build_coverage_payload(
            "Which accounts are likely to churn?",
            target_db="aiq_supply_chain",
        )

    assert "does not match a configured database name" in str(exc_info.value)
