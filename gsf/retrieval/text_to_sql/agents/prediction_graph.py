# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KumoRFM graph-preparation node.

The prediction path is split into two nodes so the (potentially slow) infra
step streams its own progress:

  1. ``prepare_prediction_graph`` (this agent) — load the relevant tables, build
     the KumoRFM ``LocalGraph`` + model, and stash the resulting
     :class:`~gsf.retrieval.kumo.predictor.PredictionContext` in ``path_state``.
  2. ``kumo_predict`` — generate/repair the PQL, predict, and format.

On failure (no relevant tables, KumoRFM unavailable on this platform, etc.) it
writes a graceful ``final_response`` and signals ``predict_failed`` so the graph
routes straight to END without attempting a prediction.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage

from gsf.catalog.constants import Labels, TableTypes

from gsf.dal.datasources import fetch_table_by_name
from gsf.retrieval.kumo import PredictionContext
from gsf.retrieval.kumo import build_prediction_context
from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import load_graph_contract
from gsf.retrieval.kumo.pql_gen import _TABLE_COL
from gsf.retrieval.kumo.rag import fetch_pql_examples
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState, get_standalone_question

logger = logging.getLogger(__name__)


def _error_response(message: str) -> dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


def _pql_tables(pql: str) -> set[str]:
    """Table names referenced in a PQL (the left side of every ``TABLE.COLUMN``)."""
    return {m.group(1) for m in _TABLE_COL.finditer(pql or "")}


def _enrich_relevant_tables(
    relevant_tables: list[dict[str, Any]],
    examples: list[dict[str, str]],
    *,
    database_name: str,
) -> list[dict[str, Any]]:
    """Add any table referenced by a retrieved PQL example but missing from
    ``relevant_tables``, resolved from the catalog by name.

    Done BEFORE the graph is built so the added tables are loaded into the graph
    (and auto-linked by metadata inference) — the LLM is guided by these examples,
    so every table they reference must exist in the graph or the PQL won't parse.
    """
    existing = {str(t.get("name") or "").upper() for t in relevant_tables}
    referenced: set[str] = set()
    for ex in examples:
        referenced |= _pql_tables(ex.get("query") or "")

    enriched = list(relevant_tables)
    added: list[str] = []
    for name in sorted(referenced):
        if name.upper() in existing:
            continue
        row = fetch_table_by_name(name, database_name=database_name)
        if not row or not row.get("name"):
            continue
        enriched.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "schema_name": row.get("schema_name") or "",
                "database_name": row.get("database_name") or database_name,
                "description": row.get("description") or "",
                "pk": row.get("pk"),
                "label": Labels.TABLE,
            }
        )
        existing.add(name.upper())
        added.append(row["name"])
    if added:
        logger.info(
            "kumo: enriched graph with %d table(s) from PQL examples: %s",
            len(added),
            added,
        )
    return enriched


def _prediction_database_name(
    path_state: dict[str, Any],
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]],
) -> str:
    """Resolve one prediction database without a first-connector fallback."""

    explicit = str(
        path_state.get("target_db") or path_state.get("retrieval_database") or ""
    ).strip()
    if explicit:
        return explicit
    table_databases = {
        str(table.get("database_name") or "").strip()
        for table in relevant_tables
        if str(table.get("database_name") or "").strip()
    }
    if len(table_databases) == 1:
        return next(iter(table_databases))
    connector_databases = {
        str(getattr(connector, "database_name", "") or "").strip()
        for connector in connectors
        if str(getattr(connector, "database_name", "") or "").strip()
    }
    if len(connector_databases) == 1:
        return next(iter(connector_databases))
    raise ValueError("Prediction requires one explicit database scope.")


def _catalog_primary_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(
            str(item).strip() for item in value if str(item or "").strip()
        )
    return ()


def _contract_relevant_tables(contract: GraphContract) -> list[dict[str, Any]]:
    """Resolve every contracted view through its exact catalog path."""

    resolved: list[dict[str, Any]] = []
    for table in contract.tables:
        row = fetch_table_by_name(
            table.name,
            database_name=contract.database_name,
            schema_name=table.schema_name,
        )
        if row is None:
            raise ValueError(
                f"Contract view {contract.database_name}.{table.schema_name}."
                f"{table.name} is missing or ambiguous in the GSF catalog."
            )
        expected_path = (
            contract.database_name.casefold(),
            table.schema_name.casefold(),
            table.name.casefold(),
        )
        resolved_path = (
            str(row.get("database_name") or "").casefold(),
            str(row.get("schema_name") or "").casefold(),
            str(row.get("name") or "").casefold(),
        )
        if resolved_path != expected_path:
            raise ValueError(
                f"Contract view {contract.database_name}.{table.schema_name}."
                f"{table.name} resolved outside its governed catalog path."
            )
        if (
            str(row.get("table_type") or "").strip().casefold()
            != TableTypes.VIEW.casefold()
        ):
            raise ValueError(
                f"Contract object {contract.database_name}.{table.schema_name}."
                f"{table.name} must be a catalog VIEW."
            )
        catalog_key = _catalog_primary_key(row.get("pk"))
        if tuple(column.casefold() for column in catalog_key) != tuple(
            column.casefold() for column in table.primary_key
        ):
            raise ValueError(
                f"Contract primary key for {table.name!r} does not match "
                "the GSF catalog."
            )
        resolved.append(
            {
                "id": row.get("id"),
                "name": table.name,
                "schema_name": table.schema_name,
                "database_name": contract.database_name,
                "description": row.get("description") or "",
                "pk": list(table.primary_key),
                "expected_rows": table.rows,
                "table_type": TableTypes.VIEW,
                "label": Labels.TABLE,
            }
        )
    return resolved


def _filter_contract_examples(
    examples: list[dict[str, str]], contract: GraphContract
) -> list[dict[str, str]]:
    """Keep only examples whose PQL stays inside the contracted table set."""

    allowed = {table.name.casefold() for table in contract.tables}
    filtered = [
        example
        for example in examples
        if {
            name.casefold() for name in _pql_tables(example.get("query") or "")
        }
        <= allowed
    ]
    if len(filtered) != len(examples):
        logger.warning(
            "kumo: ignored %d scoped PQL example(s) outside the graph contract",
            len(examples) - len(filtered),
        )
    return filtered


class PredictionGraphAgent(BaseAgent):
    """Build the KumoRFM graph/model for the relevant tables (prediction phase 1)."""

    def __init__(self):
        super().__init__("prediction_graph")

    def execute(self, state: AgentState) -> dict[str, Any]:
        path_state = state.get("path_state", {})
        connectors = state.get("connectors", []) or []
        # Scope the KumoRFM graph to the tables candidate-preparation found relevant,
        # and use the catalog-derived join paths as the graph's table relationships.
        relevant_tables = path_state.get("relevant_tables") or []
        join_paths = path_state.get("attribute_join_paths") or []
        try:
            database_name = _prediction_database_name(
                path_state, connectors, relevant_tables
            )
            graph_contract = load_graph_contract(database_name)
        except Exception as exc:
            self.logger.exception("KumoRFM graph contract resolution failed")
            context = _error_response(
                "I couldn't prepare a prediction for this question. "
                f"({type(exc).__name__}: {exc})"
            )
            markdown = context.get("response", "")
            return {
                "decision": "predict_failed",
                "messages": state["messages"] + [AIMessage(content=markdown)],
                "path_state": {
                    **path_state,
                    "formatted_response": markdown,
                    "final_response": context,
                },
            }
        # Few-shot PQL examples retrieved from the verified PqlAnalysis corpus.
        examples = fetch_pql_examples(
            state.get("semantic_retriever"),
            get_standalone_question(state),
            database_name,
        )
        if graph_contract is not None:
            relevant_tables = _contract_relevant_tables(graph_contract)
            examples = _filter_contract_examples(examples, graph_contract)
        else:
            # Every table cited by an example must be present in the graph.
            relevant_tables = _enrich_relevant_tables(
                relevant_tables, examples, database_name=database_name
            )

        try:
            context = build_prediction_context(
                connectors,
                relevant_tables,
                database_name=database_name,
                graph_contract=graph_contract,
                join_paths=join_paths,
                examples=examples,
            )
        except Exception as exc:
            self.logger.exception("KumoRFM graph preparation failed")
            context = _error_response(
                "I couldn't prepare a prediction for this question. "
                f"({type(exc).__name__}: {exc})"
            )

        if isinstance(context, PredictionContext):
            return {
                "decision": "predict_ready",
                "path_state": {**path_state, "prediction_context": context},
            }

        # build_prediction_context returned a graceful error response dict —
        # terminate the prediction path with it.
        markdown = context.get("response", "")
        return {
            "decision": "predict_failed",
            "messages": state["messages"] + [AIMessage(content=markdown)],
            "path_state": {
                **path_state,
                "formatted_response": markdown,
                "final_response": context,
            },
        }
