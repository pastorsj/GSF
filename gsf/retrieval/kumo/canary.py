# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct, fixed-input Kumo qualification for one governed DuckDB graph.

This canary bypasses text generation on purpose.  It proves that a reviewed
population, explicit forecast anchor, graph contract, installed SDK, and live
provider can produce typed prediction rows.  Its stdout is one bounded JSON
receipt containing no credentials, raw database rows, or entity identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pandas as pd
from gsf.catalog.constants import TableTypes

from gsf.connectors.duckdb import DuckDBDatabase
from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import GraphContractError
from gsf.retrieval.kumo.graph_contract import load_graph_contract
from gsf.retrieval.kumo.predictor import PredictionContext
from gsf.retrieval.kumo.predictor import build_prediction_context
from gsf.retrieval.kumo.predictor import current_provider_readiness
from gsf.retrieval.kumo.provider import KumoProviderCompatibilityError
from gsf.retrieval.kumo.provider import KumoProviderUnavailableError

_MAX_ENTITIES = 200
_MAX_REPETITIONS = 5


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _typed_value(value: Any) -> dict[str, Any]:
    value = _scalar(value)
    kind = "null" if value is None else type(value).__name__
    return {"type": kind, "value": value}


def _population_receipt(values: Sequence[Any]) -> dict[str, Any]:
    types = Counter(_typed_value(value)["type"] for value in values)
    return {
        "count": len(values),
        "type_counts": dict(sorted(types.items())),
    }


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def _safe_graph_receipt(value: dict[str, Any]) -> dict[str, Any]:
    tables = value.get("tables") if isinstance(value.get("tables"), list) else []
    edges = value.get("edges") if isinstance(value.get("edges"), list) else []
    receipt = {
        "schema_version": value.get("schema_version"),
        "mode": value.get("mode"),
        "database_name": value.get("database_name"),
        "contract_revision": value.get("contract_revision"),
        "graph_revision": value.get("graph_revision"),
        "tables": [
            {
                "name": table.get("name"),
                "schema_name": table.get("schema_name"),
                "primary_key": list(table.get("primary_key") or []),
                "time_column": table.get("time_column"),
                "loaded_rows": table.get("loaded_rows"),
            }
            for table in tables[:32]
            if isinstance(table, dict)
        ],
        "edges": [
            {
                "source_table": edge.get("source_table"),
                "source_columns": list(edge.get("source_columns") or []),
                "target_table": edge.get("target_table"),
                "target_columns": list(edge.get("target_columns") or []),
            }
            for edge in edges[:64]
            if isinstance(edge, dict)
        ],
    }
    scope = value.get("prediction_scope")
    if isinstance(scope, dict):
        receipt["prediction_scope"] = {
            key: scope.get(key)
            for key in (
                "anchor_time",
                "anchor_source",
                "entity_table",
                "entity_column",
                "population_view",
                "population_column",
                "population_count",
            )
        }
    return receipt


def _catalog_sources(contract: GraphContract) -> list[dict[str, Any]]:
    return [
        {
            "name": table.name,
            "schema_name": table.schema_name,
            "database_name": contract.database_name,
            "table_type": TableTypes.VIEW,
            "pk": list(table.primary_key),
            "expected_rows": table.rows,
        }
        for table in contract.tables
    ]


def _result_receipt(
    frame: Any,
    *,
    entities: Sequence[Any],
    anchor: pd.Timestamp,
    score_column: str,
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("prediction result is not a nonempty DataFrame")
    columns = {str(column).casefold(): str(column) for column in frame.columns}
    entity_column = columns.get("entity")
    anchor_column = columns.get("anchor_timestamp")
    resolved_score = columns.get(score_column.casefold())
    if entity_column is None or anchor_column is None or resolved_score is None:
        raise ValueError("prediction result is missing a required typed column")

    returned = frame[entity_column].tolist()
    if Counter(
        json.dumps(_typed_value(value), sort_keys=True) for value in returned
    ) != Counter(json.dumps(_typed_value(value), sort_keys=True) for value in entities):
        raise ValueError(
            "prediction result population differs from the requested population"
        )

    scores = pd.to_numeric(frame[resolved_score], errors="coerce")
    if len(scores) != len(frame) or not all(
        math.isfinite(float(value)) for value in scores
    ):
        raise ValueError("prediction scores are not finite numbers")
    if not all(0.0 <= float(value) <= 1.0 for value in scores):
        raise ValueError("prediction scores are outside the unit interval")

    anchors = [_timestamp(value) for value in frame[anchor_column].tolist()]
    if any(value != anchor for value in anchors):
        raise ValueError(
            "prediction result anchor differs from the explicit canary anchor"
        )

    return {
        "passed": True,
        "row_count": len(frame),
        "columns": [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ],
        "score": {
            "column": resolved_score,
            "finite": True,
            "unit_interval": True,
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
        },
        "anchor": {
            "column": anchor_column,
            "timestamp": anchor.isoformat(),
            "matching_rows": len(anchors),
        },
    }


def run_canary(
    *,
    database_path: str,
    database_name: str,
    graph_contracts_path: str,
    pql: str,
    entities: Sequence[str | int],
    anchor: str,
    score_column: str,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Execute the fixed canary and return display-safe typed proof."""

    if not 1 <= len(entities) <= _MAX_ENTITIES:
        raise ValueError(f"entities must contain 1 through {_MAX_ENTITIES} values")
    if any(
        isinstance(value, bool) or not isinstance(value, (str, int))
        for value in entities
    ):
        raise ValueError("entities must contain only strings or integers")
    typed_entities = {
        json.dumps(_typed_value(value), sort_keys=True, separators=(",", ":"))
        for value in entities
    }
    if len(typed_entities) != len(entities):
        raise ValueError("entities must be unique with type preserved")
    if not 1 <= repetitions <= _MAX_REPETITIONS:
        raise ValueError(f"repetitions must be from 1 through {_MAX_REPETITIONS}")
    if not pql.strip().upper().startswith("PREDICT "):
        raise ValueError("pql must be one explicit PREDICT statement")
    if not score_column or len(score_column) > 128:
        raise ValueError("score_column must be a bounded nonempty name")

    explicit_anchor = _timestamp(anchor)
    connector: DuckDBDatabase | None = None
    try:
        contract = load_graph_contract(database_name, path=graph_contracts_path)
        if contract is None:
            raise GraphContractError("no graph contract was selected")
        connector = DuckDBDatabase(database_path, read_only=True)
        context = build_prediction_context(
            [connector],
            _catalog_sources(contract),
            database_name=database_name,
            graph_contract=contract,
        )
        if not isinstance(context, PredictionContext):
            raise RuntimeError("prediction context was not built")
        context.kumo_model.validate_pql(pql)

        results: list[dict[str, Any]] = []
        for attempt in range(1, repetitions + 1):
            frame = context.kumo_model.predict(
                pql,
                indices=list(entities),
                anchor_time=explicit_anchor,
                run_mode="debug",
                num_retries=0,
                verbose=False,
            )
            results.append(
                {
                    "attempt": attempt,
                    **_result_receipt(
                        frame,
                        entities=entities,
                        anchor=explicit_anchor,
                        score_column=score_column,
                    ),
                }
            )

        provider = current_provider_readiness()
        if provider is None or not provider.ready:
            raise RuntimeError("provider readiness proof is unavailable")
        return {
            "schema_version": 1,
            "status": "passed",
            "database_name": database_name,
            "provider": provider.to_receipt(),
            "graph": _safe_graph_receipt(context.graph_receipt),
            "query": {
                "pql_sha256": "sha256:" + hashlib.sha256(pql.encode()).hexdigest(),
                "explicit_anchor": explicit_anchor.isoformat(),
                "score_column": score_column,
            },
            "population": _population_receipt(entities),
            "repetitions": results,
        }
    finally:
        if connector is not None:
            connector.close()


def _entities(value: str) -> list[str | int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("entity JSON must be a list")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-path", required=True)
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--graph-contracts", required=True)
    parser.add_argument("--pql", required=True)
    parser.add_argument("--entity-json", required=True, type=_entities)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--score-column", default="TRUE_PROB")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def _safe_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, (KumoProviderCompatibilityError, KumoProviderUnavailableError)):
        return {
            "code": exc.code,
            "retryable": exc.retryable,
            "provider": exc.readiness.to_receipt(),
        }
    if isinstance(exc, GraphContractError):
        return {"code": "KUMO_CANARY_GRAPH_CONTRACT_INVALID", "retryable": False}
    if isinstance(exc, ValueError):
        return {"code": "KUMO_CANARY_CONTRACT_FAILED", "retryable": False}
    return {"code": "KUMO_CANARY_EXECUTION_FAILED", "retryable": False}


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = run_canary(
            database_path=args.database_path,
            database_name=args.database_name,
            graph_contracts_path=args.graph_contracts,
            pql=args.pql,
            entities=args.entity_json,
            anchor=args.anchor,
            score_column=args.score_column,
            repetitions=args.repetitions,
        )
    except Exception as exc:  # stdout is an allow-listed machine receipt.
        receipt = {"schema_version": 1, "status": "failed", "error": _safe_error(exc)}
        print(json.dumps(receipt, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
