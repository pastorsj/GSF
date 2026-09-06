# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM prediction pipeline (GSF entry point).

Wires the ingested-catalog data into the ported text-to-PQL pipeline:
  1. Load a bounded sample of each ingested-catalog table into DataFrames.
  2. Build a KumoRFM ``LocalGraph`` (metadata + links inferred) and a DuckDB
     mirror of the same frames (so the entity-selection SQL resolves).
  3. Run :func:`gsf.retrieval.kumo.pql_gen.generate_pql` — LLM writes the PQL,
     the static lint + cheap parse validate it, an entity-selection SQL scopes
     the entities, and KumoRFM predicts, with the full repair loop.
  4. Format the result into the standard response dict.

Everything is bounded and defensive: any failure returns a graceful response
dict in the same shape as the SQL path, so the chat never hard-errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes

from gsf.retrieval.kumo.graph_contract import GraphContract
from gsf.retrieval.kumo.graph_contract import GraphContractPredictionScope
from gsf.retrieval.kumo.kumo_model import key_columns
from gsf.retrieval.kumo.provider import KumoProviderCompatibilityError
from gsf.retrieval.kumo.provider import KumoProviderReadiness
from gsf.retrieval.kumo.provider import require_kumo_provider_ready

logger = logging.getLogger(__name__)

# Bounds so building the graph on a large database stays tractable. The whole
# (capped) dataset is uploaded to the hosted KumoRFM service.
_MAX_TABLES = int(os.environ.get("KUMO_MAX_TABLES", "20"))
# Rows sampled per table into the graph. Unlimited by default; set
# KUMO_MAX_ROWS_PER_TABLE to a positive integer to cap it.
_raw_max_rows = os.environ.get("KUMO_MAX_ROWS_PER_TABLE")
_MAX_ROWS_PER_TABLE: int | None = int(_raw_max_rows) if _raw_max_rows and int(_raw_max_rows) > 0 else None
_MAX_PREVIEW_ROWS = int(os.environ.get("KUMO_MAX_PREVIEW_ROWS", "50"))
_MAX_ENTITIES = int(os.environ.get("KUMO_MAX_ENTITIES", "2000"))

_init_lock = threading.Lock()
_initialized = False


_client: Any = None
_provider_readiness: KumoProviderReadiness | None = None
_provider_failure: KumoProviderCompatibilityError | None = None


def current_provider_readiness() -> KumoProviderReadiness | None:
    """Return the last successful, credential-free provider readiness proof."""

    return _provider_readiness


def _ensure_init() -> Any:
    """Compatibility-check and open the SDFM client once from environment.

    The readiness probe runs before graph upload or prediction. A matching
    official SDK uses its normal adapter. The one reviewed legacy/current model
    transition uses a client-local, exact-version adapter; unknown combinations
    stop as deterministic SDK/NIM incompatibilities.
    """
    global _client, _provider_failure, _provider_readiness
    if _client is not None:
        return _client
    if _provider_failure is not None:
        raise _provider_failure
    with _init_lock:
        if _client is not None:
            return _client
        if _provider_failure is not None:
            raise _provider_failure
        url = os.environ.get("KUMO_RFM_API_URL")
        if not url:
            raise RuntimeError("KUMO_RFM_API_URL is not set")
        api_key = os.environ.get("KUMO_RFM_API_KEY") or None

        try:
            _provider_readiness = require_kumo_provider_ready(url, api_key)
        except KumoProviderCompatibilityError as exc:
            # A fixed SDK/server contract mismatch cannot heal within this
            # process. Cache only that deterministic result; transient health
            # failures remain retryable on a later request.
            _provider_failure = exc
            raise

        from nvidia_sdfm import SDFMClient

        before = time.perf_counter()
        client_options: dict[str, Any] = {}
        if _provider_readiness.compatibility_adapter is not None:
            from gsf.retrieval.kumo.compatibility import compatibility_registry

            client_options["registry"] = compatibility_registry()
        _client = SDFMClient(url, api_key=api_key, **client_options)
        logger.info(
            "KumoRFM client opened after provider compatibility check (model=%s) in %.2fs",
            _provider_readiness.wire_model or _provider_readiness.expected_model,
            time.perf_counter() - before,
        )
        return _client


def _quote(schema: str, table: str) -> str:
    """Schema-qualify a table name (double quotes work for Postgres/Snowflake)."""
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _catalog_key_columns(entry: dict[str, Any]) -> list[str]:
    """Primary-key columns the catalog recorded for a table.

    Every path spells the field ``pk``, the catalog's own name for it — both
    retrieval (see ``relevant_tables``) and the PQL-example enrichment. It may
    hold a single name or several for a composite key.
    """
    raw = entry.get("pk")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(c).strip() for c in raw if str(c or "").strip()]
    return []


def _contract_view_sources(
    relevant_tables: list[dict[str, Any]],
    contract: GraphContract,
) -> list[dict[str, Any]]:
    """Return catalog sources that exactly match a contract's governed views.

    This is the final boundary before warehouse rows are read.  A caller cannot
    add a raw table, substitute a same-named object from another schema/database,
    or downgrade a contracted view to a base table.  The returned entries are
    ordered and named from the deployment-owned contract so graph/PQL logical
    names remain stable while SQL uses the exact qualified view path.
    """

    expected: dict[tuple[str, str, str], Any] = {
        (
            contract.database_name.casefold(),
            table.schema_name.casefold(),
            table.name.casefold(),
        ): table
        for table in contract.tables
    }
    resolved: dict[tuple[str, str, str], dict[str, Any]] = {}
    scope_path = None
    if contract.prediction_scope is not None:
        scope_path = (
            contract.database_name.casefold(),
            contract.schema_name.casefold(),
            contract.prediction_scope.population_view.casefold(),
        )
    for index, entry in enumerate(relevant_tables):
        path = (
            str(entry.get("database_name") or "").strip().casefold(),
            str(entry.get("schema_name") or "").strip().casefold(),
            str(entry.get("name") or "").strip().casefold(),
        )
        if not all(path):
            raise ValueError(f"Contract catalog source {index} has no exact database/schema/view path.")
        if path == scope_path:
            continue
        spec = expected.get(path)
        if spec is None:
            raise ValueError(f"Prediction graph source is outside the governed view contract: {'.'.join(path)}.")
        if path in resolved:
            raise ValueError(f"Prediction graph contains a duplicate governed view source: {'.'.join(path)}.")
        if str(entry.get("table_type") or "").strip().casefold() != TableTypes.VIEW.casefold():
            raise ValueError(f"Prediction graph source must be a catalog VIEW: {'.'.join(path)}.")
        if tuple(column.casefold() for column in _catalog_key_columns(entry)) != tuple(
            column.casefold() for column in spec.primary_key
        ):
            raise ValueError(f"Prediction graph source {'.'.join(path)} has a primary key outside its contract.")
        expected_rows = entry.get("expected_rows")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows != spec.rows:
            raise ValueError(f"Prediction graph source {'.'.join(path)} has a row contract mismatch.")
        resolved[path] = entry

    if set(resolved) != set(expected):
        missing = [f"{database}.{schema}.{view}" for database, schema, view in sorted(set(expected) - set(resolved))]
        raise ValueError(f"Prediction graph is missing governed contract view(s): {missing}.")

    ordered: list[dict[str, Any]] = []
    for table in contract.tables:
        path = (
            contract.database_name.casefold(),
            table.schema_name.casefold(),
            table.name.casefold(),
        )
        entry = resolved[path]
        ordered.append(
            {
                **entry,
                "name": table.name,
                "schema_name": table.schema_name,
                "database_name": contract.database_name,
                "pk": list(table.primary_key),
                "expected_rows": table.rows,
                "table_type": TableTypes.VIEW,
            }
        )
    return ordered


def _validate_connector_contract_views(
    connector: Any,
    contract: GraphContract,
) -> None:
    """Prove each contracted source is still a physical catalog view.

    The GSF catalog is the semantic control plane, but the connector is the
    authority for the object Kumo is about to read.  Re-enumerating metadata here
    prevents catalog drift from turning an approved view path into a raw table.
    """

    get_tables = getattr(connector, "get_tables", None)
    if not callable(get_tables):
        raise ValueError("Prediction connector cannot enumerate governed catalog views.")
    objects = get_tables()
    if not isinstance(objects, pd.DataFrame):
        raise ValueError("Prediction connector returned invalid catalog metadata.")
    column_names = {str(column).casefold(): column for column in objects.columns}
    required = {"table_schema", "table_name", "table_type"}
    if not required <= set(column_names):
        raise ValueError("Prediction connector catalog metadata has no schema/name/type boundary.")

    available_views: set[tuple[str, str]] = set()
    duplicate_views: set[tuple[str, str]] = set()
    for row in objects.to_dict(orient="records"):
        if str(row.get(column_names["table_type"]) or "").strip().casefold() != TableTypes.VIEW.casefold():
            continue
        path = (
            str(row.get(column_names["table_schema"]) or "").strip().casefold(),
            str(row.get(column_names["table_name"]) or "").strip().casefold(),
        )
        if not all(path):
            continue
        if path in available_views:
            duplicate_views.add(path)
        available_views.add(path)

    expected = {(table.schema_name.casefold(), table.name.casefold()) for table in contract.tables}
    if contract.prediction_scope is not None:
        expected.add((contract.schema_name.casefold(), contract.prediction_scope.population_view.casefold()))
    ambiguous = expected & duplicate_views
    if ambiguous:
        raise ValueError(f"Prediction connector returned ambiguous governed view(s): {sorted(ambiguous)}.")
    missing = expected - available_views
    if missing:
        raise ValueError(f"Prediction connector is missing contracted VIEW object(s): {sorted(missing)}.")


def _load_relevant_frames(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]],
    *,
    database_name: str | None = None,
    strict: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, list[str]]]:
    """Load a bounded sample of each relevant table into a DataFrame.

    Iterates only the tables the candidate-preparation step already found relevant
    (no full-catalog scan). Each table's connector is resolved by ``database_name``,
    using the explicitly selected database or the sole configured connector.

    Returns ``(frames, name_map, key_columns)`` where ``name_map`` maps each graph
    table name to its schema-qualified SQL name (used to schema-qualify
    entity-selection SQL) and ``key_columns`` maps it to the catalog's primary-key
    columns (see :func:`_declare_primary_keys`). Both are keyed by the graph table
    name chosen here, so neither has to re-derive it.
    """
    if not connectors or not relevant_tables:
        return {}, {}, {}

    selected_connector = _select_connector(connectors, database_name)

    frames: dict[str, pd.DataFrame] = {}
    name_map: dict[str, str] = {}
    key_columns: dict[str, list[str]] = {}
    for t in relevant_tables:
        if len(frames) >= _MAX_TABLES:
            if strict:
                raise ValueError(f"Explicit graph contract exceeds KUMO_MAX_TABLES={_MAX_TABLES}.")
            logger.warning(
                "kumo: reached table cap (%d); remaining tables skipped",
                _MAX_TABLES,
            )
            break
        table = str(t.get("name") or "").strip()
        if not table:
            continue
        schema = str(t.get("schema_name") or "").strip()
        table_database = str(t.get("database_name") or "").strip()
        if database_name and table_database and (table_database.casefold() != database_name.casefold()):
            if strict:
                raise ValueError(f"Table {table!r} belongs to {table_database!r}, not {database_name!r}.")
            continue
        connector = selected_connector
        if table in frames and strict:
            raise ValueError(f"Explicit graph contract contains duplicate table {table!r}.")
        name = table if table not in frames else f"{schema}_{table}"
        expected_rows = t.get("expected_rows")
        if (
            strict
            and isinstance(expected_rows, int)
            and _MAX_ROWS_PER_TABLE is not None
            and expected_rows > _MAX_ROWS_PER_TABLE
        ):
            raise ValueError(
                f"Explicit graph-contract table {schema}.{table} declares {expected_rows} rows, above "
                f"KUMO_MAX_ROWS_PER_TABLE={_MAX_ROWS_PER_TABLE}; refusing a partial graph."
            )
        limit = f" LIMIT {_MAX_ROWS_PER_TABLE}" if _MAX_ROWS_PER_TABLE else ""
        logger.info(
            "kumo: loading rows for %s.%s (limit=%s)...",
            schema,
            table,
            _MAX_ROWS_PER_TABLE or "none",
        )
        t_start = time.perf_counter()
        try:
            df = connector.execute(f"SELECT * FROM {_quote(schema, table)}{limit}")
        except Exception:
            if strict:
                raise
            logger.exception("kumo: failed to load rows for %s.%s", schema, table)
            continue
        elapsed = time.perf_counter() - t_start
        if df is None or df.empty:
            if strict:
                raise ValueError(f"Explicit graph-contract table {schema}.{table} has no rows.")
            logger.info("kumo: %s.%s returned 0 rows in %.2fs", schema, table, elapsed)
            continue
        if strict and isinstance(expected_rows, int) and len(df) != expected_rows:
            raise ValueError(
                f"Explicit graph-contract table {schema}.{table} expected {expected_rows} rows but loaded {len(df)}."
            )
        logger.info(
            "kumo: loaded %d row(s) x %d col(s) from %s.%s in %.2fs",
            len(df),
            len(df.columns),
            schema,
            table,
            elapsed,
        )
        frames[name] = df
        if schema:
            name_map[name] = _quote(schema, table)
        catalog_keys = _catalog_key_columns(t)
        if catalog_keys:
            key_columns[name] = catalog_keys
    return frames, name_map, key_columns


def _select_connector(connectors: list[Any], database_name: str | None) -> Any:
    """Resolve one connector by database name, never by list position."""

    if not connectors:
        raise ValueError("No database connection is configured.")
    if database_name:
        matches = [
            connector
            for connector in connectors
            if str(getattr(connector, "database_name", "") or "").casefold() == database_name.casefold()
        ]
        if len(matches) != 1:
            available = sorted(str(getattr(connector, "database_name", "") or "") for connector in connectors)
            raise ValueError(
                f"Prediction database {database_name!r} does not resolve to exactly one "
                f"connector (available: {available})."
            )
        return matches[0]
    if len(connectors) != 1:
        raise ValueError("Prediction requires an explicit database with multiple connectors.")
    return connectors[0]


def _error_response(message: str) -> dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


def _json_safe(value: Any) -> Any:
    """Convert a prediction-frame cell into a JSON-serializable Python value.

    KumoRFM returns pandas ``Timestamp`` (e.g. ``ANCHOR_TIMESTAMP``) and numpy
    scalars (``float64`` / ``bool_`` / ``int64``) that ``json.dumps`` can't
    encode. Timestamps/datetimes become ISO strings, numpy scalars become native
    Python, and NaN/NaT become ``None``.
    """
    import datetime

    import numpy as np

    try:
        if not isinstance(value, (list, dict, tuple)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in rows]


def _format_result(result: Any) -> dict[str, Any]:
    """Shape a :class:`PqlGenerationResult` into the standard response dict."""
    if not result.success:
        message = result.error or "KumoRFM could not answer this prediction."
        final = _error_response(f"Prediction could not be completed. {message}")
        final["sql_code"] = result.pql or ""
        return final

    parts = [f"Prediction for: {result.question}"]
    if result.pql:
        parts.append(f"PQL: `{result.pql}`")
    if result.note:
        parts.append(result.note)
    parts.append(
        f"KumoRFM scored {result.num_entities} entit"
        f"{'y' if result.num_entities == 1 else 'ies'}." + (" (truncated preview)" if result.truncated else "")
    )
    return {
        "response": "\n\n".join(parts),
        "sql_code": result.pql or "",
        "sql_columns": result.columns,
        "custom_analyses_used": [],
        "sql_response_from_db": _json_safe_rows(result.rows) or None,
    }


@dataclass
class PredictionContext:
    """Everything :func:`run_prediction` needs, built once from the relevant tables.

    Holds live, non-serializable KumoRFM handles (``kumo_model``) plus the graph
    context strings. It is passed between the ``prepare_prediction_graph`` and
    ``kumo_predict`` graph nodes via ``path_state`` within a single run only — it
    is never checkpointed or serialized.
    """

    kumo_model: Any
    connector: Any
    graph_ddl: str
    graph_edges: Any
    graph_col_stypes: Any
    # Bounded low-cardinality values from the exact frames supplied to Kumo.
    # These keep both PQL and entity-selection SQL literals faithful to the
    # warehouse without exposing identifiers or high-cardinality measures.
    column_reference: str
    time_columns: Any
    table_names: dict[str, str]
    # Per table (casefolded), the identity of every loaded row: bare values for a
    # single-column key, one tuple per row for a composite one.
    entity_ids: dict[str, list[Any]]
    examples: list[dict[str, Any]]
    database_name: str
    graph_receipt: dict[str, Any]
    prediction_scope: GraphContractPredictionScope | None = None
    prediction_scope_ids: tuple[Any, ...] = ()


def _fkey_name(fkey: Any) -> str:
    """Comparable text for an edge's foreign key.

    A composite key reads back as the single surrogate column rather than the tuple
    that declared it, so this is normally already a string; the tuple branch keeps
    the comparison total if a later SDK reports the columns themselves.
    """
    return ", ".join(str(c) for c in fkey) if isinstance(fkey, tuple) else str(fkey)


def _resolve_column(table: Any, col: str) -> str | None:
    """Case-insensitive column-name match within a graph table (connectors lowercase)."""
    target = (col or "").lower()
    for c in table.columns:
        if c.name.lower() == target:
            return c.name
    return None


def _apply_graph_contract(graph: Any, contract: GraphContract) -> None:
    """Apply exactly the contract's identities, time columns, and edges."""

    lookup = {name.casefold(): name for name in graph.tables}
    if set(lookup) != {table.name.casefold() for table in contract.tables}:
        raise ValueError("Loaded Kumo graph tables do not exactly match the contract.")

    for spec in contract.tables:
        name = lookup[spec.name.casefold()]
        table = graph[name]
        primary_key = [_resolve_column(table, column) for column in spec.primary_key]
        if any(column is None for column in primary_key):
            raise ValueError(f"Contract primary key {list(spec.primary_key)} is missing from {name!r}.")
        resolved_key = [str(column) for column in primary_key]
        table.primary_key = resolved_key[0] if len(resolved_key) == 1 else tuple(resolved_key)

        if spec.time_column is None:
            table.time_column = None
        else:
            time_column = _resolve_column(table, spec.time_column)
            if time_column is None:
                raise ValueError(f"Contract time column {spec.time_column!r} is missing from {name!r}.")
            table.time_column = time_column

    for edge in contract.edges:
        source_name = lookup[edge.source_table.casefold()]
        target_name = lookup[edge.target_table.casefold()]
        source_table = graph[source_name]
        source_columns = [_resolve_column(source_table, column) for column in edge.source_columns]
        if any(column is None for column in source_columns):
            raise ValueError(
                f"Contract edge source columns {list(edge.source_columns)} are missing from {source_name!r}."
            )
        resolved_source = [str(column) for column in source_columns]
        fkey: Any = resolved_source[0] if len(resolved_source) == 1 else tuple(resolved_source)
        graph.link(source_name, fkey, target_name)

    graph.validate()


def _receipt_edges(graph: Any) -> list[dict[str, Any]]:
    """Convert inferred graph edges into display-safe column relationships."""

    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        raw_source = list(edge.fkey) if isinstance(edge.fkey, tuple) else [str(edge.fkey)]
        edges.append(
            {
                "source_table": str(edge.src_table),
                "source_columns": raw_source,
                "target_table": str(edge.dst_table),
                "target_columns": key_columns(graph[edge.dst_table]),
            }
        )
    return edges


def _build_graph_receipt(
    *,
    database_name: str,
    graph: Any,
    frames: dict[str, pd.DataFrame],
    examples: list[dict[str, Any]],
    contract: GraphContract | None,
    prediction_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a credential- and row-free receipt for the applied graph."""

    if contract is not None:
        table_specs = {table.name.casefold(): table for table in contract.tables}
        edges = [
            {
                "source_table": edge.source_table,
                "source_columns": list(edge.source_columns),
                "target_table": edge.target_table,
                "target_columns": list(edge.target_columns),
            }
            for edge in contract.edges
        ]
    else:
        table_specs = {}
        edges = _receipt_edges(graph)

    tables: list[dict[str, Any]] = []
    for name, table in graph.tables.items():
        spec = table_specs.get(name.casefold())
        time_column = (
            spec.time_column if spec is not None else getattr(getattr(table, "time_column", None), "name", None)
        )
        tables.append(
            {
                "name": name,
                "schema_name": spec.schema_name if spec is not None else None,
                "primary_key": list(spec.primary_key) if spec else key_columns(table),
                "time_column": time_column,
                "loaded_rows": len(frames[name]),
            }
        )

    example_ids = sorted(
        {str(example["id"]) for example in examples if isinstance(example.get("id"), str) and example["id"]}
    )
    graph_identity: dict[str, Any] = {
        "schema_version": 1,
        "database_name": database_name,
        "mode": "explicit" if contract is not None else "catalog_inferred",
        "contract_revision": contract.revision if contract else None,
        "tables": sorted(tables, key=lambda item: item["name"].casefold()),
        "edges": sorted(
            edges,
            key=lambda item: (
                item["source_table"].casefold(),
                item["target_table"].casefold(),
                tuple(column.casefold() for column in item["source_columns"]),
            ),
        ),
    }
    if prediction_scope is not None:
        graph_identity["prediction_scope"] = prediction_scope
    digest = hashlib.sha256(
        json.dumps(graph_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **graph_identity,
        "pql_example_ids": example_ids,
        "pql_example_count": len(example_ids),
        "graph_revision": f"sha256:{digest}",
    }


def _declare_primary_keys(graph: Any, key_columns_by_table: dict[str, list[str]]) -> int:
    """Set each table's primary key from the catalog, as a tuple when composite.

    Metadata inference picks no key at all when several columns are ``stype=ID`` and
    none resembles the table name, and it never infers a composite key. A table keyed
    on ``('Customer ID', 'REGION')`` would therefore reach KumoRFM with no identity,
    which costs it every edge (nothing can be oriented towards a key that isn't
    there) and leaves it unusable as a prediction entity.

    Declaring only what the catalog actually recorded: a table whose key columns are
    absent from the loaded frame is left to inference. Returns the number of tables
    whose key was declared.
    """
    declared = 0
    for name, table in graph.tables.items():
        wanted = key_columns_by_table.get(name) or []
        if not wanted:
            continue
        resolved = [c for c in (_resolve_column(table, w) for w in wanted) if c]
        if len(resolved) != len(wanted):
            logger.debug(
                "kumo: catalog key %s not fully present in %s; leaving inference",
                wanted,
                name,
            )
            continue
        if [c.lower() for c in key_columns(table)] == [c.lower() for c in resolved]:
            continue
        try:
            table.primary_key = resolved[0] if len(resolved) == 1 else tuple(resolved)
        except Exception:
            logger.debug("kumo: could not declare key %s on %s", resolved, name, exc_info=True)
            continue
        declared += 1
        logger.info("kumo: declared %s primary key %s from catalog", name, resolved)
    return declared


def _path_columns(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a join-path entry into its ``(table, column)`` node sequence.

    ``entry["path"]`` is a list of hop dicts ``{source_table, source_column,
    target_table, target_column, ...}`` that ``find_join_path`` produced by pairing
    consecutive columns of a Neo4j traversal. Flattening the hops back to
    ``[h0.source, h0.target, h1.source, h1.target, ...]`` restores that column
    sequence, so consecutive columns in DIFFERENT tables are the cross-table
    semantic foreign-key joins (columns within the same table are attribute hops).
    """
    seq: list[tuple[str, str]] = []
    for hop in entry.get("path") or []:
        seq.append((str(hop.get("source_table") or ""), str(hop.get("source_column") or "")))
        seq.append((str(hop.get("target_table") or ""), str(hop.get("target_column") or "")))
    return seq


def _apply_join_paths(graph: Any, join_paths: list[dict[str, Any]] | None) -> int:
    """Link graph tables using the cross-table joins in the catalog join paths.

    ``join_paths`` is ``attribute_join_paths`` from the text-to-SQL state. Each entry
    encodes a traversal from the anchor column to a destination column; the joins are
    the adjacent columns that cross tables (see :func:`_path_columns`). For each such
    join the side whose column is that table's primary key becomes the KumoRFM
    destination and the other side's column becomes the foreign key. Joins that don't
    map cleanly (unknown table/column, neither side a primary key, incompatible key
    dtype) are skipped.

    An edge already present in the graph (e.g. auto-inferred by ``from_data``) still
    counts as covered, so the caller doesn't fall back to heuristic inference for a
    relationship the catalog already describes.

    A composite destination key is linked once with the whole tuple: KumoRFM rejects
    a link that names only part of an identity, and each part arrives here as its own
    single-column join, so the parts are collected per table pair before linking. A
    part the join paths never mention is taken from the source's same-named column
    when it has one, which is what a sharded warehouse looks like (``ORDERS`` and
    ``PEOPLE`` both carrying ``REGION``).

    Returns the number of distinct relationships covered (added or pre-existing).
    """
    lookup = {name.lower(): name for name in graph.tables}
    existing = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    existing_pairs = {(e.src_table, e.dst_table) for e in graph.edges}
    # (src, dst) -> {destination key column (lowercased): source column}
    pending: dict[tuple[str, str], dict[str, str]] = {}
    for entry in join_paths or []:
        seq = _path_columns(entry)
        for (a_tbl, a_col_raw), (b_tbl, b_col_raw) in zip(seq, seq[1:]):
            a_name = lookup.get(a_tbl.lower())
            b_name = lookup.get(b_tbl.lower())
            if not a_name or not b_name or a_name == b_name:
                continue
            a_graph, b_graph = graph[a_name], graph[b_name]
            a_col = _resolve_column(a_graph, a_col_raw)
            b_col = _resolve_column(b_graph, b_col_raw)
            if not a_col or not b_col:
                continue
            # Orient the edge FK(src) -> PK(dst): the side whose join column is part
            # of that table's primary key is the destination.
            a_keys = {c.lower() for c in key_columns(a_graph)}
            b_keys = {c.lower() for c in key_columns(b_graph)}
            if b_col.lower() in b_keys:
                src, src_col, dst, dst_col = a_name, a_col, b_name, b_col
            elif a_col.lower() in a_keys:
                src, src_col, dst, dst_col = b_name, b_col, a_name, a_col
            else:
                continue
            pending.setdefault((src, dst), {}).setdefault(dst_col.lower(), src_col)

    covered = 0
    for (src, dst), mapping in pending.items():
        dst_keys = key_columns(graph[dst])
        fkey: Any
        if len(dst_keys) > 1:
            parts = [mapping.get(k.lower()) or _resolve_column(graph[src], k) for k in dst_keys]
            if not all(parts):
                logger.debug(
                    "kumo: %s cannot reference all of %s's identity %s; skipped",
                    src,
                    dst,
                    dst_keys,
                )
                continue
            fkey = tuple(parts)
        else:
            fkey = next(iter(mapping.values()))
        if (src, fkey, dst) in existing or (isinstance(fkey, tuple) and (src, dst) in existing_pairs):
            covered += 1
            continue
        try:
            graph.link(src, fkey, dst)
        except Exception:
            logger.debug(
                "kumo: skipped join-path link %s.%s -> %s",
                src,
                fkey,
                dst,
                exc_info=True,
            )
            continue
        covered += 1
    return covered


def _deduplicate_inferred_links(graph: Any) -> int:
    """Keep one deterministic FK when inference links the same table pair twice.

    KumoRFM cannot disambiguate an aggregation when a child table has multiple
    foreign keys to the same parent (for example ``job_id`` and
    ``restart_of_job_id``). Prefer the foreign key whose name exactly matches
    the destination primary key, then the shortest/lexicographically first
    name. Catalog-derived join paths do not use this fallback because they
    already scope the graph to the relationship relevant to the question.
    """
    grouped: dict[tuple[str, str], list[Any]] = {}
    for edge in graph.edges:
        grouped.setdefault((edge.src_table, edge.dst_table), []).append(edge)

    removed = 0
    for (_src, dst), edges in grouped.items():
        if len(edges) < 2:
            continue
        # Only a single-column key gives a name worth matching against; a composite
        # one reads back as the surrogate column on both ends, so such edges fall
        # through to the shortest/lexicographically-first tie-break.
        dst_keys = key_columns(graph[dst])
        primary_key = dst_keys[0].lower() if len(dst_keys) == 1 else ""
        keep = min(
            edges,
            key=lambda edge: (
                _fkey_name(edge.fkey).lower() != primary_key,
                len(_fkey_name(edge.fkey)),
                _fkey_name(edge.fkey).lower(),
            ),
        )
        for edge in edges:
            if edge == keep:
                continue
            graph.unlink(edge.src_table, edge.fkey, edge.dst_table)
            removed += 1
            logger.info(
                "kumo: removed ambiguous inferred link %s.%s -> %s (keeping %s.%s -> %s)",
                edge.src_table,
                edge.fkey,
                edge.dst_table,
                keep.src_table,
                keep.fkey,
                keep.dst_table,
            )
    return removed


def _entity_ids(graph: Any, frames: dict[str, pd.DataFrame]) -> dict[str, list[Any]]:
    """Identity of every loaded row, per table, for scoping a prediction.

    A composite identity is one tuple per row: naming a single one of its columns
    picks out no row. The surrogate column KumoRFM adds for such a key is absent from
    the frame, so the real key columns are read instead.
    """
    entity_ids: dict[str, list[Any]] = {}
    for name, table in graph.tables.items():
        keys = key_columns(table)
        frame = frames.get(name)
        if not keys or frame is None or any(k not in frame.columns for k in keys):
            continue
        rows = frame[keys].dropna().drop_duplicates()
        entity_ids[name.casefold()] = (
            rows[keys[0]].tolist() if len(keys) == 1 else [tuple(r) for r in rows.to_numpy().tolist()]
        )
    return entity_ids


def _resolve_prediction_scope(
    connector: Any,
    contract: GraphContract,
    entity_ids: dict[str, list[Any]],
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Load and validate a graph-owned population without exposing its IDs."""

    scope = contract.prediction_scope
    if scope is None:
        return (), None
    result_column = "__gsf_prediction_scope_entity"
    frame = connector.execute(
        f'SELECT "{scope.population_column}" AS "{result_column}" '
        f"FROM {_quote(contract.schema_name, scope.population_view)} "
        f'ORDER BY "{scope.population_column}"'
    )
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != [result_column]:
        raise ValueError("Prediction-scope population view returned an invalid column contract.")
    if len(frame) != scope.population_rows:
        raise ValueError(f"Prediction-scope population expected {scope.population_rows} rows but loaded {len(frame)}.")
    if frame.iloc[:, 0].isna().any():
        raise ValueError("Prediction-scope population contains a null entity identifier.")
    values = frame.iloc[:, 0].tolist()
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError("Prediction-scope population contains duplicate entity identifiers.")
    available = entity_ids.get(scope.entity_table.casefold())
    if available is None:
        raise ValueError("Prediction-scope entity table has no loaded graph identity.")
    available_set = set(available)
    if any(value not in available_set for value in values):
        raise ValueError("Prediction-scope population contains an entity outside the loaded graph.")
    receipt = {
        "anchor_time": scope.anchor_time,
        "anchor_source": "graph_contract",
        "entity_table": scope.entity_table,
        "entity_column": scope.entity_column,
        "population_view": scope.population_view,
        "population_column": scope.population_column,
        "population_count": len(values),
    }
    return tuple(values), receipt


_MAX_CATEGORICAL_VALUES = 12
_MAX_CATEGORICAL_VALUE_CHARS = 80


def _categorical_value_reference(
    frames: dict[str, pd.DataFrame],
    col_stypes: dict[str, dict[str, str]],
) -> str:
    """Render exact low-cardinality literals for the text-to-PQL prompt.

    The graph DDL tells the model that a column is categorical, but not how its
    values are cased.  That omission can produce SQL such as ``status = 'open'``
    against a case-sensitive warehouse value ``Open``.  Only categorical
    columns whose complete distinct set fits within a small bound are included;
    IDs, free text, numeric measures, and partial high-cardinality samples never
    enter the prompt.
    """

    lines: list[str] = []
    for table_name, frame in frames.items():
        table_stypes = col_stypes.get(table_name.casefold(), {})
        frame_columns = {str(column).casefold(): column for column in frame.columns}
        for column_name, stype in table_stypes.items():
            if str(stype).casefold() != "categorical":
                continue
            frame_column = frame_columns.get(column_name.casefold())
            if frame_column is None:
                continue

            values: list[Any] = []
            too_many = False
            for raw_value in frame[frame_column].dropna().drop_duplicates().tolist():
                value = _json_safe(raw_value)
                if not isinstance(value, (str, int, float, bool)):
                    continue
                if isinstance(value, str) and len(value) > _MAX_CATEGORICAL_VALUE_CHARS:
                    continue
                values.append(value)
                if len(values) > _MAX_CATEGORICAL_VALUES:
                    too_many = True
                    break
            if not values or too_many:
                continue
            lines.append(
                f"- table={json.dumps(table_name)}, column={json.dumps(str(frame_column))}, "
                f"type=categorical, exact values={json.dumps(values, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def build_prediction_context(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]] | None = None,
    *,
    database_name: str | None = None,
    graph_contract: GraphContract | None = None,
    join_paths: list[dict[str, Any]] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> PredictionContext | dict[str, Any]:
    """Build the KumoRFM graph + model scoped to the relevant tables.

    ``relevant_tables`` (as produced by the candidate-preparation step) scopes the
    KumoRFM graph to the tables relevant to the question. ``join_paths``
    (``attribute_join_paths``) from the text-to-SQL state supplies the table
    relationships: its catalog-derived joins are used as the graph's edges, and
    KumoRFM's own heuristic ``infer_links`` is used only as a fallback when no usable
    join path is available. ``examples`` are verified ``{question, query}`` PQL
    few-shots carried into generation. Returns a :class:`PredictionContext` on
    success, or a graceful error response dict when there is nothing to build a graph
    from.
    """
    logger.info("kumo: build_prediction_context start")

    from kumorfm import rfm

    from gsf.retrieval.kumo.kumo_model import KumoModel
    from gsf.retrieval.kumo.kumo_model import build_graph_context

    if not connectors:
        return _error_response("No database connection is configured.")
    if graph_contract is not None:
        if database_name is None:
            database_name = graph_contract.database_name
        elif database_name.casefold() != graph_contract.database_name.casefold():
            raise ValueError("Graph contract database differs from prediction scope.")
        relevant_tables = _contract_view_sources(relevant_tables or [], graph_contract)
    selected_connector = _select_connector(connectors, database_name)
    selected_database = str(database_name or getattr(selected_connector, "database_name", "") or "")
    if graph_contract is not None:
        _validate_connector_contract_views(selected_connector, graph_contract)
    client = _ensure_init()

    logger.info(
        "kumo: loading sample rows for %d relevant table(s)...",
        len(relevant_tables or []),
    )
    _load_start = time.perf_counter()
    frames, name_map, catalog_keys = _load_relevant_frames(
        connectors,
        relevant_tables or [],
        database_name=selected_database or None,
        strict=graph_contract is not None,
    )
    if not frames:
        return _error_response("No relevant tables were available to build a prediction graph.")
    logger.info(
        "kumo: loaded %d frame(s) in %.2fs; building graph from %d table(s)",
        len(frames),
        time.perf_counter() - _load_start,
        len(frames),
    )

    _graph_start = time.perf_counter()
    # Passing an explicit empty edge list suppresses LocalGraph's automatic
    # relationship inference. This lets catalog join paths take precedence and
    # avoids inferring the same links twice.
    graph = rfm.Graph.from_data(
        frames,
        edges=[],
        infer_metadata=True,
        verbose=False,
    )
    logger.info(
        "kumo: LocalGraph.from_data (metadata inferred) in %.2fs",
        time.perf_counter() - _graph_start,
    )
    if graph_contract is not None:
        _apply_graph_contract(graph, graph_contract)
        logger.info(
            "kumo: applied explicit graph contract %s with %d edge(s)",
            graph_contract.revision,
            len(graph_contract.edges),
        )
    else:
        # Before any linking: an edge is oriented towards a primary key, so a table whose
        # key inference missed can take part in no relationship at all.
        _declare_primary_keys(graph, catalog_keys)
        covered = _apply_join_paths(graph, join_paths)
        if covered:
            logger.info("kumo: using %d catalog join edge(s)", covered)
        else:
            # No usable catalog join paths — fall back to KumoRFM's link heuristics.
            logger.info("kumo: no catalog join paths; inferring links heuristically")
            try:
                graph.infer_links()
                _deduplicate_inferred_links(graph)
            except Exception:
                logger.exception("kumo: infer_links failed; proceeding without inferred links")

    graph_ddl, edges, col_stypes, time_columns = build_graph_context(graph)
    kumo_model = KumoModel(client.kumorfm(graph), graph)
    entity_ids = _entity_ids(graph, frames)
    prediction_scope_ids, prediction_scope_receipt = (
        _resolve_prediction_scope(selected_connector, graph_contract, entity_ids)
        if graph_contract is not None
        else ((), None)
    )
    column_reference = _categorical_value_reference(frames, col_stypes)
    graph_receipt = _build_graph_receipt(
        database_name=selected_database,
        graph=graph,
        frames=frames,
        examples=examples or [],
        contract=graph_contract,
        prediction_scope=prediction_scope_receipt,
    )

    # Entity-selection SQL runs against the live GSF database connection (the
    # selected connector — the source of the catalog tables). ``table_names``
    # maps bare graph table names to their schema-qualified form so the SQL resolves.
    return PredictionContext(
        kumo_model=kumo_model,
        connector=selected_connector,
        graph_ddl=graph_ddl,
        graph_edges=edges,
        graph_col_stypes=col_stypes,
        column_reference=column_reference,
        time_columns=time_columns,
        table_names=name_map,
        entity_ids=entity_ids,
        examples=examples or [],
        database_name=selected_database,
        graph_receipt=graph_receipt,
        prediction_scope=graph_contract.prediction_scope if graph_contract is not None else None,
        prediction_scope_ids=prediction_scope_ids,
    )


def run_prediction(question: str, llm: Any, context: PredictionContext) -> dict[str, Any]:
    """Generate + repair the PQL, predict, and format — given a prepared context."""
    from gsf.retrieval.kumo.pql_gen import generate_pql

    result = generate_pql(
        question,
        llm=llm,
        kumo_model=context.kumo_model,
        connector=context.connector,
        graph_ddl=context.graph_ddl,
        graph_edges=context.graph_edges,
        graph_col_stypes=context.graph_col_stypes,
        column_reference=context.column_reference,
        time_columns=context.time_columns,
        table_names=context.table_names,
        available_entity_ids=context.entity_ids,
        prediction_scope=context.prediction_scope,
        prediction_scope_ids=context.prediction_scope_ids,
        max_entities=_MAX_ENTITIES,
        max_preview_rows=_MAX_PREVIEW_ROWS,
        examples=context.examples,
    )

    final = _format_result(result)
    final["graph_receipt"] = context.graph_receipt
    return final
