# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, deployment-owned graph contracts for KumoRFM prediction.

Kumo can infer table metadata, but production prediction graphs must not depend on
heuristics for identity, time, or relationships.  A graph-contract file pins those
decisions per GSF database.  The file path is supplied through
``KUMO_GRAPH_CONTRACTS_FILE``; when it names the selected database the
prediction path uses exactly its governed schema views and edges.  A deployment
can provide either one contract or a strict bundle of contracts for multiple
databases.

The parser is intentionally small and fail-closed.  Unknown keys, ambiguous table
names, incomplete edges, and relationships that do not target the declared primary
key are rejected before any warehouse rows are read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GraphContractError(ValueError):
    """The configured graph-contract document is invalid or cannot be read."""


@dataclass(frozen=True)
class GraphContractTable:
    """One exact table definition in a prediction graph."""

    name: str
    schema_name: str
    primary_key: tuple[str, ...]
    time_column: str | None
    rows: int


@dataclass(frozen=True)
class GraphContractEdge:
    """One directed foreign-key relationship: source columns to target identity."""

    source_table: str
    source_columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class GraphContract:
    """The validated contract for one catalog database."""

    database_name: str
    tables: tuple[GraphContractTable, ...]
    edges: tuple[GraphContractEdge, ...]
    revision: str

    @property
    def schema_name(self) -> str:
        """The single governed schema shared by every contracted view."""

        return self.tables[0].schema_name


def _expect_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphContractError(f"{where} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise GraphContractError(f"{where} has invalid fields: {', '.join(details)}")


def _name(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise GraphContractError(f"{where} must be a string")
    value = value.strip()
    if not value and allow_empty:
        return ""
    if not value or not _SAFE_NAME.fullmatch(value):
        raise GraphContractError(f"{where} is not a safe nonempty name")
    return value


def _names(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise GraphContractError(f"{where} must be a nonempty list")
    names = tuple(_name(item, f"{where}[{index}]") for index, item in enumerate(value))
    if len({item.casefold() for item in names}) != len(names):
        raise GraphContractError(f"{where} contains duplicate names")
    return names


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphContractError(f"{where} must be a nonnegative integer")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GraphContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _parse_contract(root: dict[str, Any]) -> GraphContract:
    _exact_keys(
        root,
        {
            "schema_version",
            "database_name",
            "database_sha256",
            "source",
            "object_count",
            "row_count",
            "tables",
            "relationships",
            "time_columns",
            "forbidden_tables",
        },
        "graph contract",
    )
    if root["schema_version"] != 1:
        raise GraphContractError("graph contract schema_version must be 1")
    database_name = _name(root["database_name"], "graph contract.database_name")
    _sha256(root["database_sha256"], "graph contract.database_sha256")
    source = _expect_object(root["source"], "graph contract.source")
    _exact_keys(
        source,
        {
            "dataset_id",
            "dataset_version",
            "revision",
            "manifest_sha256",
            "prediction_manifest_sha256",
        },
        "graph contract.source",
    )
    for key in ("dataset_id", "dataset_version", "revision"):
        _name(source[key], f"graph contract.source.{key}")
    _sha256(source["manifest_sha256"], "graph contract.source.manifest_sha256")
    _sha256(
        source["prediction_manifest_sha256"],
        "graph contract.source.prediction_manifest_sha256",
    )
    if not isinstance(root["tables"], list) or not root["tables"]:
        raise GraphContractError("graph contract.tables must be nonempty")
    if not isinstance(root["relationships"], list):
        raise GraphContractError("graph contract.relationships must be a list")
    if not isinstance(root["time_columns"], dict):
        raise GraphContractError("graph contract.time_columns must be an object")
    if not isinstance(root["forbidden_tables"], list):
        raise GraphContractError("graph contract.forbidden_tables must be a list")

    tables: list[GraphContractTable] = []
    for index, item in enumerate(root["tables"]):
        where = f"graph contract.tables[{index}]"
        table = _expect_object(item, where)
        _exact_keys(
            table,
            {"name", "schema_name", "rows", "primary_key", "time_column"},
            where,
        )
        time_column = table["time_column"]
        if time_column is not None:
            time_column = _name(time_column, f"{where}.time_column")
        tables.append(
            GraphContractTable(
                schema_name=_name(table["schema_name"], f"{where}.schema_name"),
                name=_name(table["name"], f"{where}.name"),
                primary_key=_names(table["primary_key"], f"{where}.primary_key"),
                time_column=time_column,
                rows=_nonnegative_int(table["rows"], f"{where}.rows"),
            )
        )

    table_by_name = {table.name.casefold(): table for table in tables}
    if len(table_by_name) != len(tables):
        raise GraphContractError("graph contract.tables must have unique table names")
    schema_names = {table.schema_name.casefold() for table in tables}
    if len(schema_names) != 1:
        raise GraphContractError("graph contract.tables must all belong to one governed schema")

    object_count = _nonnegative_int(root["object_count"], "graph contract.object_count")
    row_count = _nonnegative_int(root["row_count"], "graph contract.row_count")
    if object_count != len(tables):
        raise GraphContractError("graph contract.object_count does not match tables")
    if row_count != sum(table.rows for table in tables):
        raise GraphContractError("graph contract.row_count does not match table rows")

    time_columns: dict[str, str | None] = {}
    for raw_table, raw_column in root["time_columns"].items():
        table_name = _name(raw_table, "graph contract.time_columns key")
        if raw_column is not None:
            raw_column = _name(raw_column, f"graph contract.time_columns.{table_name}")
        time_columns[table_name.casefold()] = raw_column
    if set(time_columns) != set(table_by_name):
        raise GraphContractError("graph contract.time_columns must name every table exactly")
    for table in tables:
        if time_columns[table.name.casefold()] != table.time_column:
            raise GraphContractError(f"graph contract.time_columns disagrees with table {table.name!r}")

    forbidden = tuple(
        _name(item, f"graph contract.forbidden_tables[{index}]") for index, item in enumerate(root["forbidden_tables"])
    )
    if len({item.casefold() for item in forbidden}) != len(forbidden):
        raise GraphContractError("graph contract.forbidden_tables contains duplicates")
    overlap = set(table_by_name) & {item.casefold() for item in forbidden}
    if overlap:
        raise GraphContractError("graph contract includes a forbidden table")

    edges: list[GraphContractEdge] = []
    seen_edges: set[tuple[Any, ...]] = set()
    for index, item in enumerate(root["relationships"]):
        where = f"graph contract.relationships[{index}]"
        edge = _expect_object(item, where)
        _exact_keys(
            edge,
            {"source_table", "source_column", "target_table", "target_column"},
            where,
        )
        source_table = _name(edge["source_table"], f"{where}.source_table")
        target_table = _name(edge["target_table"], f"{where}.target_table")
        source_columns = (_name(edge["source_column"], f"{where}.source_column"),)
        target_columns = (_name(edge["target_column"], f"{where}.target_column"),)
        if source_table.casefold() not in table_by_name:
            raise GraphContractError(f"{where}.source_table is not declared")
        target = table_by_name.get(target_table.casefold())
        if target is None:
            raise GraphContractError(f"{where}.target_table is not declared")
        if tuple(column.casefold() for column in target_columns) != tuple(
            column.casefold() for column in target.primary_key
        ):
            raise GraphContractError(f"{where}.target_columns must exactly match the target primary key")
        signature = (
            source_table.casefold(),
            tuple(column.casefold() for column in source_columns),
            target_table.casefold(),
            tuple(column.casefold() for column in target_columns),
        )
        if signature in seen_edges:
            raise GraphContractError(f"{where} duplicates another edge")
        seen_edges.add(signature)
        edges.append(
            GraphContractEdge(
                source_table=source_table,
                source_columns=source_columns,
                target_table=target_table,
                target_columns=target_columns,
            )
        )

    canonical = {
        "database_name": database_name,
        "tables": [
            {
                "name": table.name,
                "schema_name": table.schema_name,
                "rows": table.rows,
                "primary_key": list(table.primary_key),
                "time_column": table.time_column,
            }
            for table in tables
        ],
        "edges": [
            {
                "source_table": edge.source_table,
                "source_column": edge.source_columns[0],
                "target_table": edge.target_table,
                "target_column": edge.target_columns[0],
            }
            for edge in edges
        ],
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return GraphContract(
        database_name=database_name,
        tables=tuple(tables),
        edges=tuple(edges),
        revision=f"sha256:{digest}",
    )


def _load_configured_contracts(
    *,
    path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[GraphContract, ...], bool]:
    """Load and validate all configured contracts plus the bundle marker."""

    selected = path if path is not None else os.environ.get("KUMO_GRAPH_CONTRACTS_FILE")
    if selected is None or not str(selected).strip():
        return (), False
    contract_path = Path(selected).expanduser()
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GraphContractError(
            f"Could not load KUMO graph contracts from {contract_path}: {type(exc).__name__}"
        ) from exc

    root = _expect_object(document, "graph contract document")
    if "contracts" in root:
        _exact_keys(root, {"schema_version", "contracts"}, "graph contract bundle")
        if root["schema_version"] != 1:
            raise GraphContractError("graph contract bundle schema_version must be 1")
        raw_contracts = root["contracts"]
        if not isinstance(raw_contracts, list) or not raw_contracts:
            raise GraphContractError("graph contract bundle.contracts must be a nonempty list")
        contracts: dict[str, GraphContract] = {}
        for index, raw_contract in enumerate(raw_contracts):
            contract = _parse_contract(
                _expect_object(
                    raw_contract,
                    f"graph contract bundle.contracts[{index}]",
                )
            )
            key = contract.database_name.casefold()
            if key in contracts:
                raise GraphContractError("graph contract bundle contains duplicate database names")
            contracts[key] = contract
        return tuple(contracts.values()), True

    return (_parse_contract(root),), False


def load_graph_contracts(
    *,
    path: str | os.PathLike[str] | None = None,
) -> tuple[GraphContract, ...]:
    """Load every configured contract, validating the complete document."""

    contracts, _is_bundle = _load_configured_contracts(path=path)
    return contracts


def load_graph_contract(
    database_name: str,
    *,
    path: str | os.PathLike[str] | None = None,
) -> GraphContract | None:
    """Load the contract for *database_name*.

    An unset environment variable means contract mode is disabled. Once a file is
    configured, unreadable or invalid content raises :class:`GraphContractError`.
    A strict bundle must contain the selected database. A legacy single-contract
    document for another database leaves that database on the catalog-driven path.
    """

    selected_database = _name(database_name, "selected prediction database")
    contracts, is_bundle = _load_configured_contracts(path=path)
    if not contracts:
        return None
    selected = next(
        (contract for contract in contracts if contract.database_name.casefold() == selected_database.casefold()),
        None,
    )
    if selected is None and is_bundle:
        raise GraphContractError(f"graph contract bundle has no contract for database {selected_database!r}")
    return selected


__all__ = [
    "GraphContract",
    "GraphContractEdge",
    "GraphContractError",
    "GraphContractTable",
    "load_graph_contract",
    "load_graph_contracts",
]
