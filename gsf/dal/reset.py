# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-database reset: drop a database's Neo4j subgraph and pgvector embeddings.

This is the single source of truth for wiping one database's ingested data. It
removes the Neo4j subgraph reachable from the ``Database`` node (catalog and
semantic nodes alike) and the corresponding rows in both pgvector collections.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from dataclasses import dataclass

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from gsf.semantic.constants import LABEL_ANALYSIS
from gsf.semantic.constants import LABEL_COLUMN_ATTRIBUTE
from gsf.semantic.constants import LABEL_PQL_ANALYSIS
from gsf.semantic.constants import LABEL_SQL_ATTRIBUTE
from gsf.semantic.constants import LABEL_TERM
from gsf.semantic.constants import LABEL_TEXT_ATTRIBUTE
from gsf.vdb import get_data_vdb
from gsf.vdb import get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass
class ResetResult:
    """Summary of what a :func:`delete_all_data` call removed."""

    database_name: str
    data_rows: int
    semantic_rows: int


@dataclass
class RetiredDatabaseResult:
    """Summary of retiring one exact, already-migrated database alias."""

    database_name: str
    successor_database_name: str
    catalog_nodes: int
    data_rows: int
    semantic_rows: int


_SEMANTIC_LABELS = (
    LABEL_TERM,
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TEXT_ATTRIBUTE,
    LABEL_ANALYSIS,
    LABEL_PQL_ANALYSIS,
    Labels.CUSTOM_ANALYSIS,
)


def _delete_nodes_in_batches(node_match: str, database_name: str | None) -> int:
    """``DETACH DELETE`` every node returned as ``n`` by *node_match*.

    *node_match* is the driving statement of ``apoc.periodic.iterate``, which
    batches the deletion so very large graphs do not exhaust transaction
    memory. It may reference ``$database_name``, which is forwarded to both
    statements and simply left unused by the unscoped variants.

    Returns the number of nodes deleted.
    """
    rows = get_neo4j_conn().query_write(
        f"""
        CALL apoc.periodic.iterate(
            "{node_match}",
            "DETACH DELETE n",
            {{batchSize: 1000, params: {{database_name: $database_name}}}}
        )
        YIELD total
        RETURN total
        """,
        {"database_name": database_name},
    )
    if not rows:
        return 0
    return int(rows[0].get("total") or 0)


def _delete_database_nodes(database_name: str | None = None) -> int:
    """Delete a database's every node, catalog and semantic alike.

    Traverses out from the ``Database`` node following any relationship type,
    so everything reachable from it goes. When ``database_name`` is ``None``,
    every ``Database`` node and its graph is removed. Returns the number of
    nodes deleted.
    """
    # Pin the traversal to one Database node, or start from every one of them.
    database_filter = "" if database_name is None else " {name: $database_name}"
    deleted = _delete_nodes_in_batches(
        f"""
        MATCH (db:{Labels.DB}{database_filter})
        CALL apoc.path.subgraphNodes(db, {{}}) YIELD node AS n
        RETURN DISTINCT n
        """,
        database_name,
    )
    logger.info(
        "_delete_database_nodes: removed %d nodes for database %s",
        deleted,
        database_name or "<all>",
    )
    return deleted


def _delete_semantic_nodes(database_name: str | None = None) -> int:
    """Delete semantic Neo4j nodes, leaving data nodes intact.

    Deletes only nodes carrying one of :data:`_SEMANTIC_LABELS`; data nodes
    (``DB``/``Schema``/``Table``/``Column``) are untouched. Scoped to one
    database, semantic nodes are found by traversing out from its ``Database``
    node; for a full wipe they are matched on label alone, so nodes orphaned
    from every ``Database`` node are collected too — including ``PqlAnalysis``,
    which is never attached to a ``Database``. Returns the number of nodes
    deleted.
    """
    labels = "|".join(_SEMANTIC_LABELS)
    if database_name is None:
        node_match = f"MATCH (n:{labels}) RETURN n"
    else:
        label_predicate = " OR ".join(f"n:{label}" for label in _SEMANTIC_LABELS)
        node_match = f"""
        MATCH (db:{Labels.DB} {{name: $database_name}})
        CALL apoc.path.subgraphNodes(db, {{}}) YIELD node AS n
        WHERE {label_predicate}
        RETURN DISTINCT n
        """
    deleted = _delete_nodes_in_batches(node_match, database_name)
    logger.info(
        "_delete_semantic_nodes: removed %d semantic nodes for database %s",
        deleted,
        database_name or "<all>",
    )
    return deleted


def delete_semantic_layer(database_name: str | None = None) -> int:
    """Delete semantic Neo4j nodes and pgvector embeddings.

    When ``database_name`` is given, removes only that database's semantic nodes
    (see :data:`_SEMANTIC_LABELS`) reachable from the ``Database`` node and its
    ``semantic_layer`` rows. When ``None``, removes semantic nodes and
    embeddings across every database. Data nodes are left intact.

    Custom analyses, SQL and predictive alike, are part of what goes: they are
    user-authored, so nothing recompiles them afterwards. Returns the number of
    pgvector rows deleted.
    """
    _delete_semantic_nodes(database_name)

    semantic_vdb = get_semantic_vdb()
    if database_name is None:
        semantic_deleted = semantic_vdb.delete_all()
    else:
        semantic_deleted = len(semantic_vdb.delete_by_database(database_name))

    logger.info(
        "delete_semantic_layer: removed %d semantic pgvector rows for database %s",
        semantic_deleted,
        database_name or "<all>",
    )
    return semantic_deleted


def delete_data_layer(database_name: str | None = None) -> int:
    """Delete a database's Neo4j nodes and pgvector embeddings.

    When ``database_name`` is given, removes that ``Database`` node and
    everything reachable from it in Neo4j plus its ``data_objects_layer`` rows.
    When ``None``, removes every ``Database`` node's graph and all data
    embeddings. Returns the number of pgvector rows deleted.
    """
    _delete_database_nodes(database_name)

    data_vdb = get_data_vdb()
    if database_name is None:
        data_deleted = data_vdb.delete_all()
    else:
        data_deleted = len(data_vdb.delete_by_database(database_name))

    logger.info(
        "delete_data_layer: removed %d data pgvector rows for database %s",
        data_deleted,
        database_name or "<all>",
    )
    return data_deleted


def delete_all_data(database_name: str | None = None) -> ResetResult:
    """Delete every trace of a database, both layers included.

    Removes the ``Database`` node and everything reachable from it in Neo4j,
    then deletes every ``data_objects_layer`` and ``semantic_layer`` row tagged
    with ``database_name``. Other databases are untouched.

    Semantic nodes are removed first (while still reachable from the
    ``Database`` node) before :func:`delete_data_layer` removes that node.
    """
    semantic_rows = delete_semantic_layer(database_name)
    data_rows = delete_data_layer(database_name)

    result = ResetResult(
        database_name=database_name,
        data_rows=data_rows,
        semantic_rows=semantic_rows,
    )
    logger.info(
        "delete_all_data: removed %d pgvector rows for database %s (%d data, %d semantic)",
        result.data_rows + result.semantic_rows,
        database_name,
        result.data_rows,
        result.semantic_rows,
    )
    return result


def retire_database_alias(
    database_name: str,
    *,
    successor_database_name: str,
) -> RetiredDatabaseResult:
    """Remove one obsolete database alias after its schemas were reparented.

    This migration is intentionally narrower than :func:`delete_all_data`.
    A replace-model import first resolves the stable schema/table IDs and links
    them beneath *successor_database_name*.  This function then proves every
    schema beneath the obsolete database is also owned by that successor,
    deletes only the obsolete ``Database`` node, and removes vector rows tagged
    with the obsolete name.  It never traverses into or deletes the migrated
    catalog subgraph.
    """

    retired = database_name.strip()
    successor = successor_database_name.strip()
    if not retired or not successor:
        raise ValueError("Retired and successor database names must be nonempty.")
    if retired.casefold() == successor.casefold():
        raise ValueError("Retired and successor database names must differ.")

    conn = get_neo4j_conn()
    existing = conn.query_read(
        query=f"MATCH (db:{Labels.DB} {{name: $database_name}}) RETURN db.id AS id",
        parameters={"database_name": retired},
    )
    if len(existing) > 1:
        raise RuntimeError(f"Found multiple catalog databases named {retired!r}.")

    if existing:
        unshared = conn.query_read(
            query=f"""
            MATCH (legacy:{Labels.DB} {{name: $database_name}})
                  -[:{Edges.CONTAINS}]->(schema:{Labels.SCHEMA})
            WHERE NOT EXISTS {{
                MATCH (successor:{Labels.DB} {{name: $successor_database_name}})
                      -[:{Edges.CONTAINS}]->(schema)
            }}
            RETURN coalesce(schema.imported_id, schema.id) AS id,
                   schema.name AS name
            """,
            parameters={
                "database_name": retired,
                "successor_database_name": successor,
            },
        )
        if unshared:
            paths = ", ".join(sorted(str(row.get("name") or row.get("id") or "<unknown>") for row in unshared))
            raise RuntimeError(f"Cannot retire {retired!r}: schema(s) have not migrated to {successor!r}: {paths}.")
        conn.query_write(
            query=f"MATCH (db:{Labels.DB} {{name: $database_name}}) DETACH DELETE db",
            parameters={"database_name": retired},
        )

    data_rows = len(get_data_vdb().delete_by_database(retired))
    semantic_rows = len(get_semantic_vdb().delete_by_database(retired))
    result = RetiredDatabaseResult(
        database_name=retired,
        successor_database_name=successor,
        catalog_nodes=1 if existing else 0,
        data_rows=data_rows,
        semantic_rows=semantic_rows,
    )
    logger.info(
        "retire_database_alias: removed database alias %s after migration to %s "
        "(%d data embeddings, %d semantic embeddings)",
        retired,
        successor,
        data_rows,
        semantic_rows,
    )
    return result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run narrowly scoped GSF catalog maintenance.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    retire = subparsers.add_parser(
        "retire-database",
        help="Remove an obsolete database alias after a replace-model import.",
    )
    retire.add_argument("--database-name", required=True)
    retire.add_argument("--successor-database-name", required=True)
    args = parser.parse_args(argv)
    if args.command == "retire-database":
        result = retire_database_alias(
            args.database_name,
            successor_database_name=args.successor_database_name,
        )
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(_main())
