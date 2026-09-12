# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-database reset: drop a database's catalog rows and pgvector embeddings.

The single source of truth for wiping one database's ingested data.

**This module is where the schema pays for itself.** Containment is a parent
foreign key, so deleting one database row removes the whole catalog tier by
``ON DELETE CASCADE`` — no traversal, no batching, and no way for a newly added
child table to be missed.

Two consequences worth knowing:

* **Deletes never cross a database.** Foreign keys point downward within one
  database, so resetting one cannot reach another's data — including through a
  Term the two share, which survives.
* **A scoped semantic reset reaches only owned ``PqlAnalysis`` rows.** Reviewed
  predictive examples carry their database name directly, so resetting one
  source cannot remove another source's examples.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from dataclasses import dataclass

from sqlalchemy import delete, select, union

from gsf.dal import schema as s
from gsf.dal.session import store, write_transaction
from gsf.vdb import get_data_vdb, get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass
class ResetResult:
    """Summary of what a :func:`delete_all_data` call removed."""

    #: ``None`` when the reset covered every database, matching
    #: ``delete_all_data``'s own optional argument.
    database_name: str | None
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


def _table_ids(database_name: str):
    """The tables of one database, as a subquery."""
    return (
        select(s.catalog_table.c.id)
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_database.c.name == database_name)
    )


def _delete_scoped_semantic(database_name: str) -> int:
    """Delete the semantic rows belonging to one database.

    "Belonging" is resolved per entity, because each reaches a database by a
    different path:

    * a **Term** through the tables that represent it;
    * a **ColumnAttribute** through its ``table_id``;
    * a **SqlAttribute** and a **CustomAnalysis** through the tables their SQL
      references.

    Each is deleted only when **every** table it touches belongs to this
    database — the same all-or-nothing shape the zone reads use, and for the
    same reason inverted: a Term shared with another database is that other
    database's data too, and a reset of this one must not take it.

    ``TextAttribute`` and ``Analysis`` are absent here on purpose: nothing
    connects them to a database, so there is no scope that would select them.
    """
    tables = _table_ids(database_name)
    deleted = 0

    def only_ours(link_table, owner_column, table_column):
        """Rows whose links all land inside this database, and at least one does."""
        inside = (
            select(owner_column)
            .where(table_column.in_(tables))
            .distinct()
            .scalar_subquery()
        )
        outside = (
            select(owner_column)
            .where(table_column.notin_(tables))
            .distinct()
            .scalar_subquery()
        )
        return owner_column.in_(inside) & owner_column.notin_(outside)

    # Term: reached through table__term.
    term_ids = select(s.table__term.c.term_id).where(
        only_ours(s.table__term, s.table__term.c.term_id, s.table__term.c.table_id)
    )
    deleted += len(
        store().query_write(
            delete(s.term).where(s.term.c.id.in_(term_ids)).returning(s.term.c.id)
        )
    )

    # ColumnAttribute: owned outright by exactly one table.
    deleted += len(
        store().query_write(
            delete(s.column_attribute)
            .where(s.column_attribute.c.table_id.in_(tables))
            .returning(s.column_attribute.c.id)
        )
    )

    # PQL few-shots carry their reviewed prediction database directly.
    deleted += len(
        store().query_write(
            delete(s.pql_analysis)
            .where(s.pql_analysis.c.database_name == database_name)
            .returning(s.pql_analysis.c.id)
        )
    )

    # SqlAttribute and CustomAnalysis: reached through the tables their SQL hits.
    for owner_table, link_table, owner_column in (
        (s.sql_attribute, s.sql_attribute__sql, s.sql_attribute__sql.c.attribute_id),
        (
            s.custom_analysis,
            s.custom_analysis__sql,
            s.custom_analysis__sql.c.analysis_id,
        ),
    ):
        touched = (
            select(owner_column, s.sql_query__table.c.table_id)
            .select_from(
                link_table.join(
                    s.sql_query__table,
                    s.sql_query__table.c.sql_query_id == link_table.c.sql_query_id,
                )
            )
            .subquery("touched")
        )
        inside = (
            select(touched.c[owner_column.name])
            .where(touched.c.table_id.in_(tables))
            .distinct()
        )
        outside = (
            select(touched.c[owner_column.name])
            .where(touched.c.table_id.notin_(tables))
            .distinct()
        )
        deleted += len(
            store().query_write(
                delete(owner_table)
                .where(
                    owner_table.c.id.in_(inside),
                    owner_table.c.id.notin_(outside),
                )
                .returning(owner_table.c.id)
            )
        )
    return deleted


def _delete_all_semantic() -> int:
    """Delete every semantic row, in every database.

    Unscoped, so ``PqlAnalysis``, ``TextAttribute`` and ``Analysis`` *are*
    included — nothing has to connect them to a database for this to reach
    them.
    """
    deleted = 0
    for table in (
        s.term,
        s.column_attribute,
        s.sql_attribute,
        s.text_attribute,
        s.analysis,
        s.pql_analysis,
        s.custom_analysis,
    ):
        deleted += len(store().query_write(delete(table).returning(table.c.id)))
    return deleted


def delete_semantic_layer(database_name: str | None = None) -> int:
    """Delete semantic rows and their pgvector embeddings. Catalog rows survive.

    Custom analyses, SQL and predictive alike, are part of what goes: they are
    user-authored, so nothing recompiles them afterwards.

    Returns the number of **pgvector rows** deleted, not database rows — the
    caller reports it as "embeddings removed". The row count is logged.
    """
    with write_transaction():
        _delete_semantic_rows(database_name)
    return _delete_semantic_vectors(database_name)


def _delete_semantic_rows(database_name: str | None) -> int:
    """The transactional half: semantic rows only, no embeddings."""
    if database_name is None:
        deleted = _delete_all_semantic()
    else:
        deleted = _delete_scoped_semantic(database_name)
    logger.info(
        "delete_semantic_layer: removed %d semantic rows for database %s",
        deleted,
        database_name or "<all>",
    )
    return deleted


def _delete_semantic_vectors(database_name: str | None) -> int:
    """The non-transactional half.

    pgvector lives in its own schema behind ``langchain_postgres``, on its own
    connection, so it cannot join the DAL's transaction. Run after the rows
    commit rather than before: a vector whose row is gone is dead weight the
    next embed pass overwrites, whereas a row whose vector is gone is a hit
    retrieval can no longer explain.
    """
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


def _delete_orphaned_statements(candidate_ids: list[str] | None = None) -> int:
    """Delete ``sql_query`` rows nothing references any more.

    "Orphan" is defined as having no link left in *any* of the four tables that
    point at a statement, not just the two the catalog cascade emptied. A
    statement can also be owned by a SqlAttribute or a CustomAnalysis, and those
    survive a data-layer reset — deleting on the catalog links alone would take
    the semantic tier's statements with it.

    *candidate_ids* narrows the sweep to statements the caller just unlinked.
    Without it this is a **global** delete of every unreferenced statement, and
    a statement is unreferenced for a moment during any ingest: ``add_query``
    inserts the ``sql_query`` row before its links, so a scoped reset running
    concurrently with an ingest of a *different* database would delete that
    in-flight row and the ingest would fail on a vanished foreign key. Scoped,
    the sweep can only touch rows the reset itself orphaned.

    ``None`` keeps the global behaviour, which is what a whole-store reset
    wants — there is no other database left to race with.
    """
    referenced = union(
        select(s.sql_query__table.c.sql_query_id),
        select(s.sql_query__column.c.sql_query_id),
        select(s.sql_attribute__sql.c.sql_query_id),
        select(s.custom_analysis__sql.c.sql_query_id),
    )
    statement = delete(s.sql_query).where(s.sql_query.c.id.not_in(referenced))
    if candidate_ids is not None:
        if not candidate_ids:
            return 0
        statement = statement.where(s.sql_query.c.id.in_(candidate_ids))
    removed = len(store().query_write(statement.returning(s.sql_query.c.id)))
    if removed:
        logger.info("delete_data_layer: removed %d orphaned statement rows", removed)
    return removed


def delete_data_layer(database_name: str | None = None) -> int:
    """Delete a database's catalog rows and its pgvector embeddings.

    Returns the number of pgvector rows deleted. The row deletes run as one
    transaction; the embeddings follow it, for the reason given on
    :func:`_delete_semantic_vectors`.
    """
    with write_transaction():
        _delete_data_rows(database_name)
    return _delete_data_vectors(database_name)


def _delete_data_rows(database_name: str | None = None) -> int:
    """The transactional half: catalog rows and the statements they orphaned.

    One ``DELETE`` on ``catalog_database``. Schemas, tables, columns, foreign
    keys, joins, statement links and zone targets all follow by cascade — which
    is the point of modelling ``CONTAINS`` as a parent FK rather than an edge
    table, and means a child table added later cannot be forgotten here.

    Statements are the exception and need the second delete below: ``sql_query``
    has no foreign key into the catalog, so the cascade takes its *links* and
    leaves the rows. Left behind they are invisible but not harmless — dedup
    matches on ``md5(sql_full_query)``, so the next ingest merges into the old
    row and ``total_counter`` accumulates across resets.

    Returns the number of pgvector rows deleted.
    """
    # The statements this database's tables reference, captured *before* the
    # cascade removes those links — afterwards there is no way back to them.
    candidates: list[str] | None = None
    if database_name is not None:
        candidates = [
            row["sql_query_id"]
            for row in store().query_read(
                select(s.sql_query__table.c.sql_query_id)
                .select_from(
                    s.sql_query__table.join(
                        s.catalog_table,
                        s.catalog_table.c.id == s.sql_query__table.c.table_id,
                    )
                    .join(
                        s.catalog_schema,
                        s.catalog_schema.c.id == s.catalog_table.c.schema_id,
                    )
                    .join(
                        s.catalog_database,
                        s.catalog_database.c.id == s.catalog_schema.c.database_id,
                    )
                )
                .where(s.catalog_database.c.name == database_name)
                .distinct()
            )
        ]

    statement = delete(s.catalog_database)
    if database_name is not None:
        statement = statement.where(s.catalog_database.c.name == database_name)
    deleted = len(store().query_write(statement.returning(s.catalog_database.c.id)))
    logger.info(
        "delete_data_layer: removed %d database rows for database %s",
        deleted,
        database_name or "<all>",
    )
    _delete_orphaned_statements(candidates)
    return deleted


def _delete_data_vectors(database_name: str | None) -> int:
    """The non-transactional half — see :func:`_delete_semantic_vectors`."""
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

    **Order matters**: the semantic rows are identified *through* the catalog — a Term by the tables that
    represent it, a SqlAttribute by the tables its SQL hits. Delete the catalog
    first and those links are already gone, so the semantic pass would find
    nothing to scope and leave every Term behind.
    """
    # Both layers in one transaction. Previously each delete autocommitted, so
    # a failure part-way through left the semantic layer gone and the catalog
    # intact -- and the semantic layer is user-authored, so nothing recompiles
    # it. That is exactly the half-state this function's ordering rule exists
    # to avoid, and the ordering alone could not prevent it.
    with write_transaction():
        _delete_semantic_rows(database_name)
        _delete_data_rows(database_name)

    # Outside the transaction on purpose: pgvector is reached over its own
    # connection and cannot be rolled back with the rows. Doing it after the
    # commit means a crash here leaves orphaned vectors, which the next embed
    # pass overwrites -- the tolerable direction of the two.
    semantic_rows = _delete_semantic_vectors(database_name)
    data_rows = _delete_data_vectors(database_name)

    result = ResetResult(
        database_name=database_name,
        data_rows=data_rows,
        semantic_rows=semantic_rows,
    )
    logger.info(
        "delete_all_data: removed %d pgvector rows for database %s "
        "(%d data, %d semantic)",
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
    """Remove an empty legacy database after replace-import reparenting.

    In the relational catalog a schema has exactly one database parent. A
    replace-model import first moves stable schema rows to the successor; this
    function then proves the successor exists and the retired database owns no
    remaining schemas before deleting only that alias and its vector rows.
    """
    retired = database_name.strip()
    successor = successor_database_name.strip()
    if not retired or not successor:
        raise ValueError("Retired and successor database names must be nonempty.")
    if retired.casefold() == successor.casefold():
        raise ValueError("Retired and successor database names must differ.")

    with write_transaction():
        retired_rows = store().query_read(
            select(s.catalog_database.c.id).where(s.catalog_database.c.name == retired)
        )
        successor_rows = store().query_read(
            select(s.catalog_database.c.id).where(
                s.catalog_database.c.name == successor
            )
        )
        if len(retired_rows) > 1:
            raise RuntimeError(f"Found multiple catalog databases named {retired!r}.")
        if retired_rows and len(successor_rows) != 1:
            raise RuntimeError(
                f"Cannot retire {retired!r}: successor {successor!r} does not exist."
            )

        if retired_rows:
            schemas = store().query_read(
                select(s.catalog_schema.c.id, s.catalog_schema.c.name).where(
                    s.catalog_schema.c.database_id == retired_rows[0]["id"]
                )
            )
            if schemas:
                names = ", ".join(
                    sorted(str(row.get("name") or row["id"]) for row in schemas)
                )
                raise RuntimeError(
                    f"Cannot retire {retired!r}: schema(s) have not migrated "
                    f"to {successor!r}: {names}."
                )
            store().query_write(
                delete(s.catalog_database).where(
                    s.catalog_database.c.id == retired_rows[0]["id"]
                )
            )

    data_rows = len(get_data_vdb().delete_by_database(retired))
    semantic_rows = len(get_semantic_vdb().delete_by_database(retired))
    result = RetiredDatabaseResult(
        database_name=retired,
        successor_database_name=successor,
        catalog_nodes=1 if retired_rows else 0,
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
    parser = argparse.ArgumentParser(
        description="Run narrowly scoped GSF catalog maintenance."
    )
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
