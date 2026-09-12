# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope verified PQL examples to a catalog database.

Existing rows remain nullable because releases before this migration stored
PQL examples globally. New writes always set the selected database; prediction
retrieval only accepts an exact database match.

Revision ID: 71ef2a69f1d4
Revises: e37f4a1f2f5d
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "71ef2a69f1d4"
down_revision: Union[str, Sequence[str], None] = "e37f4a1f2f5d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pql_analysis", sa.Column("database_name", sa.Text(), nullable=True)
    )
    op.create_index(
        op.f("ix_pql_analysis_database_name"),
        "pql_analysis",
        ["database_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_pql_analysis_database_name"), table_name="pql_analysis"
    )
    op.drop_column("pql_analysis", "database_name")
