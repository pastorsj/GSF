"""Scope predictive PQL examples to their reviewed database.

Revision ID: a64b3f19c2d7
Revises: e37f4a1f2f5d
Create Date: 2026-09-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "a64b3f19c2d7"
down_revision: Union[str, Sequence[str], None] = "e37f4a1f2f5d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the nullable owner used by scoped prediction retrieval."""
    op.add_column("pql_analysis", sa.Column("database_name", sa.Text(), nullable=True))
    op.create_index(
        op.f("ix_pql_analysis_database_name"),
        "pql_analysis",
        ["database_name"],
        unique=False,
    )


def downgrade() -> None:
    """Remove predictive-example database ownership."""
    op.drop_index(op.f("ix_pql_analysis_database_name"), table_name="pql_analysis")
    op.drop_column("pql_analysis", "database_name")
