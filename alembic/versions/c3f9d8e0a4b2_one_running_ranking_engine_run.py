"""at most one RUNNING ranking engine run (partial unique index)

Revision ID: c3f9d8e0a4b2
Revises: b7e4c2a91d35
Create Date: 2026-10-05 10:45:00.000000

Before the index can exist, any duplicate RUNNING ranking runs left behind
by the old check-then-insert race must be resolved: every RUNNING RANKING
row except the most recently started one is marked FAILED with an
explanatory error (the same terminal state the engine already uses for
abandoned runs). Normally there is at most one such row and this touches
nothing. The index is partial, so it only ever covers RUNNING ranking rows.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f9d8e0a4b2'
down_revision: Union[str, Sequence[str], None] = 'b7e4c2a91d35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = 'uq_engine_runs_one_running_ranking'
WHERE = "status = 'RUNNING' AND kind = 'RANKING'"


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    runs = sa.table('engine_runs', sa.column('id', sa.Integer), sa.column('kind', sa.String),
                    sa.column('status', sa.String), sa.column('started_at', sa.DateTime),
                    sa.column('finished_at', sa.DateTime), sa.column('errors', sa.JSON))
    running = bind.execute(
        sa.select(runs.c.id, runs.c.errors)
        .where(runs.c.status == 'RUNNING', runs.c.kind == 'RANKING')
        .order_by(runs.c.started_at.desc(), runs.c.id.desc())
    ).all()
    for row in running[1:]:
        errors = list(row.errors or []) + [{
            "symbol": None, "stage": "run",
            "error": "superseded: concurrent RUNNING ranking run (resolved by migration c3f9d8e0a4b2)"}]
        bind.execute(runs.update().where(runs.c.id == row.id)
                     .values(status='FAILED', finished_at=sa.func.now(), errors=errors))

    op.create_index(INDEX, 'engine_runs', ['kind'], unique=True,
                    postgresql_where=sa.text(WHERE), sqlite_where=sa.text(WHERE))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX, table_name='engine_runs')
