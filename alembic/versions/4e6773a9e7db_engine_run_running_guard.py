"""engine run guard: at most one RUNNING RANKING run (partial unique index)

Revision ID: 4e6773a9e7db
Revises: 66e61a7dffc5
Create Date: 2026-10-10 14:00:00

Adds a partial unique index so the DATABASE guarantees that only one RANKING
engine run is RUNNING at a time; two processes (web admin, scheduler,
external runner) can no longer both start one. SINGLE-stock runs are not
affected. Additive: no data is modified.

Idempotent: if the index already exists, nothing is done.

Pre-check: if more than one RANKING run is already RUNNING, the upgrade
stops with the list of those runs and changes nothing. Resolving them is a
reviewed, manual step (docs/DATA_HEALTH_AND_ENGINE_RUNS.md, "Overlapping
runs"); this migration never edits run records.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '4e6773a9e7db'
down_revision: Union[str, Sequence[str], None] = '66e61a7dffc5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = "uq_engine_runs_one_running_ranking"
WHERE = "status = 'RUNNING' AND kind = 'RANKING'"


class DuplicateRunningRunsError(RuntimeError):
    """More than one RANKING run is RUNNING; the index cannot be created."""


def running_ranking_runs(bind) -> list:
    return bind.execute(sa.text(
        f"SELECT run_id, started_at FROM engine_runs WHERE {WHERE} ORDER BY started_at")).fetchall()


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    if any(ix["name"] == INDEX for ix in sa.inspect(bind).get_indexes("engine_runs")):
        return          # already present (e.g. created by an equivalent migration on another branch)
    rows = running_ranking_runs(bind)
    if len(rows) > 1:
        listed = ", ".join(f"{r[0]} (started {r[1]})" for r in rows)
        raise DuplicateRunningRunsError(
            f"{len(rows)} RANKING runs are RUNNING: {listed}. No change was made. Resolve them first "
            f"(docs/DATA_HEALTH_AND_ENGINE_RUNS.md, 'Overlapping runs'), then run the upgrade again.")
    op.create_index(INDEX, "engine_runs", ["kind"], unique=True,
                    sqlite_where=sa.text(WHERE), postgresql_where=sa.text(WHERE))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX, table_name="engine_runs")
