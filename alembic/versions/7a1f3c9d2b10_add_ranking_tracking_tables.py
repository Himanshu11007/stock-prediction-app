"""add append-only ranking tracking tables (prospective validation of ranking v1.0)

Revision ID: 7a1f3c9d2b10
Revises: ce0fe2958679
Create Date: 2026-10-03 18:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '7a1f3c9d2b10'
down_revision: Union[str, Sequence[str], None] = 'ce0fe2958679'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('ranking_snapshots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('symbol', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('ranked_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
    sa.Column('engine_version', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('fqvf_version', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('stockai_score', sa.Float(), nullable=True),
    sa.Column('score_coverage', sa.Float(), nullable=True),
    sa.Column('eligible', sa.Boolean(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=True),
    sa.Column('fqvf_summary', sa.JSON(), nullable=True),
    sa.Column('component_scores', sa.JSON(), nullable=True),
    sa.Column('freshness', sa.JSON(), nullable=True),
    sa.Column('market_regime', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('reference_date', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('reference_price', sa.Float(), nullable=True),
    sa.Column('benchmark_symbol', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('benchmark_reference_price', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['engine_runs.run_id'], ),
    sa.ForeignKeyConstraint(['symbol'], ['companies.symbol'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'symbol', name='uq_ranking_snapshot_run_symbol')
    )
    op.create_index(op.f('ix_ranking_snapshots_run_id'), 'ranking_snapshots', ['run_id'], unique=False)
    op.create_index(op.f('ix_ranking_snapshots_symbol'), 'ranking_snapshots', ['symbol'], unique=False)
    op.create_index(op.f('ix_ranking_snapshots_ranked_at'), 'ranking_snapshots', ['ranked_at'], unique=False)
    op.create_table('ranking_outcomes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('snapshot_id', sa.Integer(), nullable=False),
    sa.Column('horizon', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('start_date', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('start_price', sa.Float(), nullable=False),
    sa.Column('outcome_date', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('outcome_price', sa.Float(), nullable=False),
    sa.Column('stock_return', sa.Float(), nullable=False),
    sa.Column('benchmark_return', sa.Float(), nullable=True),
    sa.Column('excess_return', sa.Float(), nullable=True),
    sa.Column('computed_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['snapshot_id'], ['ranking_snapshots.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('snapshot_id', 'horizon', name='uq_ranking_outcome_snapshot_horizon')
    )
    op.create_index(op.f('ix_ranking_outcomes_snapshot_id'), 'ranking_outcomes', ['snapshot_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_ranking_outcomes_snapshot_id'), table_name='ranking_outcomes')
    op.drop_table('ranking_outcomes')
    op.drop_index(op.f('ix_ranking_snapshots_ranked_at'), table_name='ranking_snapshots')
    op.drop_index(op.f('ix_ranking_snapshots_symbol'), table_name='ranking_snapshots')
    op.drop_index(op.f('ix_ranking_snapshots_run_id'), table_name='ranking_snapshots')
    op.drop_table('ranking_snapshots')
