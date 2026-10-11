"""news articles: original timestamp and zone; event entities: link evidence

Additive only: three nullable columns. No existing row is modified.

Revision ID: 3c1f2a9d7e44
Revises: 806b95b85360
Create Date: 2026-10-11 15:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


revision: str = '3c1f2a9d7e44'
down_revision: Union[str, Sequence[str], None] = '806b95b85360'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('news_articles') as batch:
        batch.add_column(sa.Column('published_raw', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch.add_column(sa.Column('published_tz', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    with op.batch_alter_table('event_entities') as batch:
        batch.add_column(sa.Column('evidence', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('event_entities') as batch:
        batch.drop_column('evidence')
    with op.batch_alter_table('news_articles') as batch:
        batch.drop_column('published_tz')
        batch.drop_column('published_raw')
