"""add password reset tokens, auth throttle events and users.token_valid_after

Revision ID: b7e4c2a91d35
Revises: b78f20585037
Create Date: 2026-10-05 10:30:00.000000

Additive only: two new tables and one nullable column, so it is safe on a
live database with existing users (no rewrite, no backfill, no lock beyond
the brief ALTER TABLE ... ADD COLUMN of a nullable column).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'b7e4c2a91d35'
down_revision: Union[str, Sequence[str], None] = 'b78f20585037'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('password_reset_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
    sa.Column('expires_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
    sa.Column('used_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True),
    sa.Column('requested_ip', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),
    sa.Column('user_agent', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
    op.create_index(op.f('ix_password_reset_tokens_token_hash'), 'password_reset_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_password_reset_tokens_created_at'), 'password_reset_tokens', ['created_at'], unique=False)

    op.create_table('auth_throttle_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('bucket', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('key_hash', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('created_at', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_auth_throttle_bucket_key_created', 'auth_throttle_events',
                    ['bucket', 'key_hash', 'created_at'], unique=False)
    op.create_index(op.f('ix_auth_throttle_events_created_at'), 'auth_throttle_events', ['created_at'], unique=False)

    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(sa.Column('token_valid_after', sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_column('token_valid_after')
    op.drop_index(op.f('ix_auth_throttle_events_created_at'), table_name='auth_throttle_events')
    op.drop_index('ix_auth_throttle_bucket_key_created', table_name='auth_throttle_events')
    op.drop_table('auth_throttle_events')
    op.drop_index(op.f('ix_password_reset_tokens_created_at'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_token_hash'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_user_id'), table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
