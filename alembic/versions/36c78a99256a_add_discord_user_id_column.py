"""add discord_user_id column

Revision ID: 36c78a99256a
Revises: f766245dd987
Create Date: 2026-06-16 09:51:28.267949

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '36c78a99256a'
down_revision: Union[str, None] = 'f766245dd987'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('discord_user_id', sa.String(length=100), nullable=True))
    op.create_index(op.f('ix_users_discord_user_id'), 'users', ['discord_user_id'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_discord_user_id'), table_name='users')
    op.drop_column('users', 'discord_user_id')
