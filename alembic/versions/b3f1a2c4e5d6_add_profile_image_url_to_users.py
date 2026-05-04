"""add profile_image_url to users

Revision ID: b3f1a2c4e5d6
Revises: a957fb9970a3
Create Date: 2026-04-20 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3f1a2c4e5d6'
down_revision: Union[str, None] = 'a957fb9970a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'profile_image_url',
            sa.String(length=500),
            nullable=True,
            comment='프로필 이미지 경로 (uploads/profiles/UUID.ext)',
        ),
    )


def downgrade() -> None:
    op.drop_column('users', 'profile_image_url')
