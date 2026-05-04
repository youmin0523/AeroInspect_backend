"""merge 0003 and inspection_schedules

Revision ID: 89b53c16de85
Revises: 0003, i2c3d4e5f6a7
Create Date: 2026-05-03 22:30:15.335512

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89b53c16de85'
down_revision: Union[str, None] = ('0003', 'i2c3d4e5f6a7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
