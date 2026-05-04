"""add is_superadmin to users and departments table

Revision ID: a957fb9970a3
Revises: d7a4d75a3d5f
Create Date: 2026-04-20 16:11:59.944205

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a957fb9970a3'
down_revision: Union[str, None] = 'd7a4d75a3d5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('is_superadmin', sa.Boolean(), server_default='false', nullable=False, comment='Platform superadmin flag'))
    # departments table may already exist from init_db auto-create
    conn = op.get_bind()
    result = conn.execute(sa.text("SELECT to_regclass('public.departments')"))
    if result.scalar() is None:
        op.create_table(
            'departments',
            sa.Column('id', sa.UUID(), nullable=False),
            sa.Column('organization_id', sa.UUID(), nullable=False),
            sa.Column('name', sa.String(length=100), nullable=False, comment='Department name'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('organization_id', 'name', name='uq_dept_org_name'),
        )
        op.create_index('idx_dept_org_id', 'departments', ['organization_id'])


def downgrade() -> None:
    op.drop_index('idx_dept_org_id', table_name='departments')
    op.drop_table('departments')
    op.drop_column('users', 'is_superadmin')
