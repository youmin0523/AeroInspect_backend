"""add inspection_schedules table

Revision ID: i2c3d4e5f6a7
Revises: h1b2c3d4e5f6
Create Date: 2026-05-03

EmployeeLanding 의 "오늘 일정" 위젯이 프론트 const(MOCK_TODAY_SCHEDULE)에 박혀있던 것을
DB 기반으로 전환하기 위한 점검 일정 테이블 신설.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "i2c3d4e5f6a7"
down_revision: Union[str, None] = "h1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "inspection_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("site_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("operator_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False, comment="점검 예정 시각 (UTC)"),
        sa.Column(
            "status",
            sa.Enum("upcoming", "in_progress", "completed", "cancelled", name="schedule_status_enum"),
            nullable=False,
            server_default="upcoming",
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["operator_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_schedule_org_time", "inspection_schedules", ["organization_id", "scheduled_at"])
    op.create_index("idx_schedule_site_time", "inspection_schedules", ["site_id", "scheduled_at"])


def downgrade() -> None:
    op.drop_index("idx_schedule_site_time", table_name="inspection_schedules")
    op.drop_index("idx_schedule_org_time", table_name="inspection_schedules")
    op.drop_table("inspection_schedules")
    op.execute("DROP TYPE IF EXISTS schedule_status_enum")
