"""add device_tokens for push notifications

Revision ID: c7e2d5f3a18b
Revises: f3d1b6c09a12
Create Date: 2026-04-22 18:00:00.000000

FCM/APNs 푸시 알림 대상 디바이스 토큰 저장 테이블.
사용자 1명이 여러 기기 등록 가능 (폰/태블릿/웹).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c7e2d5f3a18b"
down_revision: Union[str, None] = "f3d1b6c09a12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "device_tokens",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("platform", sa.String(length=10), nullable=False),
        sa.Column("token", sa.String(length=512), nullable=False),
        sa.Column("device_label", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "token", name="uq_device_tokens_user_token"),
    )
    op.create_index("idx_device_tokens_user_id", "device_tokens", ["user_id"])
    op.create_index("idx_device_tokens_active", "device_tokens", ["user_id", "is_active"])


def downgrade() -> None:
    op.drop_index("idx_device_tokens_active", table_name="device_tokens")
    op.drop_index("idx_device_tokens_user_id", table_name="device_tokens")
    op.drop_table("device_tokens")
