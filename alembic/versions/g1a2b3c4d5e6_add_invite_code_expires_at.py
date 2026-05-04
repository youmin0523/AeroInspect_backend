"""add invite_code_expires_at to organizations

Revision ID: g1a2b3c4d5e6
Revises: f3d1b6c09a12
Create Date: 2026-04-27 12:00:00.000000

초대코드 만료 기능: 보안을 위해 30일 주기로 코드 자동 만료.
퇴사자가 기존 코드를 알고 있어도 만료 후 사용 불가.
"""
from datetime import datetime, timedelta, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g1a2b3c4d5e6"
down_revision: Union[str, None] = "c7e2d5f3a18b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) nullable로 컬럼 추가
    op.add_column(
        "organizations",
        sa.Column(
            "invite_code_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="초대 코드 만료 시각 (null=무제한, 기본 30일)",
        ),
    )

    # 2) 기존 조직에 기본 만료일 부여 (현재 + 30일)
    conn = op.get_bind()
    default_expires = datetime.now(timezone.utc) + timedelta(days=30)
    conn.execute(
        sa.text(
            "UPDATE organizations SET invite_code_expires_at = :expires WHERE invite_code_expires_at IS NULL"
        ),
        {"expires": default_expires},
    )


def downgrade() -> None:
    op.drop_column("organizations", "invite_code_expires_at")
