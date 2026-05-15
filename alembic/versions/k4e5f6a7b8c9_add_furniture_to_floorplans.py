"""add furniture columns to floorplans

평면도 → 3D 모델링 라인에 가구/빌트인 검출 결과 추가.
자율비행 시 드론 충돌 회피 + LiDAR raycast 대상으로 사용.

Revision ID: k4e5f6a7b8c9
Revises: j3d4e5f6a7b8
Create Date: 2026-05-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "k4e5f6a7b8c9"
down_revision: Union[str, None] = "j3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "floorplans",
        sa.Column(
            "furniture_count",
            sa.Integer(),
            nullable=True,
            server_default="0",
            comment="검출된 가구/빌트인 수",
        ),
    )
    op.add_column(
        "floorplans",
        sa.Column(
            "furniture_data",
            JSONB(),
            nullable=True,
            comment="가구 회전 사각형 [{cx,cy,w,h,angle,label}, ...] (0-1 정규화) — 자율비행 충돌 회피용",
        ),
    )


def downgrade() -> None:
    op.drop_column("floorplans", "furniture_data")
    op.drop_column("floorplans", "furniture_count")
