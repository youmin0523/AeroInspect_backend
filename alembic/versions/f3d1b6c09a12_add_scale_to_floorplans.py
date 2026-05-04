"""add scale calibration to floorplans

Revision ID: f3d1b6c09a12
Revises: e4c9a8b27f10
Create Date: 2026-04-22 17:00:00.000000

FR-015 평면도 스케일 보정용 컬럼 추가.
사용자가 평면도 위 두 점 + 실측 거리를 지정하면 px_per_meter 환산.
이후 벽체 길이·면적을 m 단위로 표기.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f3d1b6c09a12"
down_revision: Union[str, None] = "e4c9a8b27f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "floorplans",
        sa.Column(
            "scale_px_per_meter",
            sa.Float(),
            nullable=True,
            comment="1m 당 픽셀 수 (환산 계수)",
        ),
    )
    op.add_column(
        "floorplans",
        sa.Column(
            "scale_reference",
            sa.dialects.postgresql.JSONB(),
            nullable=True,
            comment="사용자 지정 기준 {p1:[x,y], p2:[x,y], real_length_m:float}",
        ),
    )


def downgrade() -> None:
    op.drop_column("floorplans", "scale_reference")
    op.drop_column("floorplans", "scale_px_per_meter")
