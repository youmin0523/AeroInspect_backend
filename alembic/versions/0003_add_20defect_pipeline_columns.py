"""add 20-defect pipeline columns to defect_logs

Revision ID: 0003
Revises: 0002_defect_class_display
Create Date: 2026-04-21

defect_logs 테이블에 20종 하자 파이프라인 확장 컬럼 추가:
- deviation_degrees: 수직수평/직각도 편차 (도)
- deviation_mm_per_m: 편차 mm/m 환산
- delta_temperature: 단열 온도 편차 (°C)
- ensemble_boosted: PatchCore 앙상블 승격 여부
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002_defect_class_display"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "defect_logs",
        sa.Column("deviation_degrees", sa.Float(), nullable=True,
                  comment="수직수평/직각도 편차 (도)"),
    )
    op.add_column(
        "defect_logs",
        sa.Column("deviation_mm_per_m", sa.Float(), nullable=True,
                  comment="편차 mm/m 환산"),
    )
    op.add_column(
        "defect_logs",
        sa.Column("delta_temperature", sa.Float(), nullable=True,
                  comment="주변 대비 온도차 (°C)"),
    )
    op.add_column(
        "defect_logs",
        sa.Column("ensemble_boosted", sa.String(5), nullable=True,
                  comment="PatchCore 앙상블 승격 (true/false)"),
    )


def downgrade() -> None:
    op.drop_column("defect_logs", "ensemble_boosted")
    op.drop_column("defect_logs", "delta_temperature")
    op.drop_column("defect_logs", "deviation_mm_per_m")
    op.drop_column("defect_logs", "deviation_degrees")
