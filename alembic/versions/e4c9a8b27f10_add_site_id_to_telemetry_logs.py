"""add site_id to telemetry_logs

Revision ID: e4c9a8b27f10
Revises: c8f1d2e4a7b9
Create Date: 2026-04-22 16:00:00.000000

/api/v1/coverage/{site_id} 엔드포인트 정확도 개선용.
기존엔 전역 최근 텔레메트리로 convex hull을 뽑아 site가 바뀌어도 같은 면적이 나왔음.
현장별 격리를 위해 FK + 인덱스 추가. nullable — site 미지정 비행도 허용.

주의: 이 프로젝트는 현재 마이그레이션 그래프가 2 heads 상태
  - head A: 0003 (20종 파이프라인, MS 브랜치)
  - head B: c8f1d2e4a7b9 (image_crop_path, Hijin 브랜치) ← 이 리비전이 연결
통합 배포 전 누군가 merge 리비전 하나 만들어줘야 함 (`alembic merge -m "merge heads" <id1> <id2>`).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4c9a8b27f10"
down_revision: Union[str, None] = "c8f1d2e4a7b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "telemetry_logs",
        sa.Column(
            "site_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="연결 현장 ID (nullable — 현장 미지정 비행 허용)",
        ),
    )
    op.create_foreign_key(
        "fk_telemetry_logs_site_id_sites",
        source_table="telemetry_logs",
        referent_table="sites",
        local_cols=["site_id"],
        remote_cols=["id"],
        ondelete="SET NULL",  # site 삭제돼도 비행 기록은 보존
    )
    op.create_index(
        "idx_telemetry_site_id",
        "telemetry_logs",
        ["site_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_telemetry_site_id", table_name="telemetry_logs")
    op.drop_constraint("fk_telemetry_logs_site_id_sites", "telemetry_logs", type_="foreignkey")
    op.drop_column("telemetry_logs", "site_id")
