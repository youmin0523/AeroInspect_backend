"""add organization_id to floorplans and slam_maps

Revision ID: j3d4e5f6a7b8
Revises: 89b53c16de85
Create Date: 2026-05-13 00:00:00.000000

멀티테넌트 데이터 격리:
  - floorplans.organization_id (UUID FK organizations.id, nullable)
  - slam_maps.organization_id  (UUID FK organizations.id, nullable)

운영 노트:
  - nullable=True 는 점진 마이그레이션을 위한 한시 허용. 기존 로우는 NULL.
  - 신규 업로드는 라우터에서 자동으로 owning org 를 기록한다.
  - 기존 NULL 로우는 조직 격리 필터에 걸리지 않으므로 라우터가 반환하지 않는다.
    필요 시 운영자가 다음 SQL 로 backfill:
        UPDATE floorplans SET organization_id = '<org-uuid>' WHERE organization_id IS NULL;
        UPDATE slam_maps  SET organization_id = '<org-uuid>' WHERE organization_id IS NULL;
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "j3d4e5f6a7b8"
down_revision: Union[str, None] = "89b53c16de85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── floorplans ─────────────────────────────────────────────
    op.add_column(
        "floorplans",
        sa.Column(
            "organization_id",
            sa.UUID(),
            nullable=True,
            comment="소속 조직 ID (멀티테넌트 격리 기준)",
        ),
    )
    op.create_index(
        "idx_floorplans_org_id",
        "floorplans",
        ["organization_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_floorplans_organization_id",
        "floorplans",
        "organizations",
        ["organization_id"],
        ["id"],
    )

    # ── slam_maps ──────────────────────────────────────────────
    op.add_column(
        "slam_maps",
        sa.Column(
            "organization_id",
            sa.UUID(),
            nullable=True,
            comment="소속 조직 ID (멀티테넌트 격리 기준)",
        ),
    )
    op.create_index(
        "idx_slam_maps_org_id",
        "slam_maps",
        ["organization_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_slam_maps_organization_id",
        "slam_maps",
        "organizations",
        ["organization_id"],
        ["id"],
    )


def downgrade() -> None:
    # ── slam_maps ──────────────────────────────────────────────
    op.drop_constraint("fk_slam_maps_organization_id", "slam_maps", type_="foreignkey")
    op.drop_index("idx_slam_maps_org_id", table_name="slam_maps")
    op.drop_column("slam_maps", "organization_id")

    # ── floorplans ─────────────────────────────────────────────
    op.drop_constraint("fk_floorplans_organization_id", "floorplans", type_="foreignkey")
    op.drop_index("idx_floorplans_org_id", table_name="floorplans")
    op.drop_column("floorplans", "organization_id")
