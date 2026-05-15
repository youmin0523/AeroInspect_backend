# =============================================
# app/models/slam_map.py
# 역할: SLAM 맵 데이터 ORM 모델
#       - SLAM Toolbox로 생성된 맵 스냅샷 저장
#       - 점유 격자(occupancy grid) 메타데이터 + 이미지
# 테이블명: slam_maps
# =============================================

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Text,
    DateTime, ForeignKey, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class SlamMap(Base):
    """
    SLAM 맵 스냅샷 테이블.
    드론이 탐색하며 생성한 지도 데이터 1건 = 1 레코드.
    """
    __tablename__ = "slam_maps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 멀티테넌트 격리 ─────────────────────
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id"),
        nullable=True,
        index=True,
        comment="소속 조직 ID (멀티테넌트 격리 기준)",
    )

    # ── 맵 메타데이터 ────────────────────────
    name = Column(String(200), comment="맵 이름 (예: 101동 25층)")
    resolution = Column(Float, comment="격자 해상도 (m/pixel)")
    width = Column(Integer, comment="맵 너비 (pixels)")
    height = Column(Integer, comment="맵 높이 (pixels)")

    # ── 원점 좌표 ────────────────────────────
    origin_x = Column(Float, comment="맵 원점 X (m)")
    origin_y = Column(Float, comment="맵 원점 Y (m)")
    origin_yaw = Column(Float, comment="맵 원점 Yaw (rad)")

    # ── 맵 데이터 ────────────────────────────
    # Base64 인코딩된 PNG occupancy grid 이미지
    map_image = Column(Text, comment="점유 격자 이미지 (Base64 PNG)")

    # 추가 메타데이터 (SLAM 파라미터 등)
    metadata_ = Column("metadata", JSONB, comment="SLAM 파라미터 JSON")

    # ── 상태 ─────────────────────────────────
    status = Column(
        String(20),
        default="mapping",
        comment="상태 (mapping / completed / failed)"
    )

    # ── 타임스탬프 ───────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_slam_map_status", "status"),
        Index("idx_slam_maps_org_id", "organization_id"),
    )

    def __repr__(self):
        return f"<SlamMap id={self.id} name={self.name} status={self.status}>"
