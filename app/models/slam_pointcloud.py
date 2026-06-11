# =============================================
# app/models/slam_pointcloud.py
# 역할: Visual-Inertial SLAM 산출 포인트클라우드 스냅샷 ORM 모델
#       - SLAM 키프레임 1건 = 1 레코드 (drop sampling 5Hz 권장)
#       - 실제 점군 데이터(PLY/PCD)는 파일시스템에 저장, 본 테이블은 메타+경로
# 테이블명: slam_pointclouds
# =============================================

import uuid

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, ForeignKey,
    Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class SlamPointcloud(Base):
    """
    SLAM 키프레임 점군 스냅샷.
    파일은 backend/data/pointclouds/{mission_id}/{frame_idx}.ply 형태로 저장.
    """

    __tablename__ = "slam_pointclouds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── FK ──────────────────────────────────
    mission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("mission_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── 키프레임 메타 ───────────────────────
    frame_idx = Column(Integer, nullable=False, comment="미션 내 키프레임 순번")
    file_path = Column(String(512), nullable=False, comment="PLY/PCD 절대경로")
    point_count = Column(Integer, nullable=False, default=0)

    # ── 카메라 포즈 (월드 좌표) ───────────────
    pose_x = Column(Float, nullable=True)
    pose_y = Column(Float, nullable=True)
    pose_z = Column(Float, nullable=True)
    pose_qw = Column(Float, nullable=True)
    pose_qx = Column(Float, nullable=True)
    pose_qy = Column(Float, nullable=True)
    pose_qz = Column(Float, nullable=True)

    # ── 신뢰도 / 부가 ───────────────────────
    slam_confidence = Column(Float, nullable=True, comment="SLAM 트래킹 신뢰도 (0~1)")
    extra = Column(JSONB, nullable=True, comment="기타 메타 (특징점 수 등)")

    # ── 타임스탬프 ──────────────────────────
    ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="키프레임 캡처 시각 (UTC)",
    )

    __table_args__ = (
        Index("idx_pointcloud_mission_frame", "mission_id", "frame_idx", unique=True),
        Index("idx_pointcloud_mission_ts", "mission_id", ts.desc()),
    )

    def __repr__(self):
        return (
            f"<SlamPointcloud id={self.id} mission={self.mission_id} "
            f"frame={self.frame_idx} pts={self.point_count}>"
        )
