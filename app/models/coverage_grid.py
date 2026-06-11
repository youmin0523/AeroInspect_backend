# =============================================
# app/models/coverage_grid.py
# 역할: 자율비행 그리드 커버리지 셀 ORM 모델
#       - 미션 하나당 룸별 3D 그리드 셀(상하 포함) 1개 = 1 레코드
#       - 셀이 RGB+Thermal 양쪽으로 captured 되었는지 추적
#       - 배터리 교체 후 resume 시 captured=False 셀만 자동 재진입
# 테이블명: coverage_grids
# =============================================

import uuid

from sqlalchemy import (
    Column, Boolean, Integer, Float, String, DateTime, ForeignKey,
    Index, UniqueConstraint, func,
)  # noqa: F401  (Float used by face meta below)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class CoverageGrid(Base):
    """
    자율비행 3D 그리드 셀.
    cell_z 는 수직 레이어 인덱스 (예: 0=바닥, 1=중층, 2=천장).
    """

    __tablename__ = "coverage_grids"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── FK ──────────────────────────────────
    mission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("mission_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── 셀 좌표 ─────────────────────────────
    room_idx = Column(Integer, nullable=False, comment="room_topology 노드 인덱스")
    cell_x = Column(Integer, nullable=False)
    cell_y = Column(Integer, nullable=False)
    cell_z = Column(
        Integer, nullable=False,
        comment="면 코드 (0=floor, 1..N=wall_idx, N+1=ceiling, 그 위=window_idx)",
    )

    # ── 4면 정밀 스캔 메타 ────────────────────
    face_kind = Column(
        String(16), nullable=False, default="wall",
        comment="wall | ceiling | floor | window",
    )
    face_idx = Column(
        Integer, nullable=False, default=0,
        comment="같은 룸 내 face 인덱스 (벽 0..N-1, 천장/바닥=0, 창호 0..M-1)",
    )
    cam_pitch_rad = Column(
        Float, nullable=False, default=0.0,
        comment="카메라 틸트 메타 (천장 +pi/2, 바닥 -pi/2, 벽/창호 0)",
    )

    # 셀 중심 월드 좌표(WP)
    world_x = Column(Float, nullable=False)
    world_y = Column(Float, nullable=False)
    world_z = Column(Float, nullable=False)

    # ── 캡처 상태 ───────────────────────────
    captured = Column(Boolean, nullable=False, default=False, index=True)
    rgb_image_id = Column(String(128), nullable=True, comment="RGB 캡처 파일 식별자")
    thermal_image_id = Column(String(128), nullable=True, comment="Thermal 캡처 파일 식별자")
    captured_at = Column(DateTime(timezone=True), nullable=True)

    # ── 감사 ────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "mission_id", "room_idx", "cell_x", "cell_y", "cell_z",
            name="uq_coverage_cell",
        ),
        Index("idx_coverage_mission_room_cap", "mission_id", "room_idx", "captured"),
    )

    def __repr__(self):
        return (
            f"<CoverageGrid mission={self.mission_id} room={self.room_idx} "
            f"cell=({self.cell_x},{self.cell_y},{self.cell_z}) cap={self.captured}>"
        )
