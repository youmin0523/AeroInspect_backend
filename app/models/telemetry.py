# =============================================
# app/models/telemetry.py
# 역할: 드론 텔레메트리(좌표/센서) 로그 ORM 모델
#       - 드론 위치(x, y, z), 자세(roll, pitch, yaw)
#       - 배터리, 비행 모드, 센서 상태
# 테이블명: telemetry_logs
# =============================================

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer,
    DateTime, Boolean, Index, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class TelemetryLog(Base):
    """
    드론 텔레메트리 로그 테이블.
    비행 중 주기적으로 수신되는 드론 상태 1건 = 1 레코드.
    """
    __tablename__ = "telemetry_logs"

    # ── 기본 키 ──────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 현장 연결 (커버리지 산출용) ───────────
    # nullable: 드론이 site 정보 없이 기동할 수 있어야 함 (테스트/디버그 비행 등)
    # 인덱스: /api/v1/coverage/{site_id} 쿼리 최적화
    site_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sites.id"),
        nullable=True,
        index=True,
        comment="연결 현장 ID (nullable — 현장 미지정 비행 허용)",
    )

    # ── 위치 (월드 좌표계, 미터) ──────────────
    pos_x = Column(Float, nullable=False, comment="드론 X 좌표 (m)")
    pos_y = Column(Float, nullable=False, comment="드론 Y 좌표 (m)")
    pos_z = Column(Float, nullable=False, comment="드론 Z 좌표 / 고도 (m)")

    # ── 자세 (라디안) ────────────────────────
    roll = Column(Float, comment="Roll (rad)")
    pitch = Column(Float, comment="Pitch (rad)")
    yaw = Column(Float, comment="Yaw (rad)")

    # ── 속도 (m/s) ──────────────────────────
    vel_x = Column(Float, comment="X 방향 속도 (m/s)")
    vel_y = Column(Float, comment="Y 방향 속도 (m/s)")
    vel_z = Column(Float, comment="Z 방향 속도 (m/s)")

    # ── 배터리 & 상태 ────────────────────────
    battery_percent = Column(Float, comment="배터리 잔량 (%)")
    flight_mode = Column(String(30), comment="비행 모드 (예: GUIDED, LAND, RTL)")
    is_armed = Column(Boolean, default=False, comment="시동 여부")

    # ── 센서 상태 ────────────────────────────
    lidar_distance = Column(Float, comment="LiDAR 전방 거리 (m)")
    sensor_status = Column(JSONB, comment="센서 상태 JSON (예: {rgb: true, thermal: true})")

    # ── 타임스탬프 ───────────────────────────
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="수신 시각 (UTC)"
    )

    __table_args__ = (
        Index("idx_telemetry_ts", timestamp.desc()),
    )

    def __repr__(self):
        return (
            f"<TelemetryLog id={self.id} "
            f"pos=({self.pos_x}, {self.pos_y}, {self.pos_z}) "
            f"ts={self.timestamp}>"
        )
