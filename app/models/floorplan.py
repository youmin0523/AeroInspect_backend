# =============================================
# app/models/floorplan.py
# 역할: 평면도 업로드 ORM 모델
# 테이블명: floorplans
# =============================================

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class Floorplan(Base):
    """
    평면도 업로드 테이블.
    업로드된 JPG/PDF/DXF → OpenCV 처리 → Gazebo .world 생성 추적.
    """
    __tablename__ = "floorplans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 파일 정보 ────────────────────────────
    filename = Column(String(255), nullable=False, comment="원본 파일명")
    content_type = Column(String(100), comment="MIME 타입 (image/jpeg, application/pdf 등)")
    file_path = Column(String(500), comment="서버 저장 경로")

    # ── 처리 상태 ────────────────────────────
    status = Column(
        String(20),
        default="uploaded",
        comment="상태 (uploaded / processing / completed / failed)"
    )

    # ── 처리 결과 ────────────────────────────
    wall_count = Column(Integer, comment="추출된 벽체 라인 수")
    walls_data = Column(JSONB, comment="벽체 좌표 JSON [{x1,y1,x2,y2}, ...]")
    gazebo_world_path = Column(String(500), comment="생성된 .world 파일 경로")
    error_message = Column(Text, comment="처리 실패 시 오류 메시지")

    # ── 스케일 보정 (FR-015) ─────────────────
    # 사용자가 평면도 위 두 점을 찍고 "이 거리는 실제 3m" 라고 지정 → px/m 환산
    # 이후 모든 길이 계산(면적, 커버리지)이 미터 단위로 가능
    scale_px_per_meter = Column(Float, comment="1m 당 픽셀 수 (환산 계수)")
    scale_reference = Column(
        JSONB,
        comment="사용자 지정 기준 {p1:[x,y], p2:[x,y], real_length_m:float}",
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

    def __repr__(self):
        return f"<Floorplan id={self.id} filename={self.filename} status={self.status}>"
