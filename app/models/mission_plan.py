# =============================================
# app/models/mission_plan.py
# 역할: 자율비행 미션(Mission) 메타·상태 ORM 모델
#       - 1개 점검 세션의 자율비행 계획 1건 = 1 레코드
#       - 미션 FSM 현재 상태, 그리드 plan, 룸 진행 인덱스 보유
#       - coverage_grids / slam_pointclouds / room_topologies 의 모(Parent)
# 테이블명: mission_plans
# =============================================

import uuid

from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey,
    Enum as SAEnum, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class MissionPlan(Base):
    """
    자율비행 미션 1건.
    /api/v1/mission/start 호출 시 1 레코드 생성, FSM 진행에 따라 status·current_phase 갱신.
    """

    __tablename__ = "mission_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── FK ──────────────────────────────────
    site_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sites.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="대상 현장 ID",
    )
    schedule_id = Column(
        UUID(as_uuid=True),
        ForeignKey("inspection_schedules.id", ondelete="SET NULL"),
        nullable=True,
        comment="연결된 점검 일정 (없으면 즉석 미션)",
    )
    operator_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="미션 시작 운영자",
    )

    # ── 미션 상태 (FSM) ─────────────────────
    status = Column(
        SAEnum(
            "pending", "running", "paused",
            "completed", "aborted", "failsafe",
            name="mission_status_enum",
        ),
        nullable=False,
        default="pending",
        comment="미션 전체 상태",
    )
    current_phase = Column(
        SAEnum(
            "idle", "arm", "takeoff", "mapping", "verification", "path_plan",
            "coverage_fly", "room_transition", "complete", "land", "failsafe",
            name="mission_phase_enum",
        ),
        nullable=False,
        default="idle",
        comment="현재 FSM 단계",
    )
    current_room_idx = Column(
        Integer, nullable=True,
        comment="진행 중인 룸 토폴로지 노드 인덱스",
    )

    # ── 계획 / 파라미터 ─────────────────────
    # 그리드 spacing, overlap, 수직 레이어, 속도 상한 등
    params_json = Column(JSONB, nullable=False, default=dict, comment="미션 파라미터 JSON")
    # 룸별 보스트로페돈 경로(WP 리스트), 히트맵 메타 등
    plan_json = Column(JSONB, nullable=True, comment="생성된 미션 계획 JSON")

    # ── 실패/사유 ───────────────────────────
    failure_reason = Column(String(200), nullable=True, comment="abort/failsafe 사유")

    # ── 시각 ────────────────────────────────
    started_at = Column(DateTime(timezone=True), nullable=True, comment="ARM 시각")
    finished_at = Column(DateTime(timezone=True), nullable=True, comment="LAND 완료 시각")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_mission_site_status", "site_id", "status"),
        Index("idx_mission_status_created", "status", created_at.desc()),
    )

    def __repr__(self):
        return (
            f"<MissionPlan id={self.id} site={self.site_id} "
            f"status={self.status} phase={self.current_phase}>"
        )
