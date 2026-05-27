# =============================================
# app/models/audit_log.py
# 역할: 감사 추적 로그 ORM 모델 정의
#       - 누가(user) 언제(created_at) 무엇을(action) 어떻게(before/after) 변경했는지 영속 기록
#       - 입주자 분쟁·내부 감사·법적 책임 추적 핵심 인프라
#       - structlog request_id 와 연결돼 로그 ↔ DB 양방향 추적 가능
# 테이블명: audit_logs
# 사용 진입점: app/services/audit_logger.py 의 write_audit() 헬퍼
# =============================================

import uuid

from sqlalchemy import (
    Column, String, DateTime, Index, func, ForeignKey, Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class AuditLog(Base):
    """
    감사 로그 테이블.
    하자 검수/리포트 발행/현장 수정/조직원 권한 변경 등 책임 추적이 필요한 모든 사건 1건 = 1 레코드.
    민감 정보(password/token/secret)는 write_audit() 헬퍼에서 redact 후 저장.
    """
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ── 주체 ─────────────────────────────────
    # user_id NULL = 시스템 동작 (AI 자동 탐지, 스케줄러, 외부 webhook 등)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, comment="행위 주체 사용자 (NULL=시스템)")
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, comment="조직 컨텍스트 (다조직 격리)")

    # ── 행위 ─────────────────────────────────
    # action: 점(.) 구분 doted-name. 예: defect.review.approve / defect.review.reject / defect.delete /
    #         report.publish / report.update / site.update / org.member.role_change / auth.login.success
    action = Column(String(80), nullable=False, comment="행위 식별자 (점 구분)")

    # resource_type: defect / report / site / user / organization / floorplan / ai_chat_thread 등
    resource_type = Column(String(40), nullable=False, comment="대상 자원 종류")
    resource_id = Column(UUID(as_uuid=True), nullable=True, comment="대상 자원 ID (없을 수도 있음)")

    # ── 변경 스냅샷 ──────────────────────────
    # before/after: 변경 전·후 직렬화 JSON. CREATE 는 before=None, DELETE 는 after=None.
    # 큰 객체는 keys 일부만 저장(요약). 민감 키 redact 필수.
    before = Column(JSONB, nullable=True, comment="변경 전 상태 (요약 JSON)")
    after = Column(JSONB, nullable=True, comment="변경 후 상태 (요약 JSON)")

    # ── 요청 컨텍스트 ─────────────────────────
    ip = Column(String(45), nullable=True, comment="클라이언트 IP (IPv6 지원 45자)")
    user_agent = Column(String(500), nullable=True, comment="User-Agent 헤더")
    # structlog RequestIDMiddleware 가 부여하는 ID — 로그 ↔ DB 양방향 추적
    request_id = Column(String(64), nullable=True, comment="요청 ID (structlog 연결)")

    # ── 부가 ─────────────────────────────────
    note = Column(Text, nullable=True, comment="자유 형식 사유/메모")

    # ── 시각 ─────────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="기록 시각 (UTC)",
    )

    __table_args__ = (
        Index("idx_audit_org_ts", "organization_id", created_at.desc()),
        Index("idx_audit_user_ts", "user_id", created_at.desc()),
        Index("idx_audit_resource_ts", "resource_type", "resource_id", created_at.desc()),
        Index("idx_audit_action_ts", "action", created_at.desc()),
    )

    def __repr__(self):
        return (
            f"<AuditLog id={self.id} action={self.action} "
            f"resource={self.resource_type}:{self.resource_id} "
            f"user={self.user_id} ts={self.created_at}>"
        )
