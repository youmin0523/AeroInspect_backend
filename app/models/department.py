# =============================================
# app/models/department.py
# 역할: 부서 ORM 모델
#       - 조직별 부서 관리 (추가/삭제/이름변경)
#       - 슈퍼어드민 + 조직 관리자 모두 수정 가능
# 테이블명: departments
# =============================================

import uuid
from sqlalchemy import (
    Column, String, DateTime, Index, func, ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class Department(Base):
    """
    조직별 부서 테이블.
    조직 관리자 또는 슈퍼어드민이 부서를 추가/삭제/이름변경 가능.
    """
    __tablename__ = "departments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        comment="소속 조직 ID",
    )

    name = Column(String(100), nullable=False, comment="부서명")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_dept_org_name"),
        Index("idx_dept_org_id", "organization_id"),
    )

    def __repr__(self):
        return f"<Department id={self.id} name={self.name}>"
