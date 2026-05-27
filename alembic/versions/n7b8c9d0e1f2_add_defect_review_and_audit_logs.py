"""add defect review meta, detection_model_id, GPS, and audit_logs

Revision ID: n7b8c9d0e1f2
Revises: m6a7b8c9d0e1
Create Date: 2026-05-27 00:00:00.000000

Track B (감사·신뢰성 기반) 도입:
  - defect_logs 컬럼 추가:
      * review_status (Enum pending/approved/rejected/flagged_false_positive)
      * reviewed_by_user_id (FK users.id, SET NULL)
      * reviewed_at, review_note
      * detection_model_id (탐지 모델 출처 식별)
      * gps_lat / gps_lon / gps_alt (WGS84 GPS 좌표)
      * 인덱스: idx_defect_review_status, idx_defect_reviewer
  - audit_logs 테이블 신규:
      * 누가/언제/무엇을/어떻게 변경했는지 영속 기록
      * before/after JSONB 스냅샷 (민감 키 redact 후 저장)
      * request_id 로 structlog 와 연결
      * 인덱스 4종 (org/user/resource/action × 시간 DESC)

설계 요지:
  - 입주자 분쟁·내부 감사·법적 책임 추적 기반 인프라
  - APP_ENV=production 에서 alembic upgrade head 1회만 실행하면 적용 완료
  - down_revision 만 변경하면 분리/통합 repo 양쪽 동기 가능
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "n7b8c9d0e1f2"
down_revision: Union[str, None] = "m6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── defect_logs 컬럼 추가 ─────────────────────
    # 1) 신규 Enum (Postgres CREATE TYPE 자동)
    defect_review_status_enum = sa.Enum(
        "pending", "approved", "rejected", "flagged_false_positive",
        name="defect_review_status_enum",
    )
    defect_review_status_enum.create(op.get_bind(), checkfirst=True)

    # 2) review_status (server_default=pending → 기존 레코드도 즉시 채워짐)
    op.add_column(
        "defect_logs",
        sa.Column(
            "review_status",
            defect_review_status_enum,
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "defect_logs",
        sa.Column("reviewed_by_user_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("review_note", sa.Text(), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("detection_model_id", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("gps_lat", sa.Float(), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("gps_lon", sa.Float(), nullable=True),
    )
    op.add_column(
        "defect_logs",
        sa.Column("gps_alt", sa.Float(), nullable=True),
    )

    # 3) FK + 인덱스
    op.create_foreign_key(
        "fk_defect_logs_reviewer",
        "defect_logs", "users",
        ["reviewed_by_user_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_defect_review_status",
        "defect_logs",
        ["review_status", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_defect_reviewer",
        "defect_logs",
        ["reviewed_by_user_id", "reviewed_at"],
    )

    # ── audit_logs 테이블 신규 ────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("organization_id", sa.UUID(), nullable=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("resource_type", sa.String(length=40), nullable=False),
        sa.Column("resource_id", sa.UUID(), nullable=True),
        sa.Column("before", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("after", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("ip", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "idx_audit_org_ts",
        "audit_logs",
        ["organization_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_audit_user_ts",
        "audit_logs",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_audit_resource_ts",
        "audit_logs",
        ["resource_type", "resource_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_audit_action_ts",
        "audit_logs",
        ["action", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    # audit_logs 제거
    op.drop_index("idx_audit_action_ts", table_name="audit_logs")
    op.drop_index("idx_audit_resource_ts", table_name="audit_logs")
    op.drop_index("idx_audit_user_ts", table_name="audit_logs")
    op.drop_index("idx_audit_org_ts", table_name="audit_logs")
    op.drop_table("audit_logs")

    # defect_logs 신규 인덱스/FK/컬럼 제거 (역순)
    op.drop_index("idx_defect_reviewer", table_name="defect_logs")
    op.drop_index("idx_defect_review_status", table_name="defect_logs")
    op.drop_constraint("fk_defect_logs_reviewer", "defect_logs", type_="foreignkey")
    op.drop_column("defect_logs", "gps_alt")
    op.drop_column("defect_logs", "gps_lon")
    op.drop_column("defect_logs", "gps_lat")
    op.drop_column("defect_logs", "detection_model_id")
    op.drop_column("defect_logs", "review_note")
    op.drop_column("defect_logs", "reviewed_at")
    op.drop_column("defect_logs", "reviewed_by_user_id")
    op.drop_column("defect_logs", "review_status")

    # Enum 제거 (Postgres DROP TYPE)
    sa.Enum(name="defect_review_status_enum").drop(op.get_bind(), checkfirst=True)
