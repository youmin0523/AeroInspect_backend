"""add ai_chat_threads and ai_chat_messages tables

Revision ID: m6a7b8c9d0e1
Revises: k4e5f6a7b8c9
Create Date: 2026-05-15 00:00:00.000000

OpenAI 챗봇 도입:
  - ai_chat_threads: 사용자별 대화방 (ChatGPT 스타일, 영속화)
  - ai_chat_messages: 메시지 (role=user/assistant/system)

설계 요지:
  - 멀티테넌트 격리: user_id + organization_id 이중 키
  - 컨텍스트 압축: thread.summary 로 오래된 턴 압축 저장
  - soft delete: archived_at IS NULL = 활성
  - 인덱스: (user_id, last_message_at DESC) 목록 조회 / (thread_id, created_at ASC) 히스토리

⚠ FK 사이클 회피:
  threads → messages (summary_until_message_id) / messages → threads (thread_id)
  CREATE 순서: threads 먼저 (FK 임시 미설정) → messages → ALTER threads FK 추가.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "m6a7b8c9d0e1"
down_revision: Union[str, None] = "k4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── ai_chat_threads (1단계: FK 사이클 회피 위해 summary_until_message_id FK 임시 미생성) ─
    op.create_table(
        "ai_chat_threads",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(200), nullable=True, comment="대화방 제목"),
        sa.Column(
            "last_message_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            comment="목록 정렬 기준",
        ),
        sa.Column("summary", sa.Text(), nullable=True, comment="이전 대화 압축 요약"),
        sa.Column("summary_until_message_id", sa.UUID(), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True, comment="soft delete"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name="fk_ai_chat_threads_user_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"],
            name="fk_ai_chat_threads_organization_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_ai_threads_user_last",
        "ai_chat_threads",
        ["user_id", sa.text("last_message_at DESC")],
        unique=False,
    )
    op.create_index(
        "idx_ai_threads_org",
        "ai_chat_threads",
        ["organization_id"],
        unique=False,
    )

    # ── ai_chat_messages ─────────────────────────────────────
    op.create_table(
        "ai_chat_messages",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("thread_id", sa.UUID(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", "system", name="ai_chat_role_enum"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("meta", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"], ["ai_chat_threads.id"],
            name="fk_ai_chat_messages_thread_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_ai_messages_thread_created",
        "ai_chat_messages",
        ["thread_id", sa.text("created_at ASC")],
        unique=False,
    )

    # ── 2단계: threads.summary_until_message_id → messages.id FK 후추가 ─
    op.create_foreign_key(
        "fk_ai_chat_threads_summary_until_message_id",
        "ai_chat_threads",
        "ai_chat_messages",
        ["summary_until_message_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # FK 사이클 역순 해제
    op.drop_constraint(
        "fk_ai_chat_threads_summary_until_message_id",
        "ai_chat_threads",
        type_="foreignkey",
    )
    op.drop_index("idx_ai_messages_thread_created", table_name="ai_chat_messages")
    op.drop_table("ai_chat_messages")
    op.execute("DROP TYPE IF EXISTS ai_chat_role_enum")

    op.drop_index("idx_ai_threads_org", table_name="ai_chat_threads")
    op.drop_index("idx_ai_threads_user_last", table_name="ai_chat_threads")
    op.drop_table("ai_chat_threads")
