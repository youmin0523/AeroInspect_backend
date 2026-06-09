"""case-insensitive unique email (lower(email)) + 기존 이메일 정규화

Revision ID: q9d0e1f2a3b4
Revises: p8c9d0e1f2a3
Create Date: 2026-06-09 00:00:00.000000

대소문자 무시 이메일 유일성 강제:
  1) 충돌이 없는 행만 email 을 lowercase 로 정규화(안전한 UPDATE).
     - lower(email) 이 다른 행과 겹치는 경우는 건드리지 않는다(데이터 손실 0).
  2) lower(email) 에 UNIQUE 함수 인덱스 생성.
     - 진짜 case-collision(대소문자만 다른 실제 중복 계정)이 남아 있으면 이 단계가
       명시적으로 실패한다 → 운영자가 수동 병합/정리 후 재시도해야 한다(자동 삭제 금지).

멱등: CREATE UNIQUE INDEX IF NOT EXISTS — dev(create_all)가 만든 경우와 충돌하지 않음.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "q9d0e1f2a3b4"
down_revision: Union[str, None] = "p8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) 충돌 없는 행만 lowercase 정규화 (기존 case-sensitive UNIQUE(email) 위반 방지)
    op.execute(
        """
        UPDATE users u
        SET email = lower(u.email)
        WHERE u.email <> lower(u.email)
          AND NOT EXISTS (
              SELECT 1 FROM users o
              WHERE o.id <> u.id AND lower(o.email) = lower(u.email)
          )
        """
    )
    # 2) lower(email) UNIQUE 함수 인덱스 — 잔존 case-collision 있으면 여기서 실패(의도된 안전장치)
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower ON users (lower(email))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_users_email_lower")
