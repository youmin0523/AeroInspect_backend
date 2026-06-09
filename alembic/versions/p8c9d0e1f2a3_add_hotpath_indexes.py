"""add hot-path indexes (defect_logs.site_id, org_members user+status, conversations.org)

Revision ID: p8c9d0e1f2a3
Revises: n7b8c9d0e1f2
Create Date: 2026-06-09 00:00:00.000000

지연(latency) 핫패스 인덱스 3종 추가:
  - idx_defect_site_ts  : defect_logs(site_id, timestamp DESC)
      조직 스코프 조회(site_id IN (...)) + 최신순 정렬을 단일 인덱스로 커버.
      defect_logs 는 최대 볼륨 테이블이라 site_id 무인덱스 시 seq-scan 비용이 큼.
  - idx_org_member_user_status : organization_members(user_id, status)
      get_current_org_member 가 거의 모든 인증 요청에서 user_id+status='active' 로 조회.
  - idx_conversations_org : conversations(organization_id)
      조직 스코프 대화방 조회.

안전성:
  - 전부 CREATE INDEX IF NOT EXISTS (추가/가역적). dev(create_all) 가 이미 만든 경우와
    충돌하지 않도록 IF NOT EXISTS 사용 → 멱등.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "p8c9d0e1f2a3"
down_revision: Union[str, None] = "n7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_defect_site_ts "
        "ON defect_logs (site_id, timestamp DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_org_member_user_status "
        "ON organization_members (user_id, status)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_org "
        "ON conversations (organization_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_conversations_org")
    op.execute("DROP INDEX IF EXISTS idx_org_member_user_status")
    op.execute("DROP INDEX IF EXISTS idx_defect_site_ts")
