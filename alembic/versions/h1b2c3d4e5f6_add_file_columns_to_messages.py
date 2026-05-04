"""add file columns to messages

Revision ID: h1b2c3d4e5f6
Revises: g1a2b3c4d5e6
Create Date: 2026-04-27

채팅 첨부파일 전송 기능: Message 테이블에 파일 관련 컬럼 추가.
text를 nullable로 변경하여 파일만 보내는 경우도 지원.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "h1b2c3d4e5f6"
down_revision: Union[str, None] = "g1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("file_url", sa.String(500), nullable=True, comment="첨부파일 URL 경로"))
    op.add_column("messages", sa.Column("file_name", sa.String(300), nullable=True, comment="원본 파일명"))
    op.add_column("messages", sa.Column("file_content_type", sa.String(100), nullable=True, comment="MIME 타입"))
    op.alter_column("messages", "text", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    op.alter_column("messages", "text", existing_type=sa.Text(), nullable=False)
    op.drop_column("messages", "file_content_type")
    op.drop_column("messages", "file_name")
    op.drop_column("messages", "file_url")
