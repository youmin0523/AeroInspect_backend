"""add image_crop_path to defect_logs

Revision ID: c8f1d2e4a7b9
Revises: b3f1a2c4e5d6
Create Date: 2026-04-21 10:00:00.000000

Base64로 DB Text 컬럼에 저장하던 이미지를 파일시스템으로 이전.
image_crop(Text) 은 과거 데이터 호환을 위해 유지, 신규는 image_crop_path 사용.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8f1d2e4a7b9'
down_revision: Union[str, None] = 'b3f1a2c4e5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'defect_logs',
        sa.Column(
            'image_crop_path',
            sa.String(length=255),
            nullable=True,
            comment='하자 크롭 이미지 상대 경로 (uploads/ 기준)',
        ),
    )


def downgrade() -> None:
    op.drop_column('defect_logs', 'image_crop_path')
