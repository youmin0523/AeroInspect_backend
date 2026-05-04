# =============================================
# alembic/versions/0002_defect_class_display.py
# 역할: defect_logs 테이블에 3-모델 파이프라인 컬럼 추가 +
#       레거시 A-E taxonomy 컬럼을 NULLABLE로 완화.
#
# 추가 컬럼:
#   - defect_source (enum: yolo_thermal | yolo_delam | wallpaper)
#   - defect_class (varchar 50, 모델 내부 클래스명 — 예: 'Crack', 'good')
#   - defect_class_display_en (varchar 80)
#   - defect_class_display_ko (varchar 80)
#
# 변경 컬럼 (NOT NULL → NULL):
#   - area, category_code, defect_type
#
# ⚠️ down_revision=None: 이 마이그레이션이 레포 첫 Alembic revision.
#    기존 DB는 `alembic stamp head`로 현재 스키마를 베이스라인 처리한 뒤
#    `alembic upgrade head`로 이 마이그레이션만 적용하는 절차를 README에 명시.
# =============================================

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision: str = "0002_defect_class_display"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# enum 타입 이름 (PostgreSQL)
DEFECT_SOURCE_ENUM_NAME = "defect_source_enum"
DEFECT_SOURCE_VALUES = ("yolo_thermal", "yolo_delam", "wallpaper")


def upgrade() -> None:
    # ── enum 타입 생성 (이미 존재하면 스킵) ──
    defect_source_enum = sa.Enum(
        *DEFECT_SOURCE_VALUES,
        name=DEFECT_SOURCE_ENUM_NAME,
    )
    defect_source_enum.create(op.get_bind(), checkfirst=True)

    # ── 신규 컬럼 4개 추가 ──
    op.add_column(
        "defect_logs",
        sa.Column(
            "defect_source",
            sa.Enum(*DEFECT_SOURCE_VALUES, name=DEFECT_SOURCE_ENUM_NAME, create_type=False),
            nullable=True,
            comment="탐지 모델 (yolo_thermal | yolo_delam | wallpaper)",
        ),
    )
    op.add_column(
        "defect_logs",
        sa.Column("defect_class", sa.String(length=50), nullable=True, comment="모델 내부 클래스명"),
    )
    op.add_column(
        "defect_logs",
        sa.Column(
            "defect_class_display_en",
            sa.String(length=80),
            nullable=True,
            comment="영문 표시명 (예: 'Burst')",
        ),
    )
    op.add_column(
        "defect_logs",
        sa.Column(
            "defect_class_display_ko",
            sa.String(length=80),
            nullable=True,
            comment="한글 표시명 (예: '터짐')",
        ),
    )

    # ── 레거시 컬럼 NOT NULL 해제 ──
    # 신규 3-모델 중 A-E taxonomy로 매핑 안 되는 클래스를 위해 NULL 허용
    op.alter_column("defect_logs", "area", existing_type=sa.String(length=1), nullable=True)
    op.alter_column("defect_logs", "category_code", existing_type=sa.String(length=10), nullable=True)
    op.alter_column("defect_logs", "defect_type", existing_type=sa.String(length=100), nullable=True)


def downgrade() -> None:
    # ── NOT NULL 복원 (기존 레코드에 NULL 있으면 실패하니 주의) ──
    op.alter_column("defect_logs", "area", existing_type=sa.String(length=1), nullable=False)
    op.alter_column("defect_logs", "category_code", existing_type=sa.String(length=10), nullable=False)
    op.alter_column("defect_logs", "defect_type", existing_type=sa.String(length=100), nullable=False)

    # ── 신규 컬럼 4개 제거 ──
    op.drop_column("defect_logs", "defect_class_display_ko")
    op.drop_column("defect_logs", "defect_class_display_en")
    op.drop_column("defect_logs", "defect_class")
    op.drop_column("defect_logs", "defect_source")

    # ── enum 타입 제거 ──
    sa.Enum(*DEFECT_SOURCE_VALUES, name=DEFECT_SOURCE_ENUM_NAME).drop(
        op.get_bind(), checkfirst=True
    )
