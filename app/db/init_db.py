# =============================================
# app/db/init_db.py
# 역할: 애플리케이션 시작 시 DB 테이블 자동 생성 (개발/테스트 전용)
#       - Base.metadata.create_all로 존재하지 않는 테이블을 생성
#       - 약관 시드 데이터 삽입 (idempotent)
#
# ⚠️ 운영(APP_ENV=production)에서는 create_all 을 자동 스킵.
#   운영 스키마는 반드시 `alembic upgrade head` 로 적용해야 한다.
#   이유: create_all 은 ALTER 를 못 해서 alembic 과 상태가 어긋나면 silent 실수가 누적되고,
#         alembic 마이그레이션이 "이미 존재함" 에러로 실패할 수 있음.
#
# 호출: main.py lifespan 핸들러에서 await init_db()
# =============================================

import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import APP_ENV_VAR, PROD_ENV_VALUES
from app.db.base import engine, Base
from app.db.session import async_session_factory

# ORM 모델을 Base에 등록하기 위해 모델 패키지 임포트
# (app/models/__init__.py 에서 모든 모델을 일괄 임포트해 메타데이터에 등록)
from app.models import (  # noqa: F401
    defect,
    user,
    business_profile,
    term,
    user_term_agreement,
)
from app.models.term import Term


# ── 약관 시드 데이터 ─────────────────────────
_TERM_SEEDS = [
    {"code": "service",   "title": "서비스 이용약관",       "is_required": True,  "version": "1.0"},
    {"code": "privacy",   "title": "개인정보 수집·이용 동의", "is_required": True,  "version": "1.0"},
    {"code": "marketing", "title": "마케팅 정보 수신 동의",  "is_required": False, "version": "1.0"},
]


async def _seed_terms(session: AsyncSession) -> None:
    """terms 테이블에 기본 약관이 없으면 삽입 (idempotent — 운영에서도 안전)."""
    result = await session.execute(select(Term.code))
    existing_codes = {row[0] for row in result.all()}

    for seed in _TERM_SEEDS:
        if seed["code"] not in existing_codes:
            session.add(Term(**seed))

    await session.commit()


def _is_production() -> bool:
    return os.environ.get(APP_ENV_VAR, "").strip().lower() in PROD_ENV_VALUES


async def init_db() -> None:
    """
    개발/테스트: create_all 로 누락 테이블 자동 생성 + 약관 시드.
    운영(APP_ENV=production): create_all 스킵, 약관 시드만 실행.
        스키마는 alembic 이 책임지므로 create_all 호출 자체가 위험.
    """
    if _is_production():
        # 운영: create_all 비활성. alembic upgrade head 가 이미 적용됐다고 가정.
        # 시드만 idempotent 하게 보장.
        async with async_session_factory() as session:
            await _seed_terms(session)
        return

    # 개발/테스트: 테이블 생성 후 시드.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        await _seed_terms(session)
