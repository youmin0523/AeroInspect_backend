# =============================================
# app/db/session.py
# 역할: 비동기 DB 세션 팩토리 정의
#       - AsyncSession을 생성하는 async_session_factory 제공
#       - 라우터에서는 dependencies.py의 get_db()를 통해 세션을 주입받아 사용
#       - autocommit=False: 명시적 commit/rollback 필요
# =============================================

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import engine

# ── 세션 팩토리 ──────────────────────────────
# async with async_session_factory() as session: 형태로 사용
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # commit 후 객체 접근 시 재조회 방지
)
