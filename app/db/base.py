# =============================================
# app/db/base.py
# 역할: SQLAlchemy 비동기 엔진 및 Base 클래스 정의
#       - create_async_engine으로 asyncpg 드라이버 기반 비동기 연결 풀 생성
#       - declarative_base()로 모든 ORM 모델의 공통 Base 제공
#       - 이 파일의 engine과 Base를 session.py, init_db.py에서 임포트해 사용
# =============================================

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import declarative_base

from app.config import settings

# ── 비동기 엔진 생성 ──────────────────────────
# pool_size/max_overflow: 동시 연결 상한 (SSE·WS 가 세션을 오래 점유 → config 로 상향).
# pool_timeout: 풀 고갈 시 대기 한도(초) — 무한 대기 대신 빠르게 실패.
# pool_recycle: 유휴 커넥션 재생성 주기 — 클라우드 PG 의 idle 연결 끊김 대비.
# pool_pre_ping: 연결 유효성 사전 확인 (네트워크 끊김 방지).
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,
    echo=False,  # SQL 로그 출력 여부 (개발 시 True로 변경 가능)
)

# ── ORM 베이스 클래스 ─────────────────────────
# 모든 모델 클래스는 이 Base를 상속받는다
Base = declarative_base()
