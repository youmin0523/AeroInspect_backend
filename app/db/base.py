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
# pool_size: 기본 연결 풀 크기 (단일 워커 환경에서 5면 충분)
# max_overflow: 최대 추가 연결 수
# pool_pre_ping: 연결 유효성 사전 확인 (네트워크 끊김 방지)
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,  # SQL 로그 출력 여부 (개발 시 True로 변경 가능)
)

# ── ORM 베이스 클래스 ─────────────────────────
# 모든 모델 클래스는 이 Base를 상속받는다
Base = declarative_base()
