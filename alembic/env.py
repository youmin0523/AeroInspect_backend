# =============================================
# alembic/env.py
# 역할: Alembic 마이그레이션 환경 설정
#       - 비동기 SQLAlchemy 엔진을 사용하는 Alembic async 패턴 적용
#       - .env 파일의 DATABASE_URL을 자동으로 읽어 연결
#       - 모든 ORM 모델을 Base에 등록하여 autogenerate 지원
# 사용: alembic revision --autogenerate -m "설명"
# =============================================

import asyncio
import os
import sys
from logging.config import fileConfig

# alembic CLI는 작업 디렉토리를 sys.path에 안 넣으므로
# 프로젝트 루트(backend/)를 수동 추가 — 이래야 `from app.db.base import Base`가 동작.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Alembic Config 객체
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── 모델 임포트 (autogenerate를 위해 모든 모델 등록) ──
from app.db.base import Base
from app.models import *  # noqa: F401, F403

target_metadata = Base.metadata

# ── DATABASE_URL을 .env에서 로드 ──────────────
from app.config import settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """오프라인 모드: URL만으로 마이그레이션 생성"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """비동기 엔진으로 마이그레이션 실행"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
