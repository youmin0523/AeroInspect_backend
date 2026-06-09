# =============================================
# scripts/provision_db.py
# 역할: 신규(빈) 데이터베이스를 1회 프로비저닝한다.
#       1) Base.metadata.create_all 로 현재 ORM 스키마 전체 생성
#       2) alembic stamp head 로 마이그레이션 히스토리를 head 로 표시
#
# 왜 'alembic upgrade head' 가 아니라 이 방식인가:
#   현재 마이그레이션 히스토리는 핵심 테이블(users/organizations/sites/defect_logs 등)을
#   create 하는 베이스라인이 없고, 초기 리비전들이 곧장 ALTER 부터 수행한다
#   (down_revision=None 루트 2개). 따라서 빈 DB 에 'alembic upgrade head' 를 돌리면
#   "존재하지 않는 테이블에 ALTER" 로 실패한다.
#   → 신규 DB 는 create_all 로 최종 스키마를 한 번에 만들고 stamp head 로 동기화한다.
#     (기존 운영 DB 는 이미 head 로 stamp 되어 있으므로 이 스크립트를 돌리지 않는다.)
#
# 사용:
#   python -m scripts.provision_db            # create_all + stamp head
#   python -m scripts.provision_db --check    # 현재 head / 적용 여부만 출력
#
# 주의: 이미 테이블이 있는 DB 에서 실행해도 create_all 은 checkfirst 라 안전하지만,
#       stamp 는 alembic_version 을 head 로 덮어쓴다. 신규 DB 에만 사용할 것.
# =============================================

import argparse
import asyncio

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db.base import engine, Base
# 모든 모델을 메타데이터에 등록 (create_all 이 전체 테이블을 알도록)
from app.models import *  # noqa: F401,F403


async def _create_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _alembic_config() -> Config:
    return Config("alembic.ini")


def main() -> None:
    parser = argparse.ArgumentParser(description="신규 DB 프로비저닝 (create_all + stamp head)")
    parser.add_argument("--check", action="store_true", help="head 리비전만 출력하고 종료")
    args = parser.parse_args()

    cfg = _alembic_config()
    heads = ScriptDirectory.from_config(cfg).get_heads()
    print(f"[provision] alembic heads: {heads}")
    if args.check:
        return

    print("[provision] 1/2 create_all 로 스키마 생성 중...")
    asyncio.run(_create_all())
    print("[provision] 2/2 alembic stamp head 중...")
    command.stamp(cfg, "head")
    print("[provision] 완료 — 신규 DB 프로비저닝 성공.")


if __name__ == "__main__":
    main()
