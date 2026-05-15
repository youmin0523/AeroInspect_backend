# =============================================
# scripts/backfill_chat_titles.py
# 역할: R-v1.1.05 이전에 생성된 기존 AiChatThread 의 자동 제목을
#       "대화 흐름 요약" 으로 1회 일괄 재생성.
#
# 배경:
#   R-v1.1.05 (2026-05-15) 에서 자동 제목 로직이 개선됨 — 첫 3턴 동안 매 응답 후 갱신
#   + 프롬프트 강화. 그러나 그 이전 thread 는 "안녕하세요" / "제목 없음" 같이 부실한
#   제목으로 굳은 상태. 이 스크립트가 1회 일괄 보정.
#
# 동작:
#   - 활성(archived_at IS NULL) thread 중 user 메시지 ≥ 1 인 thread 만 대상
#   - openai_chat_service.regenerate_thread_title() 호출 (현재 코드의 흐름 요약 로직 그대로)
#   - LLM 호출 비용은 thread 당 약 $0.0001 (gpt-4o-mini, 40 토큰 출력) — 무시 가능
#
# 사용 (운영 컨테이너):
#   flyctl ssh console -a aeroinspect-backend -C "python -m scripts.backfill_chat_titles --dry-run"
#   flyctl ssh console -a aeroinspect-backend -C "python -m scripts.backfill_chat_titles"
#
# 안전장치:
#   - --dry-run : 대상 thread 목록만 출력하고 실제 LLM/DB 변경 X
#   - 직렬 처리(동시 LLM 호출 없음) — race 회피 + Rate Limit 보수적
#   - 개별 thread 실패는 무시하고 다음으로 진행 (서비스의 try/except 활용)
# =============================================

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# scripts/ 에서 실행해도 app.* import 가능하게 backend/ 를 path 에 추가
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select, func

from app.db.session import async_session_factory
from app.models.ai_chat import AiChatMessage, AiChatThread
from app.services.openai_chat import openai_chat_service


async def _collect_targets():
    """활성 thread 중 user 메시지가 1개 이상인 것 + 기존 제목 / user 메시지 수 반환."""
    async with async_session_factory() as session:
        # user 메시지가 있는 활성 thread 만
        q = (
            select(
                AiChatThread.id,
                AiChatThread.title,
                func.count(AiChatMessage.id).label("user_msg_count"),
            )
            .join(AiChatMessage, AiChatMessage.thread_id == AiChatThread.id)
            .where(AiChatThread.archived_at.is_(None))
            .where(AiChatMessage.role == "user")
            .group_by(AiChatThread.id, AiChatThread.title)
            .having(func.count(AiChatMessage.id) >= 1)
            .order_by(AiChatThread.last_message_at.desc())
        )
        rows = (await session.execute(q)).all()
        return [(r.id, r.title, r.user_msg_count) for r in rows]


async def main():
    parser = argparse.ArgumentParser(description="기존 chat thread 제목 일괄 흐름 요약 재생성")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="대상 thread 목록만 출력하고 LLM/DB 변경은 하지 않음",
    )
    args = parser.parse_args()

    targets = await _collect_targets()
    print(f"[backfill] 대상 thread: {len(targets)}건")
    for tid, title, n in targets:
        print(f"  - {tid}  user_msgs={n}  기존 제목={title!r}")

    if args.dry_run:
        print("[backfill] --dry-run 이므로 실제 갱신은 건너뜁니다.")
        return

    if not targets:
        print("[backfill] 갱신 대상 없음 — 종료.")
        return

    print(f"[backfill] {len(targets)}건 순차 갱신 시작…")
    ok = 0
    for tid, _title, _n in targets:
        try:
            await openai_chat_service.regenerate_thread_title(tid)
            ok += 1
        except Exception as exc:  # 서비스 내부 try 가 잡아내지만 이중 안전망
            print(f"  [skip] {tid} 갱신 실패: {exc}")

    # 갱신 결과 출력
    async with async_session_factory() as session:
        for tid, _title, _n in targets:
            new = await session.scalar(select(AiChatThread.title).where(AiChatThread.id == tid))
            print(f"  - {tid}  새 제목={new!r}")

    print(f"[backfill] 완료: {ok}/{len(targets)} 갱신.")


if __name__ == "__main__":
    asyncio.run(main())
