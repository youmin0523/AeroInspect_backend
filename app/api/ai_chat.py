# =============================================
# app/api/ai_chat.py
# 역할: OpenAI 챗봇 REST + SSE API
#       - GET    /threads               대화방 목록
#       - POST   /threads               대화방 생성
#       - PATCH  /threads/{id}          제목 수정
#       - DELETE /threads/{id}          soft delete (?hard=true 시 영구 삭제)
#       - GET    /threads/{id}/messages 메시지 히스토리 (커서 페이지네이션)
#       - POST   /threads/{id}/messages SSE 스트리밍 응답
#
# 보안:
#   - 모든 엔드포인트 Depends(get_current_org_member) 필수
#   - thread.user_id == 현재 user.id AND thread.organization_id == 현재 org.id 이중 검증
#   - 메시지 전송: 사용자별 분당 20회 in-memory rate limit (단일 워커 가정)
# =============================================

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Optional
from uuid import UUID

from fastapi import (
    APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_org_member
from app.models.ai_chat import AiChatMessage, AiChatThread
from app.schemas.ai_chat import (
    MessageCreate,
    MessageHistoryResponse,
    MessageResponse,
    ThreadCreate,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdate,
)
from app.services.openai_chat import openai_chat_service

router = APIRouter()


# ── 메시지 전송 사용자별 rate limit (분당 N회) ────
# rate_limit.py prefix 매칭은 /threads/{id}/messages 까지 분기 불가능 → 라우터 내부에서 보강.
_USER_MSG_HITS: dict[UUID, Deque[float]] = defaultdict(deque)
_USER_MSG_LOCK = asyncio.Lock()
_MSG_LIMIT_PER_MIN = 20
_MSG_WINDOW_SEC = 60.0


async def _check_user_message_rate(user_id: UUID) -> None:
    """사용자별 분당 메시지 전송 한도. 초과 시 429."""
    now = time.monotonic()
    cutoff = now - _MSG_WINDOW_SEC
    async with _USER_MSG_LOCK:
        q = _USER_MSG_HITS[user_id]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= _MSG_LIMIT_PER_MIN:
            retry = int(_MSG_WINDOW_SEC - (now - q[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"메시지 전송이 너무 잦습니다. {retry}초 후 다시 시도해 주세요.",
            )
        q.append(now)


# ── 권한 검증 헬퍼 ────────────────────────────


async def _load_thread_or_404(
    thread_id: UUID,
    user_id: UUID,
    org_id: UUID,
    db: AsyncSession,
    include_archived: bool = False,
) -> AiChatThread:
    """현재 사용자 + 현재 조직 소유의 thread 만 반환. 그 외 404 (정보 누설 회피)."""
    q = (
        select(AiChatThread)
        .where(AiChatThread.id == thread_id)
        .where(AiChatThread.user_id == user_id)
        .where(AiChatThread.organization_id == org_id)
    )
    if not include_archived:
        q = q.where(AiChatThread.archived_at.is_(None))
    thread = await db.scalar(q)
    if thread is None:
        raise HTTPException(status_code=404, detail="대화방을 찾을 수 없습니다.")
    return thread


def _to_thread_response(t: AiChatThread) -> ThreadResponse:
    return ThreadResponse(
        id=t.id,
        title=t.title,
        last_message_at=t.last_message_at,
        created_at=t.created_at,
        has_summary=bool(t.summary),
    )


# ── 대화방 CRUD ───────────────────────────────


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(30, ge=1, le=100),
    before: Optional[datetime] = Query(None, description="이 시각보다 오래된 thread 만 (커서)"),
) -> ThreadListResponse:
    """현재 사용자 + 현재 조직의 활성 대화방 목록 (last_message_at desc, 커서 페이지네이션)."""
    user, _member, org = org_tuple
    q = (
        select(AiChatThread)
        .where(AiChatThread.user_id == user.id)
        .where(AiChatThread.organization_id == org.id)
        .where(AiChatThread.archived_at.is_(None))
        .order_by(desc(AiChatThread.last_message_at))
        .limit(limit + 1)
    )
    if before is not None:
        q = q.where(AiChatThread.last_message_at < before)
    rows = list((await db.execute(q)).scalars().all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    return ThreadListResponse(
        threads=[_to_thread_response(t) for t in rows],
        has_more=has_more,
    )


@router.post("/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(
    payload: ThreadCreate,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
) -> ThreadResponse:
    """새 대화방 생성."""
    user, _member, org = org_tuple
    thread = AiChatThread(
        user_id=user.id,
        organization_id=org.id,
        title=(payload.title or None),
    )
    db.add(thread)
    await db.flush()
    return _to_thread_response(thread)


@router.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: UUID,
    payload: ThreadUpdate,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
) -> ThreadResponse:
    """대화방 제목 수정."""
    user, _member, org = org_tuple
    thread = await _load_thread_or_404(thread_id, user.id, org.id, db)
    thread.title = payload.title
    await db.flush()
    return _to_thread_response(thread)


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
    hard: bool = Query(False, description="true 시 영구 삭제, 기본은 soft delete"),
):
    """대화방 삭제. 기본은 soft delete (archived_at), hard=true 면 완전 삭제."""
    user, _member, org = org_tuple
    thread = await _load_thread_or_404(thread_id, user.id, org.id, db, include_archived=True)
    if hard:
        await db.delete(thread)
    else:
        thread.archived_at = datetime.now(timezone.utc)
    await db.flush()


# ── 메시지 ────────────────────────────────────


@router.get("/threads/{thread_id}/messages", response_model=MessageHistoryResponse)
async def list_messages(
    thread_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    before: Optional[datetime] = Query(None, description="이 시각보다 오래된 메시지만"),
) -> MessageHistoryResponse:
    """메시지 히스토리 (시간 오름차순, 커서 페이지네이션). system 메시지는 노출 제외."""
    user, _member, org = org_tuple
    await _load_thread_or_404(thread_id, user.id, org.id, db, include_archived=True)

    q = (
        select(AiChatMessage)
        .where(AiChatMessage.thread_id == thread_id)
        .where(AiChatMessage.role.in_(("user", "assistant")))
        .order_by(desc(AiChatMessage.created_at))
        .limit(limit + 1)
    )
    if before is not None:
        q = q.where(AiChatMessage.created_at < before)
    rows = list((await db.execute(q)).scalars().all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    rows.reverse()  # 화면 표시는 오름차순
    return MessageHistoryResponse(
        messages=[MessageResponse.model_validate(m) for m in rows],
        has_more=has_more,
    )


@router.post("/threads/{thread_id}/messages")
async def post_message_stream(
    thread_id: UUID,
    payload: MessageCreate,
    request: Request,
    background: BackgroundTasks,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """메시지 전송 → SSE 스트리밍 응답.

    응답 형식 (text/event-stream):
        data: {"delta": "...텍스트..."}\\n\\n
        data: {"delta": "..."}\\n\\n
        data: {"done": true, "message_id": "<uuid>"}\\n\\n

    오류 시:
        data: {"error": "..."}\\n\\n
    """
    user, _member, org = org_tuple
    await _check_user_message_rate(user.id)
    thread = await _load_thread_or_404(thread_id, user.id, org.id, db)

    # 요약 트리거 사전 체크 (이번 턴 INSERT 전 기준)
    should_summarize = await openai_chat_service.maybe_schedule_summarization(
        thread_id=thread.id, db=db,
    )
    if should_summarize:
        background.add_task(openai_chat_service.run_summarization, thread.id)

    async def stream():
        async for chunk in openai_chat_service.astream(
            thread=thread,
            user_id=user.id,
            org_id=org.id,
            user_text=payload.content,
            db=db,
            is_disconnected=request.is_disconnected,
            background_tasks=background,
        ):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # nginx/proxy 버퍼링 비활성
            "Connection": "keep-alive",
        },
    )
