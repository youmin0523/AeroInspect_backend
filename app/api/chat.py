# =============================================
# app/api/chat.py
# 역할: 사내 메신저 REST API 엔드포인트
#       - GET    /chat/conversations                   → 대화방 목록
#       - POST   /chat/conversations                   → 새 대화방 생성
#       - GET    /chat/conversations/{id}/messages      → 메시지 목록
#       - POST   /chat/conversations/{id}/messages      → 메시지 전송 + WS broadcast
#       - PATCH  /chat/conversations/{id}/read          → 읽음 처리
#       - GET    /chat/unread-counts                    → 미읽음 카운트
# =============================================

import os
import uuid as uuid_mod
from pathlib import Path
from typing import Optional
from uuid import UUID

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, func, desc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user, get_current_org_member, get_ws_manager

CHAT_UPLOAD_DIR = "./uploads/chat"
MAX_CHAT_FILE_SIZE = 200 * 1024 * 1024  # 200MB
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.conversation_member import ConversationMember
from app.models.user import User
from app.schemas.chat import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    MemberBrief,
    LastMessageBrief,
    UnreadCountResponse,
)
from app.core.ws_manager import ConnectionManager

router = APIRouter()


# ── 헬퍼: 대화방 응답 빌드 ──────────────────
async def _build_conv_response(db: AsyncSession, conv: Conversation) -> ConversationResponse:
    # 참여자 조회
    members_q = await db.execute(
        select(ConversationMember.user_id, User.name)
        .join(User, User.id == ConversationMember.user_id)
        .where(ConversationMember.conversation_id == conv.id)
    )
    participants = [
        MemberBrief(user_id=row[0], name=row[1], initials=row[1][:2].upper() if row[1] else "??")
        for row in members_q
    ]

    # 마지막 메시지
    last_msg_q = await db.execute(
        select(Message, User.name)
        .join(User, User.id == Message.sender_id)
        .where(Message.conversation_id == conv.id)
        .order_by(desc(Message.created_at))
        .limit(1)
    )
    last_row = last_msg_q.first()
    last_message = None
    if last_row:
        last_message = LastMessageBrief(
            text=last_row[0].text,
            file_name=last_row[0].file_name,
            sender_name=last_row[1],
            created_at=last_row[0].created_at,
        )

    return ConversationResponse(
        id=conv.id,
        type=conv.type,
        name=conv.name,
        participants=participants,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        last_message=last_message,
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 대화방 목록 조회 (소속 조직 한정) — 배치 쿼리 최적화"""
    user, member, org = org_tuple

    # 1) 내가 참여한 대화방 ID 목록
    conv_ids_q = await db.execute(
        select(ConversationMember.conversation_id)
        .where(ConversationMember.user_id == user.id)
    )
    conv_ids = [row[0] for row in conv_ids_q]
    if not conv_ids:
        return ConversationListResponse(items=[], total=0)

    # 2) 대화방 정보 조회 (1회 쿼리)
    convs_q = await db.execute(
        select(Conversation)
        .where(Conversation.id.in_(conv_ids))
        .where(Conversation.organization_id == org.id)
        .order_by(desc(Conversation.updated_at))
    )
    convs = convs_q.scalars().all()
    if not convs:
        return ConversationListResponse(items=[], total=0)

    active_ids = [c.id for c in convs]

    # 3) 전체 참여자 배치 조회 (1회 쿼리)
    all_members_q = await db.execute(
        select(ConversationMember.conversation_id, ConversationMember.user_id, User.name, User.profile_image_url)
        .join(User, User.id == ConversationMember.user_id)
        .where(ConversationMember.conversation_id.in_(active_ids))
    )
    participants_map = {}
    for conv_id, uid, uname, profile_img in all_members_q:
        participants_map.setdefault(conv_id, []).append(
            MemberBrief(user_id=uid, name=uname, initials=uname[:2].upper() if uname else "??")
        )

    # 4) 각 대화방의 마지막 메시지 배치 조회 (1회 쿼리 — LATERAL 대용)
    from sqlalchemy.orm import aliased
    sub = (
        select(Message.conversation_id, Message.text, Message.file_name, Message.created_at, User.name.label("sender_name"))
        .join(User, User.id == Message.sender_id)
        .where(Message.conversation_id.in_(active_ids))
        .order_by(Message.conversation_id, desc(Message.created_at))
    )
    last_msgs_q = await db.execute(sub)
    last_msg_map = {}
    for row in last_msgs_q:
        cid = row[0]
        if cid not in last_msg_map:  # 첫 번째 = 최신 메시지
            last_msg_map[cid] = LastMessageBrief(
                text=row[1], file_name=row[2], sender_name=row[4], created_at=row[3],
            )

    # 5) 응답 조립
    items = [
        ConversationResponse(
            id=c.id, type=c.type, name=c.name,
            participants=participants_map.get(c.id, []),
            created_at=c.created_at, updated_at=c.updated_at,
            last_message=last_msg_map.get(c.id),
        )
        for c in convs
    ]
    return ConversationListResponse(items=items, total=len(items))


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    payload: ConversationCreate,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """새 대화방 생성 (소속 조직에 자동 배정, 참여자 같은 조직 검증)"""
    user, member, org = org_tuple
    from app.models.organization import OrganizationMember
    # 참여자가 같은 조직인지 검증
    for pid in payload.participant_ids:
        if pid == user.id:
            continue
        exists = await db.scalar(
            select(OrganizationMember.id).where(
                OrganizationMember.organization_id == org.id,
                OrganizationMember.user_id == pid,
                OrganizationMember.status == "active",
            )
        )
        if not exists:
            raise HTTPException(status_code=400, detail=f"참여자 {pid}는 같은 조직에 소속되어 있지 않습니다.")

    # DM 중복 방지: 같은 두 사용자 간 기존 DM이 있으면 그것을 반환
    all_ids = set(payload.participant_ids) | {user.id}
    if payload.type == "dm" and len(all_ids) == 2:
        from sqlalchemy.orm import aliased
        CM1 = aliased(ConversationMember)
        CM2 = aliased(ConversationMember)
        id_list = list(all_ids)
        existing_q = await db.execute(
            select(Conversation)
            .join(CM1, CM1.conversation_id == Conversation.id)
            .join(CM2, CM2.conversation_id == Conversation.id)
            .where(Conversation.type == "dm")
            .where(Conversation.organization_id == org.id)
            .where(CM1.user_id == id_list[0])
            .where(CM2.user_id == id_list[1])
            .limit(1)
        )
        existing_dm = existing_q.scalar_one_or_none()
        if existing_dm:
            return await _build_conv_response(db, existing_dm)

    conv = Conversation(
        type=payload.type,
        name=payload.name,
        created_by=user.id,
        organization_id=org.id,
    )
    db.add(conv)
    await db.flush()

    # 참여자 추가 (생성자 포함)
    for uid in all_ids:
        db.add(ConversationMember(conversation_id=conv.id, user_id=uid))
    await db.flush()

    return await _build_conv_response(db, conv)


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """대화방 메시지 목록 조회"""
    # 참여자 확인
    member = await db.scalar(
        select(ConversationMember.id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    if member is None:
        raise HTTPException(status_code=403, detail="이 대화방에 참여하고 있지 않습니다.")

    total = await db.scalar(
        select(func.count(Message.id))
        .where(Message.conversation_id == conversation_id)
    )

    msgs_q = await db.execute(
        select(Message, User.name, User.profile_image_url)
        .join(User, User.id == Message.sender_id)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .limit(limit)
        .offset(offset)
    )

    # 읽음 상태 계산: 다른 멤버들의 last_read_at 조회
    other_members_q = await db.execute(
        select(ConversationMember.last_read_at)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id != current_user.id)
    )
    read_times = [r[0] for r in other_members_q if r[0] is not None]

    items = []
    for msg, sender_name, profile_img in msgs_q:
        read_count = sum(1 for t in read_times if t >= msg.created_at) if msg.sender_id == current_user.id else 0
        items.append(MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            sender_id=msg.sender_id,
            sender_name=sender_name,
            sender_initials=sender_name[:2].upper() if sender_name else "??",
            sender_profile_image_url=profile_img,
            text=msg.text,
            file_url=msg.file_url,
            file_name=msg.file_name,
            file_content_type=msg.file_content_type,
            read_by_count=read_count,
            created_at=msg.created_at,
        ))

    return MessageListResponse(items=items, total=total, has_more=(offset + limit) < total)


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def send_message(
    conversation_id: UUID,
    payload: MessageCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_ws_manager),
):
    """메시지 전송 + WebSocket 브로드캐스트"""
    # 참여자 확인
    member = await db.scalar(
        select(ConversationMember.id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    if member is None:
        raise HTTPException(status_code=403, detail="이 대화방에 참여하고 있지 않습니다.")

    msg = Message(
        conversation_id=conversation_id,
        sender_id=current_user.id,
        text=payload.text,
    )
    db.add(msg)

    # 대화방 updated_at 갱신
    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = func.now()

    await db.flush()

    response = MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        sender_id=msg.sender_id,
        sender_name=current_user.name,
        sender_initials=current_user.name[:2].upper() if current_user.name else "??",
        sender_profile_image_url=current_user.profile_image_url,
        text=msg.text,
        created_at=msg.created_at,
    )

    # WebSocket 브로드캐스트 — 대화방 채널 + 각 참여자 개인 채널
    msg_payload = {
        "type": "chat.new_message",
        "data": response.model_dump(mode="json"),
    }
    await ws_manager.broadcast(f"chat:{conversation_id}", msg_payload)

    # 참여자별 개인 채널에도 전송 (다른 대화방에 있거나 채팅 페이지 밖일 때도 알림)
    participants = await db.execute(
        select(ConversationMember.user_id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id != current_user.id)
    )
    for (uid,) in participants:
        await ws_manager.broadcast(f"user:{uid}", msg_payload)

    return response


@router.post("/conversations/{conversation_id}/messages/file", response_model=MessageResponse, status_code=201)
async def send_file_message(
    conversation_id: UUID,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_ws_manager),
):
    """첨부파일 메시지 전송 (multipart/form-data, 최대 200MB)"""
    # 참여자 확인
    member = await db.scalar(
        select(ConversationMember.id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    if member is None:
        raise HTTPException(status_code=403, detail="이 대화방에 참여하고 있지 않습니다.")

    # 파일 크기 확인 (200MB)
    contents = await file.read()
    if len(contents) > MAX_CHAT_FILE_SIZE:
        raise HTTPException(status_code=413, detail="파일 크기는 200MB 이하여야 합니다.")

    # 파일 저장
    ext = os.path.splitext(file.filename or "")[1] or ".bin"
    saved_name = f"{uuid_mod.uuid4()}{ext}"
    save_path = os.path.join(CHAT_UPLOAD_DIR, saved_name)
    os.makedirs(CHAT_UPLOAD_DIR, exist_ok=True)

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    file_url = f"/uploads/chat/{saved_name}"

    msg = Message(
        conversation_id=conversation_id,
        sender_id=current_user.id,
        text=text.strip() if text and text.strip() else None,
        file_url=file_url,
        file_name=file.filename,
        file_content_type=file.content_type,
    )
    db.add(msg)

    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = func.now()

    await db.flush()

    response = MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        sender_id=msg.sender_id,
        sender_name=current_user.name,
        sender_initials=current_user.name[:2].upper() if current_user.name else "??",
        sender_profile_image_url=current_user.profile_image_url,
        text=msg.text,
        file_url=msg.file_url,
        file_name=msg.file_name,
        file_content_type=msg.file_content_type,
        created_at=msg.created_at,
    )

    msg_payload = {
        "type": "chat.new_message",
        "data": response.model_dump(mode="json"),
    }
    await ws_manager.broadcast(f"chat:{conversation_id}", msg_payload)

    participants = await db.execute(
        select(ConversationMember.user_id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id != current_user.id)
    )
    for (uid,) in participants:
        await ws_manager.broadcast(f"user:{uid}", msg_payload)

    return response


@router.get("/messages/{message_id}/download")
async def download_message_file(
    message_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """첨부파일 다운로드 — 원본 파일명을 Content-Disposition 으로 보존 (RFC 5987).

    프런트엔드는 이 엔드포인트로 fetch → blob → object URL → <a download> 클릭 패턴을 사용한다.
    StaticFiles 가 헤더를 못 붙이고, cross-origin 환경에서 <a download> 속성이 무시되는 문제를
    한꺼번에 우회한다.
    """
    msg = await db.get(Message, message_id)
    if msg is None or not msg.file_url:
        raise HTTPException(status_code=404, detail="첨부파일이 없습니다.")

    # 참여자 확인 (해당 대화방 멤버만 다운로드 가능)
    member = await db.scalar(
        select(ConversationMember.id)
        .where(ConversationMember.conversation_id == msg.conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    if member is None:
        raise HTTPException(status_code=403, detail="이 대화방에 참여하고 있지 않습니다.")

    # file_url 은 "/uploads/chat/uuid.ext" 형식. CHAT_UPLOAD_DIR 하위로만 허용 (path traversal 방어)
    file_url = msg.file_url or ""
    prefix = "/uploads/chat/"
    if not file_url.startswith(prefix):
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.")
    saved_name = file_url[len(prefix):]
    if "/" in saved_name or "\\" in saved_name or ".." in saved_name:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.")

    abs_path = os.path.abspath(os.path.join(CHAT_UPLOAD_DIR, saved_name))
    upload_root = os.path.abspath(CHAT_UPLOAD_DIR)
    if not abs_path.startswith(upload_root + os.sep) and abs_path != upload_root:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="파일이 디스크에 없습니다.")

    return FileResponse(
        path=abs_path,
        filename=msg.file_name or saved_name,
        media_type=msg.file_content_type or "application/octet-stream",
    )


@router.patch("/conversations/{conversation_id}/read")
async def mark_read(
    conversation_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_ws_manager),
):
    """대화방 읽음 처리 (last_read_at 갱신 + WS broadcast)"""
    result = await db.execute(
        select(ConversationMember)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    member = result.scalar_one_or_none()
    if member is None:
        raise HTTPException(status_code=403, detail="이 대화방에 참여하고 있지 않습니다.")

    member.last_read_at = func.now()
    await db.flush()

    # 읽음 이벤트 브로드캐스트 → 상대방 화면의 "읽음" 표시 실시간 갱신
    read_payload = {
        "type": "chat.read",
        "data": {"conversation_id": str(conversation_id), "user_id": str(current_user.id)},
    }
    await ws_manager.broadcast(f"chat:{conversation_id}", read_payload)
    # 참여자 개인 채널에도 전송
    others = await db.execute(
        select(ConversationMember.user_id)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id != current_user.id)
    )
    for (uid,) in others:
        await ws_manager.broadcast(f"user:{uid}", read_payload)

    return {"ok": True}


@router.delete("/conversations/{conversation_id}/leave", status_code=204)
async def leave_conversation(
    conversation_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """대화방 나가기 (참여자 레코드 삭제)"""
    result = await db.execute(
        select(ConversationMember)
        .where(ConversationMember.conversation_id == conversation_id)
        .where(ConversationMember.user_id == current_user.id)
    )
    member = result.scalar_one_or_none()
    if member is None:
        raise HTTPException(status_code=404, detail="이 대화방에 참여하고 있지 않습니다.")

    await db.delete(member)
    await db.flush()

    # 남은 참여자가 없으면 대화방 자체도 삭제
    remaining = await db.scalar(
        select(func.count(ConversationMember.id))
        .where(ConversationMember.conversation_id == conversation_id)
    )
    if remaining == 0:
        conv = await db.get(Conversation, conversation_id)
        if conv:
            await db.delete(conv)
            await db.flush()


@router.get("/unread-counts", response_model=UnreadCountResponse)
async def get_unread_counts(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """전체 미읽음 카운트 조회.

    단일 GROUP BY 쿼리로 모든 대화방의 미읽음 수를 한 번에 집계한다.
    (과거: 대화방 1개당 COUNT 1쿼리 = N+1. 50개 방이면 51왕복 → 1쿼리로 축소)
    """
    cm = ConversationMember
    q = (
        select(cm.conversation_id, func.count(Message.id))
        .join(
            Message,
            and_(
                Message.conversation_id == cm.conversation_id,
                Message.sender_id != current_user.id,
                or_(
                    cm.last_read_at.is_(None),
                    Message.created_at > cm.last_read_at,
                ),
            ),
        )
        .where(cm.user_id == current_user.id)
        .group_by(cm.conversation_id)
    )
    rows = (await db.execute(q)).all()

    per_conversation = {}
    total = 0
    for conv_id, count in rows:
        count = count or 0
        if count > 0:
            per_conversation[str(conv_id)] = count
            total += count

    return UnreadCountResponse(total=total, per_conversation=per_conversation)
