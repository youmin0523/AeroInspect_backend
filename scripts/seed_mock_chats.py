"""
seed_mock_chats.py
역할: UI 디자인 검증용 목업 채팅 데이터 생성
      - 1:1 DM: 텍스트 + 이미지 + 파일 + 이모지 포함 대화
      - 그룹 채팅: 다인원 프로젝트 협업 대화
      - 채널: 전체 공지사항 채널

실행: cd backend && python -m scripts.seed_mock_chats
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime, timedelta, timezone

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.db.session import async_session_factory
from app.models.user import User
from app.models.organization import Organization, OrganizationMember
from app.models.conversation import Conversation
from app.models.conversation_member import ConversationMember
from app.models.message import Message


def ts(minutes_ago: int) -> datetime:
    """현재 시각에서 N분 전 타임스탬프"""
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


async def main():
    async with async_session_factory() as db:
        # ── AICC8 조직 + 멤버 조회 ──────────────
        org_q = await db.execute(select(Organization).where(Organization.name == "AICC8"))
        org = org_q.scalar_one_or_none()
        if not org:
            print("❌ AICC8 조직이 없습니다.")
            return

        # AICC8 소속 멤버 조회
        members_q = await db.execute(
            select(OrganizationMember, User)
            .join(User, User.id == OrganizationMember.user_id)
            .where(OrganizationMember.organization_id == org.id)
            .where(OrganizationMember.status == "active")
            .order_by(OrganizationMember.role)  # owner 먼저
        )
        members_list = [(m, u) for m, u in members_q]
        if len(members_list) < 2:
            print("❌ AICC8 조직에 활성 멤버가 2명 이상이어야 합니다.")
            return

        user_a = members_list[0][1]  # 유민수 (소유자)
        user_b = members_list[1][1]  # 백승희 (멤버)

        print(f"👤 사용자 A: {user_a.name} ({user_a.email})")
        print(f"👤 사용자 B: {user_b.name} ({user_b.email})")
        print(f"🏢 조직: {org.name}")

        # ══════════════════════════════════════════
        # 1) 1:1 DM — 텍스트 + 이미지 + 파일 + 이모지
        # ══════════════════════════════════════════
        dm_conv = Conversation(
            type="dm",
            name=None,
            organization_id=org.id,
            created_by=user_a.id,
            updated_at=ts(0),
        )
        db.add(dm_conv)
        await db.flush()

        for u in [user_a, user_b]:
            db.add(ConversationMember(conversation_id=dm_conv.id, user_id=u.id, last_read_at=ts(0)))
        await db.flush()

        dm_messages = [
            # 텍스트 메시지
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="승희님, 오늘 A동 외벽 점검 드론 비행 완료했습니다 👍",
                    created_at=ts(120)),
            Message(conversation_id=dm_conv.id, sender_id=user_b.id,
                    text="수고하셨어요! 결과 데이터 공유해주실 수 있나요?",
                    created_at=ts(115)),

            # 이미지 첨부
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="네, 외벽 균열 탐지 결과입니다",
                    file_url="/uploads/chat/mock_crack_detection.jpg",
                    file_name="A동_외벽_균열탐지_결과.jpg",
                    file_content_type="image/jpeg",
                    created_at=ts(110)),

            Message(conversation_id=dm_conv.id, sender_id=user_b.id,
                    text="오 깔끔하네요 👏 3층 부분에 균열이 좀 심한 것 같은데 확대 사진 있나요?",
                    created_at=ts(105)),

            # 추가 이미지
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    file_url="/uploads/chat/mock_crack_closeup.jpg",
                    file_name="3층_균열_확대.jpg",
                    file_content_type="image/jpeg",
                    created_at=ts(100)),
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="3층 북측 외벽 0.3mm 균열입니다. 보수 필요해 보여요 🔍",
                    created_at=ts(99)),

            # 파일 첨부 (PDF)
            Message(conversation_id=dm_conv.id, sender_id=user_b.id,
                    text="감사합니다! 이전 점검 보고서랑 비교해볼게요. 지난달 보고서 첨부합니다",
                    file_url="/uploads/chat/mock_report_march.pdf",
                    file_name="2026년_3월_정기점검_보고서.pdf",
                    file_content_type="application/pdf",
                    created_at=ts(90)),

            # 이모지 많은 대화
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="확인했습니다 ✅ 3월 대비 균열 폭이 0.1mm 증가했네요 😟",
                    created_at=ts(85)),
            Message(conversation_id=dm_conv.id, sender_id=user_b.id,
                    text="그러면 긴급 보수 대상으로 올려야겠네요 ⚠️ 내일 현장 미팅 때 논의하시죠!",
                    created_at=ts(80)),
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="넵! 내일 오전 10시에 현장에서 뵙겠습니다 🙋‍♂️",
                    created_at=ts(75)),

            # 엑셀 파일
            Message(conversation_id=dm_conv.id, sender_id=user_b.id,
                    text="미팅 전에 이 데이터 한번 봐주세요",
                    file_url="/uploads/chat/mock_defect_data.xlsx",
                    file_name="하자_데이터_분석_v2.xlsx",
                    file_content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    created_at=ts(60)),

            # 최근 메시지
            Message(conversation_id=dm_conv.id, sender_id=user_a.id,
                    text="확인 완료! 내일 뵙겠습니다 😊",
                    created_at=ts(30)),
        ]
        for m in dm_messages:
            db.add(m)
        await db.flush()

        print(f"✅ 1:1 DM 생성 완료 (메시지 {len(dm_messages)}개)")

        # ══════════════════════════════════════════
        # 2) 그룹 채팅 — 프로젝트 협업
        # ══════════════════════════════════════════
        group_conv = Conversation(
            type="group",
            name="A동 외벽 점검 프로젝트",
            organization_id=org.id,
            created_by=user_a.id,
            updated_at=ts(0),
        )
        db.add(group_conv)
        await db.flush()

        for u in [user_a, user_b]:
            db.add(ConversationMember(conversation_id=group_conv.id, user_id=u.id, last_read_at=ts(5)))
        await db.flush()

        group_messages = [
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    text="A동 외벽 점검 프로젝트 그룹입니다. 일정 및 진행상황을 여기서 공유합니다 📋",
                    created_at=ts(300)),
            Message(conversation_id=group_conv.id, sender_id=user_b.id,
                    text="네 알겠습니다! 드론 비행 일정 공유드립니다",
                    file_url="/uploads/chat/mock_flight_schedule.pdf",
                    file_name="드론_비행_일정표_4월.pdf",
                    file_content_type="application/pdf",
                    created_at=ts(290)),
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    text="1차 비행 완료했습니다. 결과 이미지 공유드립니다",
                    created_at=ts(200)),
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    file_url="/uploads/chat/mock_drone_overview.jpg",
                    file_name="A동_전경_드론촬영.jpg",
                    file_content_type="image/jpeg",
                    created_at=ts(199)),
            Message(conversation_id=group_conv.id, sender_id=user_b.id,
                    text="잘 나왔네요! 북측 외벽 쪽 클로즈업도 부탁드려요 🙏",
                    created_at=ts(190)),
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    text="2차 비행에서 촬영하겠습니다. 열화상 이미지도 같이 올릴게요 🔥",
                    created_at=ts(180)),
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    file_url="/uploads/chat/mock_thermal_image.jpg",
                    file_name="북측_열화상_분석.jpg",
                    file_content_type="image/jpeg",
                    created_at=ts(100)),
            Message(conversation_id=group_conv.id, sender_id=user_b.id,
                    text="열화상에서 단열 취약부위가 보이네요. 표시해서 다시 올려주실 수 있나요?",
                    created_at=ts(95)),
            Message(conversation_id=group_conv.id, sender_id=user_a.id,
                    text="마크업 완료했습니다 ✅",
                    file_url="/uploads/chat/mock_thermal_marked.jpg",
                    file_name="북측_열화상_마크업.jpg",
                    file_content_type="image/jpeg",
                    created_at=ts(50)),
            Message(conversation_id=group_conv.id, sender_id=user_b.id,
                    text="완벽합니다 👍 이걸로 보고서 작성 들어갈게요. 금요일까지 초안 공유드리겠습니다!",
                    created_at=ts(40)),
        ]
        for m in group_messages:
            db.add(m)
        await db.flush()

        print(f"✅ 그룹 채팅 생성 완료 (메시지 {len(group_messages)}개)")

        # ══════════════════════════════════════════
        # 3) 채널 — 전체 공지
        # ══════════════════════════════════════════
        channel_conv = Conversation(
            type="channel",
            name="전체 공지",
            organization_id=org.id,
            created_by=user_a.id,
            updated_at=ts(0),
        )
        db.add(channel_conv)
        await db.flush()

        for u in [user_a, user_b]:
            db.add(ConversationMember(conversation_id=channel_conv.id, user_id=u.id, last_read_at=ts(500)))
        await db.flush()

        channel_messages = [
            Message(conversation_id=channel_conv.id, sender_id=user_a.id,
                    text="📢 [공지] 드론 점검 플랫폼 v2.0 업데이트 안내\n\n안녕하세요, AICC8 팀 여러분.\n\n다음 기능이 추가되었습니다:\n• 채팅 첨부파일 전송 (이미지/문서, 최대 200MB)\n• 이모지 피커 🎉\n• 실시간 읽음 표시\n• 초대코드 30일 자동 만료\n\n문의사항은 DM으로 부탁드립니다.",
                    created_at=ts(1440)),

            Message(conversation_id=channel_conv.id, sender_id=user_a.id,
                    text="📢 [공지] 4월 정기 안전 점검 일정\n\n• 4/28(월) A동 외벽 — 담당: 유민수\n• 4/29(화) B동 옥상 — 담당: 백승희\n• 4/30(수) 주차장 구조물 — 담당: TBD\n\n비행 전 반드시 기상 확인 부탁드립니다 🌤️",
                    created_at=ts(720)),

            Message(conversation_id=channel_conv.id, sender_id=user_a.id,
                    text="📋 안전 점검 체크리스트 업데이트했습니다. 반드시 숙지 부탁드립니다.",
                    file_url="/uploads/chat/mock_safety_checklist.pdf",
                    file_name="2026_4월_안전점검_체크리스트_v3.pdf",
                    file_content_type="application/pdf",
                    created_at=ts(360)),

            Message(conversation_id=channel_conv.id, sender_id=user_b.id,
                    text="확인했습니다! 👍",
                    created_at=ts(350)),
        ]
        for m in channel_messages:
            db.add(m)
        await db.flush()

        print(f"✅ 채널 생성 완료 (메시지 {len(channel_messages)}개)")

        await db.commit()
        print(f"\n🎉 목업 데이터 생성 완료!")
        print(f"   - 1:1 DM: {len(dm_messages)}개 메시지 (텍스트+이미지+파일+이모지)")
        print(f"   - 그룹 채팅: {len(group_messages)}개 메시지 (프로젝트 협업)")
        print(f"   - 채널: {len(channel_messages)}개 메시지 (전체 공지)")


if __name__ == "__main__":
    asyncio.run(main())
