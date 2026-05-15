# =============================================
# app/models/__init__.py
# 역할: models 패키지 초기화 파일
#       - 여기서 모델을 임포트하면 Base.metadata에 자동 등록됨
#       - 임포트 순서: 참조 당하는 쪽 → 참조 하는 쪽
#       - 새 모델 추가 시 해당 영역 주석 아래에 추가
# =============================================

# ── 사용자 / 인증 ────────────────────────────
from app.models.user import User  # noqa: F401
from app.models.business_profile import BusinessProfile  # noqa: F401
from app.models.term import Term  # noqa: F401
from app.models.user_term_agreement import UserTermAgreement  # noqa: F401

# ── 하자 탐지 ────────────────────────────────
from app.models.defect import DefectLog  # noqa: F401

# ── 드론 / 센서 ──────────────────────────────
from app.models.telemetry import TelemetryLog  # noqa: F401
from app.models.slam_map import SlamMap  # noqa: F401

# ── 평면도 / 환경 생성 ───────────────────────
from app.models.floorplan import Floorplan  # noqa: F401

# ── 현장 관리 ────────────────────────────────
from app.models.site import Site  # noqa: F401
from app.models.inspection_schedule import InspectionSchedule  # noqa: F401

# ── 보고서 ───────────────────────────────────
from app.models.report import Report  # noqa: F401

# ── 알림 ─────────────────────────────────────
from app.models.notification import Notification  # noqa: F401
from app.models.device_token import DeviceToken  # noqa: F401

# ── 조직 / 팀 ────────────────────────────────
from app.models.organization import Organization, OrganizationMember  # noqa: F401
from app.models.department import Department  # noqa: F401

# ── 메신저 / 채팅 ───────────────────────────
from app.models.conversation import Conversation  # noqa: F401
from app.models.message import Message  # noqa: F401
from app.models.conversation_member import ConversationMember  # noqa: F401

# ── AI 챗봇 (OpenAI 도메인 어시스턴트) ──────
from app.models.ai_chat import AiChatThread, AiChatMessage  # noqa: F401

__all__ = [
    # 사용자 / 인증
    "User",
    "BusinessProfile",
    "Term",
    "UserTermAgreement",
    # 하자 탐지
    "DefectLog",
    # 드론 / 센서
    "TelemetryLog",
    "SlamMap",
    # 평면도
    "Floorplan",
    # 현장 관리
    "Site",
    "InspectionSchedule",
    # 보고서
    "Report",
    # 알림
    "Notification",
    "DeviceToken",
    # 조직 / 팀
    "Organization",
    "OrganizationMember",
    "Department",
    # 메신저 / 채팅
    "Conversation",
    "Message",
    "ConversationMember",
    # AI 챗봇
    "AiChatThread",
    "AiChatMessage",
]
