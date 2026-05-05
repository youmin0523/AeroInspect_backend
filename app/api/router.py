# =============================================
# app/api/router.py
# 역할: 모든 서브 라우터를 하나의 api_router로 통합
#       - main.py에서 app.include_router(api_router, prefix="/api/v1")로 마운트
#       - 각 도메인별 라우터에 태그와 prefix 부여
# =============================================

from fastapi import APIRouter

from app.api import auth, oauth, defects, stream, websocket, report, telemetry, slam, floorplan, ai_webhook, sites, notifications, chat, organization, detect, ws_stream, coverage, employee, admin_gpu
from app.schemas.common import PROTECTED_RESPONSES, PUBLIC_RESPONSES, WEBHOOK_RESPONSES

api_router = APIRouter()

# 인증 / 회원가입 — public (login 자체가 401 의미가 다름)
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Auth"],
    responses=PUBLIC_RESPONSES,
)

# OAuth 소셜 로그인 (Google / Kakao / Naver)
api_router.include_router(
    oauth.router,
    prefix="/oauth",
    tags=["OAuth"],
    responses=PUBLIC_RESPONSES,
)

# 하자 탐지 로그 CRUD
api_router.include_router(
    defects.router,
    prefix="/defects",
    tags=["Defects"],
    responses=PROTECTED_RESPONSES,
)

# 카메라 스트리밍 (RGB / 열화상 / 블렌드) — StreamingResponse 라 별도 응답 스키마 X
api_router.include_router(
    stream.router,
    prefix="/stream",
    tags=["Stream"],
)

# WebSocket 실시간 이벤트 — ws 핸드셰이크는 OpenAPI responses 의미 X
api_router.include_router(
    websocket.router,
    prefix="",
    tags=["WebSocket"],
)

# LLM 하자 점검 보고서 생성/저장/조회/다운로드
api_router.include_router(
    report.router,
    prefix="/report",
    tags=["Report"],
    responses=PROTECTED_RESPONSES,
)

# 드론 텔레메트리 (좌표/센서/배터리)
api_router.include_router(
    telemetry.router,
    prefix="/telemetry",
    tags=["Telemetry"],
    responses=PROTECTED_RESPONSES,
)

# SLAM 맵 데이터
api_router.include_router(
    slam.router,
    prefix="/slam",
    tags=["SLAM"],
    responses=PROTECTED_RESPONSES,
)

# 평면도 업로드 & 처리
api_router.include_router(
    floorplan.router,
    prefix="/floorplan",
    tags=["Floorplan"],
    responses=PROTECTED_RESPONSES,
)

# AI 서버 연동 웹훅 (탐지 이벤트 수신) — X-AI-Webhook-Secret 인증
api_router.include_router(
    ai_webhook.router,
    prefix="/ai",
    tags=["AI Webhook"],
    responses=WEBHOOK_RESPONSES,
)

# 현장(Site) 관리 CRUD
api_router.include_router(
    sites.router,
    prefix="/sites",
    tags=["Sites"],
    responses=PROTECTED_RESPONSES,
)

# 알림 관리 CRUD
api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["Notifications"],
    responses=PROTECTED_RESPONSES,
)

# 조직(회사) 관리 — 멤버 목록 / 초대 / 권한
api_router.include_router(
    organization.router,
    prefix="/organizations",
    tags=["Organizations"],
    responses=PROTECTED_RESPONSES,
)

# 메신저 / 채팅
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
    responses=PROTECTED_RESPONSES,
)

# 3-모델 하자 탐지 (multipart 업로드) — 인증 의존성에 따라 일부 보호
api_router.include_router(
    detect.router,
    prefix="/detect",
    tags=["Detect"],
)

# 실시간 스트림 수신 WebSocket (/ws/stream)
api_router.include_router(
    ws_stream.router,
    prefix="",
    tags=["WebSocket"],
)

# 현장별 점검 커버리지 산출 (텔레메트리 convex hull)
api_router.include_router(
    coverage.router,
    prefix="/coverage",
    tags=["Coverage"],
    responses=PROTECTED_RESPONSES,
)

# Employee 랜딩 통합 (오늘 일정 / 월간 KPI / 최근 활동)
api_router.include_router(
    employee.router,
    prefix="/employee",
    tags=["Employee"],
    responses=PROTECTED_RESPONSES,
)

# 관리자 — GCP GPU VM 원격 제어 (슈퍼어드민 전용)
api_router.include_router(
    admin_gpu.router,
    prefix="/admin/gpu",
    tags=["Admin"],
    responses=PROTECTED_RESPONSES,
)
