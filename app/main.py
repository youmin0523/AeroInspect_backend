# =============================================
# app/main.py
# 역할: FastAPI 애플리케이션 진입점
#       - lifespan 컨텍스트 매니저로 시작/종료 시 리소스 초기화/해제
#       - CORS 미들웨어 설정
#       - 모든 API 라우터 마운트
#       - 서비스 싱글톤(카메라, YOLO, WebSocket 매니저) 초기화
# 실행: uvicorn app.main:app --reload --port 8000
# =============================================

# Windows cmd(cp949)에서 한글/이모지 print 시 UnicodeEncodeError → 500 에러 방지.
# stdout/stderr를 UTF-8로 강제. 인코딩 불가 문자는 ?로 대체(errors='replace').
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import PrometheusMiddleware, render_metrics
from app.core.middleware import RequestIDMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.core.sentry import init_sentry
from app.db.init_db import init_db
from app.api.router import api_router
from app.services.camera import rgb_camera_service, thermal_camera_service
from app.services.recording import recording_service
from app.services.yolo_inference import yolo_service
from app.services.inference_pipeline import pipeline as inference_pipeline
from app.services.wallpaper_classifier import wallpaper_classifier
from app.services.lidar import lidar_service
from app.services.telemetry_cache import telemetry_cache
from app.core.stream_inference import stream_inference_worker
from app.services.defect_taxonomy import WALLPAPER_CLASSES

# 로깅 설정 (앱 import 시점에 1회)
configure_logging(
    json_output=settings.LOG_JSON,
    level=settings.LOG_LEVEL,
)
logger = get_logger(__name__)


async def _ensure_superadmin_seed():
    """슈퍼어드민 시드 계정이 없으면 자동 생성 (admin / admin)."""
    from sqlalchemy import select
    from app.db.session import async_session_factory
    from app.models.user import User
    from app.core.security import hash_password

    async with async_session_factory() as db:
        existing = await db.scalar(
            select(User.id).where(User.username == "admin")
        )
        if existing:
            logger.info("superadmin_seed_exists", username="admin")
            return

        superadmin = User(
            username="admin",
            email="admin@aeroinspect.io",
            password_hash=hash_password("admin"),
            name="슈퍼관리자",
            phone="000-0000-0000",
            account_type="personal",
            is_superadmin=True,
        )
        db.add(superadmin)
        await db.commit()
        logger.info("superadmin_seed_created", username="admin")
        print("[AeroInspect] 슈퍼어드민 시드 계정 생성 완료 (admin / admin)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 시 실행되는 lifespan 핸들러.
    순서: DB 초기화 → 카메라 오픈 → YOLO 모델 로드
    """
    # ── 시작 ─────────────────────────────────
    print("[AeroInspect] 서버 시작 중...")

    # Sentry 초기화 (DSN 미설정 시 no-op — 로컬 개발 차단 X)
    # 가장 먼저 호출하여 이후 모든 startup 오류도 Sentry 로 캡처되도록.
    try:
        sentry_enabled = init_sentry(settings)
        if sentry_enabled:
            print(f"[AeroInspect] Sentry 활성화 (env={settings.SENTRY_ENVIRONMENT})")
    except Exception as e:
        print(f"[AeroInspect] Sentry 초기화 실패 (계속 진행): {e}")

    # WebSocket 백엔드: Redis 모드면 매니저 교체 + pub/sub 구독 시작
    if settings.WS_BACKEND.lower() == "redis":
        try:
            from app.core import ws_manager as wsmod
            from app.core.ws_manager_redis import create_ws_manager

            redis_mgr = create_ws_manager(
                backend="redis",
                redis_url=settings.REDIS_URL,
            )
            await redis_mgr.start()
            # 모듈 어트리뷰트 재바인딩 금지 — 핫패스가 캡처한 프록시가 못 따라온다.
            # 반드시 set_active_manager 로 활성 매니저를 교체한다.
            wsmod.set_active_manager(redis_mgr)
            print(f"[AeroInspect] WS 백엔드: Redis ({settings.REDIS_URL})")
        except Exception as e:
            print(f"[AeroInspect] Redis WS 백엔드 시작 실패 — 메모리 백엔드로 폴백: {e}")
    else:
        print("[AeroInspect] WS 백엔드: memory (단일 워커 한정)")

    # DB 테이블 생성 (처음 실행 시)
    try:
        await init_db()
        print("[AeroInspect] DB 초기화 완료")

        # 슈퍼어드민 시드 계정 생성 (없으면 자동 생성)
        await _ensure_superadmin_seed()
    except Exception as e:
        print(f"[AeroInspect] DB 초기화 실패 (DB 연결 안 됨, 임시 무시): {e}")

    # Pipeline20 자동 로드 (USE_20DEFECT_PIPELINE=true 시 — GCP GPU VM 등 정밀 추론 서버용)
    if settings.USE_20DEFECT_PIPELINE:
        try:
            from app.services.inference_pipeline_20 import pipeline20 as _p20
            _p20.load_models()
            print("[AeroInspect] Pipeline20 로드 완료")
        except Exception as _e:
            print(f"[AeroInspect] Pipeline20 로드 실패: {_e}")

    if settings.DRONE_CONNECTED:
        # RGB 카메라 (USB Capture Card) 열기
        await rgb_camera_service.open()
        print(f"[AeroInspect] RGB 카메라 (index={settings.RGB_CAMERA_INDEX}) 열림")

        # 열화상 카메라 (IRC-256CA) 열기
        await thermal_camera_service.open()
        print(f"[AeroInspect] 열화상 카메라 (index={settings.THERMAL_CAMERA_INDEX}) 열림")

        # 3-모델 추론 파이프라인 로드 (YOLO thermal + delam + ResNet 벽지)
        try:
            yolo_service.load_model()
            print("[AeroInspect] 3-모델 추론 파이프라인 로드 완료")
        except FileNotFoundError as e:
            print(f"[AeroInspect] AI 모델 로드 실패 (가중치 없음): {e}")

        # TF-Luna LiDAR 시리얼 연결
        try:
            await lidar_service.start()
        except Exception as e:
            print(f"[AeroInspect] LiDAR 시작 실패 (좌표 없이 계속): {e}")

        # WebSocket 스트림 추론 워커 시작 (드롭 큐)
        await stream_inference_worker.start()
    else:
        print("[AeroInspect] DRONE_CONNECTED=False → 카메라/LiDAR/추론 파이프라인 건너뜀 (API 전용 모드)")

    print("[AeroInspect] 서버 준비 완료")

    yield  # 앱 실행 중

    # ── 종료 ─────────────────────────────────
    print("[AeroInspect] 서버 종료 중...")

    if settings.DRONE_CONNECTED:
        try:
            await lidar_service.stop()
        except Exception as e:
            print(f"[AeroInspect] LiDAR 종료 중 오류: {e}")

        await stream_inference_worker.stop()

        if recording_service.is_recording:
            await recording_service.stop()
            print("[AeroInspect] 녹화 중지 완료")

        await rgb_camera_service.release()
        await thermal_camera_service.release()
        print("[AeroInspect] 카메라 자원 해제 완료")

    telemetry_cache.clear()

    # GCP Compute 클라이언트 정리 — 공유 httpx 커넥션 풀 종료
    try:
        from app.services.gcp_compute import gcp_compute
        await gcp_compute.aclose()
    except Exception as e:
        print(f"[AeroInspect] GCP Compute 클라이언트 종료 중 오류: {e}")

    # 진행 중 녹화 정리 — VideoWriter 미해제로 mp4 가 잘리는 것 방지
    try:
        from app.services.recording import release_all as release_all_recordings
        await release_all_recordings()
    except Exception as e:
        print(f"[AeroInspect] 녹화 자원 정리 중 오류: {e}")

    # 공유 Redis 클라이언트 정리 (레이트리밋/토큰 폐기/스트림모드)
    try:
        from app.core.redis_client import close_redis
        await close_redis()
    except Exception as e:
        print(f"[AeroInspect] Redis 클라이언트 종료 중 오류: {e}")

    # Redis WS 백엔드 정리
    if settings.WS_BACKEND.lower() == "redis":
        try:
            from app.core import ws_manager as wsmod
            from app.core.ws_manager_redis import RedisConnectionManager
            # 주의: wsmod.ws_manager 는 프록시이므로 get_active_manager() 로 실제 매니저를 얻는다.
            mgr = wsmod.get_active_manager()
            if isinstance(mgr, RedisConnectionManager):
                await mgr.stop()
        except Exception as e:
            print(f"[AeroInspect] Redis WS 백엔드 종료 중 오류: {e}")


# ── OpenAPI 메타데이터 ───────────────────────
# Swagger UI 좌측 상단/태그 그룹 헤더에 노출되는 설명 + 외부 문서 링크.
tags_metadata = [
    {"name": "Health", "description": "서버/모델/카메라/스트림 워커 헬스체크. 운영 알람 훅 대상."},
    {"name": "Employee", "description": "직원 랜딩(/employee) 통합 데이터 — 오늘 일정·월간 KPI·최근 활동. 조직 단위 격리."},
    {"name": "Auth", "description": "회원가입·로그인·JWT 발급/갱신·내 정보 관리."},
    {"name": "OAuth", "description": "Google / Kakao / Naver 소셜 로그인 콜백."},
    {"name": "Organizations", "description": "조직(회사) 멤버 / 초대 / 역할 관리. X-Organization-Id 헤더로 다중 조직 분기."},
    {"name": "Sites", "description": "현장(점검 대상) CRUD. 평면도·텔레메트리·결함 로그가 사이트 단위로 묶임."},
    {"name": "Floorplan", "description": "평면도 업로드/처리. 드론 비행 좌표를 평면도 좌표계로 매핑."},
    {"name": "Telemetry", "description": "드론 좌표/배터리/IMU/LiDAR 등 실시간 텔레메트리. 캐시 + WebSocket 브로드캐스트."},
    {"name": "Coverage", "description": "드론 비행 텔레메트리 convex hull로 점검 커버리지·미점검 구역 산출."},
    {"name": "SLAM", "description": "SLAM 맵(point cloud / occupancy grid) 저장·조회."},
    {"name": "Defects", "description": "하자 탐지 로그 CRUD. 보고서·통계·심각도 매핑의 원천 데이터."},
    {"name": "Detect", "description": "3-모델 파이프라인(YOLO + ResNet) multipart 업로드 추론 엔드포인트."},
    {"name": "Stream", "description": "MJPEG/HLS 카메라 스트림 (RGB / 열화상 / 블렌드)."},
    {"name": "WebSocket", "description": "실시간 이벤트(추론 결과 / 텔레메트리 / 알림). HTTP가 아니라 ws:// 핸드셰이크 — Swagger에선 테스트 불가, 별도 클라이언트 필요."},
    {"name": "Report", "description": "LLM 기반 점검 보고서 생성·저장·조회·다운로드."},
    {"name": "Notifications", "description": "사용자 알림 CRUD + 푸시(FCM/APNS) 발송."},
    {"name": "Chat", "description": "조직 내 메신저(채널/DM) + 파일 첨부."},
    {"name": "AI Webhook", "description": "외부 AI 추론 서버 → 백엔드 콜백. X-AI-Webhook-Secret 헤더 인증 필수."},
    {"name": "Admin", "description": "플랫폼 관리자 전용 — GCP GPU VM 원격 제어 등 인프라 운영. 슈퍼어드민 권한 필수."},
]


def _custom_openapi():
    """
    OpenAPI 스키마에 JWT Bearer 보안 스키마(bearerFormat=JWT)와
    설명 마크다운을 명시적으로 주입한다. FastAPI 기본은 bearerFormat을 비워둠.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
        servers=app.servers,
        contact={
            "name": "DRONE INSPECT Backend",
            "email": "droneinspect.noreply@gmail.com",
        },
        license_info={"name": "Proprietary", "identifier": "LicenseRef-Proprietary"},
    )

    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "POST `/api/v1/auth/login` 응답의 `access_token` 값을 그대로 입력 (Bearer 접두사 자동 부여).",
    }
    security_schemes["AIWebhookSecret"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-AI-Webhook-Secret",
        "description": "외부 AI 추론 서버 → 백엔드 콜백(/api/v1/ai/*) 인증용 사전 공유 시크릿.",
    }

    app.openapi_schema = schema
    return schema


# FastAPI 앱 생성
app = FastAPI(
    title="DRONE INSPECT API",
    description=(
        "실제 드론 기반 자율 하자 점검 플랫폼 — FastAPI 백엔드.\n\n"
        "## 인증\n"
        "1. `/api/v1/auth/login` 으로 access_token 획득\n"
        "2. 우측 상단 **Authorize** 버튼 → `HTTPBearer` 에 토큰 입력 → 보호 엔드포인트 테스트 가능\n\n"
        "## WebSocket\n"
        "Swagger 에선 ws:// 핸드셰이크를 직접 호출할 수 없습니다. "
        "`WebSocket` 태그의 엔드포인트는 메시지 스키마/이벤트 흐름 참고용이며, "
        "실제 연결은 프론트엔드 또는 `wscat` 같은 ws 클라이언트로 테스트하세요.\n\n"
        "## 다중 조직\n"
        "한 사용자가 여러 조직에 소속된 경우 `X-Organization-Id` 헤더로 분기합니다 (미지정 시 가장 최근 활성 조직)."
    ),
    version="1.3.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    servers=[
        {"url": "https://aeroinspect-backend.fly.dev", "description": "Production (Fly.io)"},
        {"url": "http://localhost:8000", "description": "Local dev"},
    ],
    swagger_ui_parameters={
        # 페이지 새로고침 후에도 Authorize 토큰 유지 — Swagger 사용성 핵심
        "persistAuthorization": True,
        # 대규모 스키마(20+ 라우터)는 기본 펼침 시 노이즈가 큼
        "defaultModelsExpandDepth": 0,
        "docExpansion": "none",
        "filter": True,
        "tryItOutEnabled": True,
    },
)
app.openapi = _custom_openapi  # type: ignore[method-assign]

# ── 미들웨어 ─────────────────────────────────
# RequestIDMiddleware 먼저 (가장 바깥) → CORS
# add_middleware는 LIFO 순서로 dispatch되므로 마지막에 추가한 것이 가장 바깥.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(RateLimitMiddleware)

# ── 라우터 마운트 ─────────────────────────────
app.include_router(api_router, prefix="/api/v1")

# ── 정적 파일 서빙 (업로드된 프로필 이미지 등) ──
import os
os.makedirs("./uploads/profiles", exist_ok=True)
os.makedirs("./uploads/chat", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")


@app.get("/", tags=["Health"])
async def root():
    """서버 상태 확인용 헬스체크 엔드포인트"""
    return {"status": "ok", "service": "AeroInspect API", "version": "1.3.0"}


@app.get("/metrics", tags=["Observability"], include_in_schema=False)
async def prometheus_metrics():
    """
    Prometheus 스크래퍼용 메트릭 (OpenMetrics 텍스트).
    Grafana → Prometheus datasource → aeroinspect_* 시리즈로 조회.
    Swagger 스키마에서 제외 (include_in_schema=False) — 스크래퍼 전용.
    """
    return render_metrics()


@app.get("/health", tags=["Health"])
async def health_check():
    """
    카메라 + AI 모델 + 스트림 워커 상태 확인.
    3-모델 파이프라인 또는 20종 파이프라인 상태 모두 포함.
    필수 모델 미로드 시 503 반환.
    """
    from app.services.inference_pipeline_20 import pipeline20

    models = inference_pipeline.models_loaded
    all_3model_loaded = models.yolo_thermal and models.yolo_delam and models.wallpaper

    # 20종 파이프라인 상태 (활성화 시)
    pipeline20_status = None
    if settings.USE_20DEFECT_PIPELINE:
        pipeline20_status = {
            "loaded": pipeline20.is_loaded,
            "models": pipeline20.models_loaded.model_dump() if pipeline20.is_loaded else None,
        }

    # 상태 판정: 활성 파이프라인의 필수 모델이 모두 로드되어야 "ok"
    if settings.USE_20DEFECT_PIPELINE:
        is_healthy = pipeline20.is_loaded
    else:
        is_healthy = all_3model_loaded

    from fastapi.responses import JSONResponse

    status_code = 200 if is_healthy else 503
    body = {
        "status": "ok" if is_healthy else "degraded",
        "device": inference_pipeline.device,
        "active_pipeline": "20defect" if settings.USE_20DEFECT_PIPELINE else "3model",
        "models_loaded_3model": {
            "yolo_thermal": models.yolo_thermal,
            "yolo_delam": models.yolo_delam,
            "wallpaper": models.wallpaper,
        },
        "pipeline20": pipeline20_status,
        "wallpaper_classes_count": len(WALLPAPER_CLASSES),
        "stream_worker_running": stream_inference_worker.is_running,
        "stream_worker_stats": stream_inference_worker.stats,
        "frame_skip": settings.FRAME_SKIP,
        "rgb_camera": rgb_camera_service.is_open,
        "thermal_camera": thermal_camera_service.is_open,
        "lidar": {
            "distance_m": lidar_service.latest_distance_m,
            "connected": lidar_service.latest_distance_m is not None,
        },
        "telemetry_cache": {
            "ready": telemetry_cache.is_ready,
            "age_sec": telemetry_cache.age_sec,
        },
    }
    return JSONResponse(content=body, status_code=status_code)
