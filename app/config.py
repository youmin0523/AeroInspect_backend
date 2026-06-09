# =============================================
# app/config.py
# 역할: 애플리케이션 환경변수 설정 관리
#       pydantic-settings를 사용해 .env 파일에서 값을 로드하고
#       타입 검증 후 전역 settings 객체로 제공한다.
# 사용: from app.config import settings
# =============================================

from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
import json
import os
import warnings
from typing import List, Optional


# 운영(prod) 환경 판정에 사용되는 환경변수.
# placeholder secret 검증 정책 (fail-closed):
# APP_ENV 가 명시적으로 dev/test 계열일 때만 경고 후 통과한다.
# 그 외 모든 값(production·prod·live·staging·오타·미설정/빈값)은 기동을 차단한다.
# → APP_ENV 오타/누락으로 공개 placeholder 키를 단 채 운영 기동되는 사고 방지.
APP_ENV_VAR = "APP_ENV"
PROD_ENV_VALUES = {"production", "prod", "live"}
# 이 값들로 명시 선언한 경우에만 placeholder 시크릿을 경고 수준으로 허용.
DEV_ENV_VALUES = {"development", "dev", "local", "test", "testing", "ci"}

# 운영에서 절대 통과되면 안 되는 placeholder 값 목록.
# config.py default 와 .env.example sentinel 모두 포함.
_PLACEHOLDER_SECRETS = {
    "JWT_SECRET": {"change-me-in-production", "", "secret", "changeme"},
    "AI_WEBHOOK_SECRET": {"change-me-in-production", ""},
    "DB_PASSWORD": {"password", ""},
    "GOOGLE_CLIENT_SECRET": {"your-google-client-secret", ""},
    "KAKAO_CLIENT_SECRET": {"your-kakao-client-secret", ""},
    "NAVER_CLIENT_SECRET": {"your-naver-client-secret", ""},
}


class Settings(BaseSettings):
    # ── Database (개별 변수 → DATABASE_URL 자동 조립) ──
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "aeroinspect"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "aeroinspect_db"
    DATABASE_URL: str = ""

    # ── 커넥션 풀 튜닝 ──
    # SSE 챗/스트리밍 핸들러가 세션을 오래 점유 → 풀 포화 방지를 위해 기본값 상향.
    # pool_timeout: 풀 고갈 시 대기 한도(초). 기본 30s 는 사용자 체감 지연이 큼 → 10s.
    # pool_recycle: 유휴 커넥션 재생성 주기(초). 클라우드 PG/asyncpg 가 유휴 연결을 끊음.
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 10
    DB_POOL_RECYCLE: int = 1800

    @model_validator(mode="after")
    def assemble_database_url(self):
        """개별 DB 환경변수로부터 DATABASE_URL을 자동 조립한다."""
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            )
        return self

    # ── Hardware (드론/카메라/LiDAR 연결 플래그) ──
    # False면 카메라·LiDAR·추론 파이프라인 초기화를 모두 건너뜀 (API만 기동)
    DRONE_CONNECTED: bool = False

    # ── Test Mode (드론 미연결 시 로컬 이미지/영상으로 하자 검출 프로토타입 테스트) ──
    TEST_MODE_ENABLED: bool = True
    TEST_IMAGE_INTERVAL: float = 3.0      # 이미지 전환 주기 (초)
    # 이미지 프레임 detection(ONNX+VLM 하이브리드) 백그라운드 태스크 타임아웃(초).
    # 이 시간을 넘으면 해당 프레임 detection 을 포기 — 느린/멈춘 VLM 이 좀비 태스크로
    # 쌓여 1 vCPU 를 잠식하는 것을 방지. 이미지 표시 자체는 detection 과 무관하게 즉시.
    TEST_DETECT_TIMEOUT_SEC: float = 12.0
    TEST_IMAGES_DIR: str = "./training/test_external"
    TEST_THERMAL_DIR: str = "./training/gdrive_raw/B01_B02_D01_crack900_thermal_rgb_seg/Crack900/data/Images/2_IR/train"
    TEST_UPLOAD_DIR: str = "./uploads/test_files"

    # ── Camera ───────────────────────────────
    RGB_CAMERA_INDEX: int = 0
    THERMAL_CAMERA_INDEX: int = 1

    # ── LiDAR ────────────────────────────────
    LIDAR_SERIAL_PORT: str = "COM3"
    LIDAR_BAUD_RATE: int = 115200

    # ── AI Model (3-모델 파이프라인) ──────────
    # 가중치 디렉토리 + 개별 파일명 분리 → 배포 환경별로 경로만 바꾸면 됨
    AEROINSPECT_WEIGHTS_DIR: str = "./models_weights"
    YOLO_THERMAL_WEIGHTS: str = "yolov8s_crack_moisture_best.pt"
    YOLO_DELAM_WEIGHTS: str = "yolov8s_delamination_best.pt"
    WALLPAPER_WEIGHTS: str = "resnet50_wallpaper_best.pt"

    # YOLO 공통 신뢰도 임계값 — 0.10으로 하향하여 Recall 극대화
    # Precision 하락은 TemporalFilter가 보완
    YOLO_CONF_THRESHOLD: float = 0.10
    # ResNet50 벽지 분류 신뢰도 임계값 (val_acc 54% 감안, top1 최소 신뢰도)
    WALLPAPER_CONF_THRESHOLD: float = 0.35
    # top1 - top2 최소 마진. 모호한 예측(top1/top2 근소차) 차단용
    WALLPAPER_MARGIN_THRESHOLD: float = 0.15

    # WebSocket 스트림 추론 — N프레임 중 1프레임만 추론 (GPU 부하 분산)
    FRAME_SKIP: int = 3

    # 추론 디바이스: 'auto' | 'cuda' | 'cpu'
    DEVICE: str = "auto"

    # 로깅 — JSON 출력은 운영 권장, 개발은 컬러 콘솔
    LOG_JSON: bool = False
    LOG_LEVEL: str = "INFO"

    # ── 20종 하자 검출 ONNX 파이프라인 (6-Model + Geometric) ──
    # M1: 구조·방수 (2-Stage YOLO→ResNet)
    M1_YOLO_ONNX: str = "m1_yolo_structural.onnx"
    M1_RESNET_ONNX: str = "m1_resnet_crack_classifier.onnx"
    M1_CONF_THRESHOLD: float = 0.25          # 4차 결과 최고 — precision 우위 (FP 감소)

    # M2: 마감·표면 (2-Stage YOLO→ResNet)
    M2_YOLO_ONNX: str = "m2_yolo_surface.onnx"
    M2_RESNET_ONNX: str = "m2_resnet_surface_classifier.onnx"
    M2_CONF_THRESHOLD: float = 0.30

    # M3: 바닥·창호 (2-Stage YOLO→ResNet)
    M3_YOLO_ONNX: str = "m3_yolo_floor_window.onnx"
    M3_RESNET_ONNX: str = "m3_resnet_floor_window_classifier.onnx"
    M3_CONF_THRESHOLD: float = 0.30

    # M4: 열화상 단열 (U-Net + RGB Context)
    M4_UNET_ONNX: str = "m4_unet_thermal_insulation.onnx"
    M4_CONTEXT_ONNX: str = "m4_yolo_context_elements.onnx"
    M4_INSULATION_WALL_DELTA: float = 3.5    # 벽체 단열 온도차 임계값 (°C)
    M4_INSULATION_WINDOW_DELTA: float = 2.0  # 창호 단열 온도차 임계값 (°C)
    M4_AIRTIGHT_DELTA: float = 1.5           # 기밀 불량 온도차 임계값 (°C)
    M4_FLOOR_HEATING_DELTA: float = 2.0      # 바닥 난방 편차 임계값 (°C)

    # M5+G1: 기하학 (YOLOv8m-seg + Hough/RANSAC)
    M5_SEG_ONNX: str = "m5_yolo_seg_frames.onnx"
    ALIGNMENT_ANGLE_THRESHOLD: float = 0.2   # 수직수평 편차 임계값 (도)
    SQUARENESS_ANGLE_THRESHOLD: float = 0.3  # 직각도 편차 임계값 (도)

    # M6: PatchCore 앙상블 폴백
    M6_PATCHCORE_ONNX: str = "m6_patchcore_feature_extractor.onnx"
    PATCHCORE_THRESHOLD: float = 27.0        # 이상 점수 임계값 (feature extractor + coreset 거리 기반)

    # Thermal Anomaly (Moisture/delam YOLO 대체 — PatchCore unsupervised)
    # 학습: thermal_yolo 정상 패치 2000개 (라벨 영역 제외 crop)
    # 출력: anomaly heatmap → bbox 변환 후 grade 분류
    # 활성화: 사용자 명시 (2026-05-28) — Thermal Anomaly 일시 보류, M4 U-Net 단열은 유지
    THERMAL_ANOMALY_ENABLED: bool = False    # False면 ONNX 있어도 로드/추론 X (보류 상태)
    THERMAL_ANOMALY_ONNX: str = "thermal_anomaly.onnx"
    THERMAL_ANOMALY_THRESHOLD: float = 0.5   # anomaly score 임계 (0~1 정규화). 첫 적용 후 튜닝
    THERMAL_ANOMALY_BBOX_MIN_AREA: int = 400 # anomaly mask → bbox 변환 시 최소 픽셀 영역

    # furniture_aware: 빌트인 가구 인식 (M1+M2+M3 검출이 가구 위면 false positive 차단용)
    FURNITURE_AWARE_ONNX: str = "furniture_aware.onnx"
    FURNITURE_AWARE_CONF_THRESHOLD: float = 0.60  # 매우 보수적 (확실한 가구만)
    PATCHCORE_ENSEMBLE_BOOST: float = 0.15   # 앙상블 신뢰도 승격 값

    # 열화상-RGB 공간 정렬 Homography (3x3 JSON)
    THERMAL_RGB_HOMOGRAPHY: str = "thermal_rgb_homography.json"

    # 계층적 실행 설정
    TIER1_FRAME_SKIP: int = 3                # M1+M2 실행 주기
    TIER2_FRAME_SKIP: int = 6                # M3+M5 실행 주기
    TIER3_FRAME_SKIP: int = 9                # M4+M6 실행 주기

    # 시간 일관성 필터
    TEMPORAL_FILTER_WINDOW: int = 5          # 프레임 윈도우 크기
    TEMPORAL_FILTER_MIN_DETECTIONS: int = 2  # 최소 검출 횟수
    TEMPORAL_INSTANT_THRESHOLD: float = 0.85 # 즉시 보고 임계값
    TEMPORAL_FILTER_IOU: float = 0.3         # IoU 매칭 임계값

    # ByteTrack 객체 추적
    TRACKER_MIN_HITS: int = 3               # 트랙 확정 최소 탐지 횟수
    TRACKER_MAX_AGE: int = 15               # 미탐지 허용 프레임 수
    TRACKER_IOU_THRESHOLD: float = 0.3      # ByteTrack IoU 매칭 임계값

    # Hard Example Mining (Active Learning Phase 1)
    HARD_EXAMPLE_ENABLED: bool = False       # 기본 비활성 (명시적 활성화 필요)
    HARD_EXAMPLE_DIR: str = "./training/hard_examples"
    HARD_EXAMPLE_LOW_CONF_MIN: float = 0.15  # 수집 대상 하한
    HARD_EXAMPLE_LOW_CONF_MAX: float = 0.40  # 수집 대상 상한
    HARD_EXAMPLE_SAVE_INTERVAL: float = 30.0 # 디스크 저장 주기 (초)

    # 신규 파이프라인 활성화 플래그 (기존 파이프라인과 전환용)
    USE_20DEFECT_PIPELINE: bool = False

    # ── Legacy (하위 호환용, 신규 코드에선 사용 금지) ──
    YOLO_WEIGHTS_PATH: str = "./models_weights/aeroinspect_yolov8.pt"

    # ── LLM ──────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # 보고서 생성 LLM (llm_report). 모델/타임아웃을 코드 변경 없이 갱신 가능하게 설정화.
    # Opus 는 보고서 대량 텍스트엔 과한 비용 — 기본은 Sonnet 으로 비용/지연 절충.
    REPORT_CLAUDE_MODEL: str = "claude-sonnet-4-6"
    REPORT_GEMINI_MODEL: str = "gemini-1.5-pro"
    REPORT_MAX_TOKENS: int = 4096
    # 모든 외부 LLM 호출 공통 타임아웃(초). 멈춘 프로바이더가 슬롯/스레드를 무한 점유하지 않도록.
    LLM_REQUEST_TIMEOUT: float = 60.0

    # OpenAI 챗봇 (건축물·하자 도메인 어시스턴트)
    # gpt-4o-mini 기본 — 저비용/저지연. 운영에서 품질 필요 시 OPENAI_MODEL 만 갱신.
    # OPENAI_MAX_OUTPUT_TOKENS: 응답 한 회당 최대 출력 토큰 (비용/길이 가드).
    # OPENAI_SUMMARY_MODEL: 컨텍스트 압축 요약 전용 모델. 비용 절감 위해 mini 동일.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_OUTPUT_TOKENS: int = 1200
    OPENAI_SUMMARY_MODEL: str = "gpt-4o-mini"

    # ── VLM 하자 검출 (비전 LLM, 기존 ONNX와 병행 비교 PoC) ──
    # 학습 모델 검출률이 낮아(M4 mAP 0.503), Gemini/Claude/GPT-4o 비전 모델로
    # 이미지/키프레임을 직접 판정. 기존 ONNX 경로는 그대로 두고 병렬 오버레이로 추가.
    #   VLM_PROVIDER: gemini(기본·저비용) | claude | openai
    #   VLM_MODE: classify(이미지 단위 판정, bbox 없음) | grounding(Gemini bbox 0~1000)
    #   VLM_KEYFRAME_INTERVAL_SEC: 근실시간 스트림 샘플 주기 (프레임마다 호출 불가)
    #   VLM_DAILY_CALL_CAP: 일일 호출 상한 (비용 가드)
    VLM_DETECTION_ENABLED: bool = False
    VLM_PROVIDER: str = "gemini"
    VLM_MODEL: str = "gemini-2.5-flash"
    VLM_MODE: str = "classify"
    VLM_KEYFRAME_INTERVAL_SEC: float = 4.0
    VLM_MAX_CONCURRENCY: int = 2
    VLM_DAILY_CALL_CAP: int = 2000

    # ── 하이브리드 판정 (ONNX 제안 + VLM 판정, 상업용 기본 경로) ──
    #   VLM_HYBRID_ENABLED: 스트림 키프레임을 하이브리드로 처리 (False면 VLM 단독)
    #   VLM_ADJUDICATE_CONFLICTS: 고conf 충돌 시 1회 한정 재판정(토론-lite). 비용 ↑ 소폭.
    #   VLM_CONFLICT_ONNX_CONF: 재판정 발동 ONNX 신뢰도 하한
    VLM_HYBRID_ENABLED: bool = True
    VLM_ADJUDICATE_CONFLICTS: bool = True
    VLM_CONFLICT_ONNX_CONF: float = 0.70
    # VLM이 추가한(거친) 박스를 고전 CV로 실제 하자 픽셀에 스냅 보정
    # (균열=선형, 녹물=색, 박리=텍스처). 실패 시 원본 박스 유지(안전).
    VLM_BOX_REFINE: bool = True

    # ── 검출 주도권 / VLM 앙상블 (2026-06-09) ──
    #   VLM_PRIMARY: True면 VLM(grounding)이 1차 검출 주도 + ONNX 교차검증(합의→CONFIRMED,
    #                박스=ONNX 정밀 / VLM 단독=box_refiner 보정 후 REVIEW / ONNX 단독=REVIEW).
    #                False면 기존 ONNX 주도 캐스케이드(adjudicate).  ONNX recall이 약해 VLM 우선.
    #   VLM_ENSEMBLE: 병렬 호출할 "provider:model" 쉼표구분 — 여러 VLM 합의(앙상블)로 신뢰도↑.
    #   VLM_PRIMARY_IOU: VLM↔ONNX를 같은 검출로 볼 IoU 하한.
    VLM_PRIMARY: bool = True
    VLM_ENSEMBLE_ENABLED: bool = True
    VLM_ENSEMBLE: str = "gemini:gemini-2.5-flash,openai:gpt-4o"
    VLM_PRIMARY_IOU: float = 0.3

    # ── JWT ──────────────────────────────────
    JWT_SECRET: str = "change-me-in-production"
    JWT_EXPIRE_MINUTES: int = 120
    # Refresh token = "재접속 유휴 윈도우". /auth/refresh 로 access 재발급 + 회전(rotation)으로
    # 사용할 때마다 수명이 갱신됨 → 이 시간 안에 다시 접속하면 로그인 유지, 넘기면 자동 로그아웃.
    # 영구 로그인이 아니라 "실수로 브라우저 닫고 다시 열기 / 일정 시간 내 재접속" UX 용. 기본 24시간.
    JWT_REFRESH_EXPIRE_HOURS: int = 24

    # ── AI Webhook 인증 ──────────────────────
    # AI 추론 서버 → 백엔드 콜백(/api/v1/ai/*) 보호용 사전 공유 시크릿.
    # 빈 값이면 모든 요청이 401로 거부됨 (운영 안전 기본값).
    AI_WEBHOOK_SECRET: str = ""

    # 푸시 알림 프로바이더: "noop" | "fcm" | "apns"
    # 운영 배포 시 자격증명 설정 후 "fcm" 또는 "apns" 로 전환.
    PUSH_PROVIDER: str = "noop"
    # FCM HTTP v1 — 서비스 계정 JSON(원문 또는 base64) + 프로젝트 ID.
    #   GCP Compute 용(GCP_SERVICE_ACCOUNT_JSON)과 별개의 Firebase 서비스 계정 권장.
    FCM_CREDENTIALS_JSON: str = ""
    FCM_PROJECT_ID: str = ""
    # APNs (token-based, .p8) — 키/팀/토픽. APNS_USE_SANDBOX=True 면 개발 게이트웨이.
    APNS_AUTH_KEY: str = ""       # .p8 키 원문(또는 base64)
    APNS_KEY_ID: str = ""
    APNS_TEAM_ID: str = ""
    APNS_TOPIC: str = ""         # 앱 번들 ID
    APNS_USE_SANDBOX: bool = True

    # WebSocket 브로드캐스트 백엔드: "memory" (단일 워커) | "redis" (수평 확장)
    # redis 선택 시 REDIS_URL 필수.
    WS_BACKEND: str = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"

    # 레이트리밋 백엔드: "memory" (단일 워커) | "redis" (멀티워커 정합).
    # redis 선택 시 REDIS_URL 사용, 연결 실패하면 자동으로 memory 폴백.
    RATE_LIMIT_BACKEND: str = "memory"

    # 토큰 폐기(denylist): 로그아웃/리프레시 회전 시 jti 를 Redis 에 등록해 즉시 무효화.
    # Redis 미가용/미설정이면 자동 비활성(폐기 없이 만료까지 유효 — 기존 동작).
    TOKEN_DENYLIST_ENABLED: bool = True

    # 타일드 추론 멀티스케일 imgsz 후보 (실배치 추론용).
    TILED_INFERENCE_IMGSZ: list[int] = [640, 1024]

    # ── GCP GPU VM 원격 제어 (관리자 콘솔) ────
    # Fly.io 백엔드(항상 켜진 상태)에서 GCP Compute Engine REST API 를 호출해
    # GPU L4 VM 의 ON/OFF 를 제어. 시간당 ~$0.71 비용 절감 + 상용 멀티유저 운영용.
    # GCP_SERVICE_ACCOUNT_JSON: 서비스 계정 키(JSON) 원문 또는 base64.
    #   - 권한: roles/compute.instanceAdmin.v1 (instances.start / instances.stop / instances.get)
    GCP_SERVICE_ACCOUNT_JSON: str = ""
    GCP_PROJECT_ID: str = ""
    GCP_GPU_ZONE: str = "asia-northeast3-a"
    GCP_GPU_INSTANCE: str = "drone-stream-api"

    # ── OAuth (SNS 로그인) ────────────────────
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    KAKAO_CLIENT_ID: str = ""
    KAKAO_CLIENT_SECRET: str = ""
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    OAUTH_REDIRECT_BASE: str = "http://localhost:5173"

    # 이메일 본문 등에서 쓰는 프론트엔드 기본 URL (로그인 링크 등). 운영에선 실제 도메인으로.
    FRONTEND_BASE_URL: str = "http://localhost:5173"

    # ── WebSocket ────────────────────────────
    WS_HEARTBEAT_INTERVAL: int = 3

    # ── Streaming ────────────────────────────
    MJPEG_JPEG_QUALITY: int = 80
    THERMAL_BLEND_ALPHA: float = 0.5

    # ── Recording ────────────────────────────
    RECORDING_OUTPUT_DIR: str = "./recordings"
    RECORDING_FPS: float = 30.0
    RECORDING_CODEC: str = "mp4v"

    # ── Email (SMTP) ─────────────────────────
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "noreply@droneinspect.com"
    SMTP_FROM_NAME: str = "DRONE INSPECT"

    # ── Sentry (운영 에러 모니터링) ────────────
    # SENTRY_DSN 이 비어 있으면 init_sentry 가 no-op → 로컬 개발 영향 0.
    # APP_ENV=production 이고 DSN 이 비어 있으면 startup 시 경고 로그만 (기동 차단 X).
    # TRACES_SAMPLE_RATE: 트랜잭션 성능 추적 샘플링 비율 (0.0~1.0). 운영 비용 가드용 0.1 기본.
    # PROFILES_SAMPLE_RATE: 프로파일링은 비용 큼 → 기본 비활성(0.0). 필요 시 0.05~0.1.
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: str = "development"
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    SENTRY_PROFILES_SAMPLE_RATE: float = 0.0

    # ── CORS ─────────────────────────────────
    # R-v1.1.17: Vercel 배포 URL 추가 (memory reference_production_urls)
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://www.aeroinspect.site",
        "https://aeroinspect.site",
        "https://aero-inspect-frontend.vercel.app",
        "https://aero-inspect-frontend-git-main.vercel.app",
        "https://aero-inspect-frontend-git-develop.vercel.app",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """JSON 배열 또는 콤마 구분 문자열 모두 허용.

        허용 형식:
        - '["https://a.com","https://b.com"]' (JSON)
        - 'https://a.com,https://b.com' (콤마)
        - '' (빈 문자열) → []
        """
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @model_validator(mode="after")
    def enforce_no_placeholder_secrets_in_prod(self):
        """
        placeholder/빈값 시크릿 검증 — fail-closed.
        APP_ENV 가 명시적으로 dev/test 계열(DEV_ENV_VALUES)일 때만 경고 후 통과하고,
        그 외 모든 값(production/staging/오타/미설정)은 기동을 차단한다.
        → APP_ENV 오타·누락 시 공개 placeholder 키로 운영 기동되는 사고 방지.
        """
        env = os.environ.get(APP_ENV_VAR, "").strip().lower()
        offending: list[str] = []
        for field, bad_values in _PLACEHOLDER_SECRETS.items():
            value = getattr(self, field, None)
            if value in bad_values:
                offending.append(field)

        if not offending:
            return self

        msg = (
            "보안: 다음 환경변수가 placeholder/빈값 상태입니다 → " + ", ".join(offending)
            + ". .env 또는 시크릿 매니저에서 실제 값을 주입하세요."
        )
        # 명시적으로 dev/test 라고 선언한 경우에만 경고로 완화.
        if env in DEV_ENV_VALUES:
            warnings.warn(f"[CONFIG] 개발 모드 경고 — {msg}", stacklevel=2)
            return self
        # production/staging/오타/미설정 등 그 외 전부 차단 (fail-closed).
        raise RuntimeError(
            f"[CONFIG] 기동 차단(fail-closed) — {msg} "
            f"(로컬 개발이라면 APP_ENV 를 {sorted(DEV_ENV_VALUES)} 중 하나로 명시 설정하세요.)"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # //* [Modified Code 2026-05-13] .env 의 추가 키(APP_ENV 등 도구용 변수)가
        # 들어와도 부팅 차단하지 않음 — 알 수 없는 키는 무시.
        extra = "ignore"


# 전역 싱글톤: 애플리케이션 전체에서 공유
settings = Settings()
