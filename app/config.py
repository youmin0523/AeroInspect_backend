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
# 값이 "production"·"prod"·"live" 중 하나면 placeholder secret 검증을 강제(raise)한다.
# 그 외 값(개발/테스트)에서는 경고만 띄우고 통과 → 로컬 개발 흐름 방해 X.
APP_ENV_VAR = "APP_ENV"
PROD_ENV_VALUES = {"production", "prod", "live"}

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

    # OpenAI 챗봇 (건축물·하자 도메인 어시스턴트)
    # gpt-4o-mini 기본 — 저비용/저지연. 운영에서 품질 필요 시 OPENAI_MODEL 만 갱신.
    # OPENAI_MAX_OUTPUT_TOKENS: 응답 한 회당 최대 출력 토큰 (비용/길이 가드).
    # OPENAI_SUMMARY_MODEL: 컨텍스트 압축 요약 전용 모델. 비용 절감 위해 mini 동일.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_OUTPUT_TOKENS: int = 1200
    OPENAI_SUMMARY_MODEL: str = "gpt-4o-mini"

    # ── JWT ──────────────────────────────────
    JWT_SECRET: str = "change-me-in-production"
    JWT_EXPIRE_MINUTES: int = 120
    # Refresh token: 장기 유효 (기본 14일). /auth/refresh 엔드포인트로 access token 재발급용.
    JWT_REFRESH_EXPIRE_DAYS: int = 14

    # ── AI Webhook 인증 ──────────────────────
    # AI 추론 서버 → 백엔드 콜백(/api/v1/ai/*) 보호용 사전 공유 시크릿.
    # 빈 값이면 모든 요청이 401로 거부됨 (운영 안전 기본값).
    AI_WEBHOOK_SECRET: str = ""

    # 푸시 알림 프로바이더: "noop" | "fcm" | "apns"
    # 운영 배포 시 firebase-admin 설치 후 "fcm" 으로 전환.
    PUSH_PROVIDER: str = "noop"

    # WebSocket 브로드캐스트 백엔드: "memory" (단일 워커) | "redis" (수평 확장)
    # redis 선택 시 REDIS_URL 필수.
    WS_BACKEND: str = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"

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
        APP_ENV=production 일 때 placeholder/빈값 시크릿이 그대로 들어가면 기동 차단.
        그 외 환경(개발/테스트)에서는 경고만 출력 — 운영 사고 방지.
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
        if env in PROD_ENV_VALUES:
            raise RuntimeError(f"[CONFIG] 운영 환경 기동 차단 — {msg}")
        warnings.warn(f"[CONFIG] 개발 모드 경고 — {msg}", stacklevel=2)
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # //* [Modified Code 2026-05-13] .env 의 추가 키(APP_ENV 등 도구용 변수)가
        # 들어와도 부팅 차단하지 않음 — 알 수 없는 키는 무시.
        extra = "ignore"


# 전역 싱글톤: 애플리케이션 전체에서 공유
settings = Settings()
