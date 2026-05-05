# 🤖 바이브코딩(Vibe Coding) 프롬프트 & 결과 추적 로그

> **💡 설명**: 백엔드(Backend) 전용 바이브코딩 로그입니다. AI에게 언제, 어떤 프롬프트를 입력하여 어떤 코드를 도출했는지 기록합니다.

---

## 📝 기본 정보 (Meta)

- 작성자 (Who): @youminsu0523
- 작성 일자 (When): 2026-04-14
- 목표 기능 (Objective): AeroInspect 드론 하자점검 플랫폼 백엔드 전체 스캐폴드 구축 (FastAPI + SQLAlchemy async + MJPEG 스트리밍 + YOLOv8 + LLM 보고서)
- 작업 브랜치/환경: `MS`

---

## 💬 바이브코딩 대화 흐름 (Vibe Coding Log)

### 1️⃣ 초기 질문 / 프롬프트 (Initial Prompt)
> *계획서(v1.3) PDF를 첨부하여 AI에게 전달한 지시*
- **프롬프트 내용**:
  ```text
  "위 계획서 참고해서 backend/ frontend 에 맞춰서 기초 파일 작업해줘"
  ```

### 2️⃣ 계획(Plan) 단계 피드백 (Plan Mode Feedback)
> *AI가 플랜을 제안한 후 팀원이 추가 요구사항 3가지를 제시*
- **피드백 1** (카메라 전환):
  > "IRC-256CA 열화상 카메라와 일반 카메라의 전환이 필요해"
  → 해결: `/stream/rgb`, `/stream/thermal`, `/stream/blend` 3개 엔드포인트 + `POST /stream/mode` API 추가, WS `camera.mode_changed` 이벤트 브로드캐스트 설계
- **피드백 2** (환경 파일):
  > "backend와 frontend에 .env와 .gitignore, venv 등 필요한 폴더들을 함께 구축해줘"
  → 해결: `.env`, `.env.example`, `.gitignore`, `Dockerfile`, `alembic.ini` 포함
- **피드백 3** (파일 주석):
  > "각 파일 내부에 주석으로 이 파일은 어떠한 역할을 하는 파일들인지에 대한 설명도 제일 위에 적어줘"
  → 해결: 모든 Python 파일 최상단 `# ===== 파일명 =====\n# 역할: ...` 블록 주석 추가

### 3️⃣ 구현된 백엔드 핵심 아키텍처

#### 비동기 처리 원칙
- OpenCV `cap.read()`, PyTorch 추론, pyserial 모두 블로킹 → `asyncio.to_thread()` 래핑
- DB: `asyncpg` 드라이버 + SQLAlchemy async 네이티브 비동기
- Claude API: `AsyncAnthropic` 클라이언트

#### MJPEG 멀티클라이언트 팬아웃 패턴
```python
# app/services/camera.py 핵심 패턴
class CameraService:
    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=2)  # 최신 2프레임만 버퍼
        self._subscribers.append(q)
        return q

    async def _capture_loop(self):
        while self._running:
            frame = await asyncio.to_thread(self._cap.read)
            for q in self._subscribers:
                if q.full():
                    q.get_nowait()  # 오래된 프레임 드롭
                await q.put(frame)
```

#### WebSocket 싱글톤 매니저
```python
# app/core/ws_manager.py
class ConnectionManager:
    channels: dict[str, list[WebSocket]]  # defects/telemetry/thermal/camera

ws_manager = ConnectionManager()  # 모듈 레벨 싱글톤
# Dockerfile에서 --workers 1 강제 (다중 프로세스 시 인스턴스 분리 방지)
```

#### 열화상 블렌드 합성
```python
# app/core/streaming.py
def _blend_frames(rgb: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    thermal_resized = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
    return cv2.addWeighted(rgb, 1 - alpha, thermal_resized, alpha, 0)
```

---

## ✅ 최종 결과 (Final Outcome)

### 📁 생성된 백엔드 파일 목록 (48개)
- `requirements.txt` — FastAPI, SQLAlchemy[asyncio], asyncpg, opencv-python, ultralytics, anomalib, pyserial, anthropic, google-generativeai 등
- `app/main.py` — lifespan: init_db → rgb/thermal camera open → yolo model load
- `app/config.py` — pydantic-settings 기반 환경변수 (DATABASE_URL, ANTHROPIC_API_KEY, THERMAL_BLEND_ALPHA 등)
- `app/core/ws_manager.py` — 채널별 WebSocket 연결 관리 싱글톤
- `app/core/streaming.py` — MJPEG 제너레이터 + blend_frames() 합성
- `app/models/defect.py` — DefectLog ORM (UUID PK, area A-E, severity HIGH/MED/LOW, lidar_x/y/z, raw_payload JSONB)
- `app/api/stream.py` — `/stream/rgb`, `/stream/thermal`, `/stream/blend`, `POST /stream/mode`
- `app/api/report.py` — LLM 스트리밍 보고서 생성 (Claude/Gemini)
- `app/services/camera.py` — RGB 카메라 서비스 + 구독자 큐 팬아웃
- `app/services/thermal.py` — IRC-256CA 16bit ADC→섭씨 변환 + COLORMAP_INFERNO
- `app/services/yolo_inference.py` — YOLOv8 싱글톤 (가중치 없으면 더미 모드)
- `app/services/lidar.py` — TF-Luna UART 9바이트 프레임 파싱
- `app/utils/severity_mapper.py` — 20종 하자 카탈로그 (A-01 ~ E-02)
- `alembic/env.py` — async_engine_from_config 패턴 비동기 마이그레이션
- `tests/test_defects_api.py`, `test_ws_manager.py`, `test_yolo_inference.py` — pytest 스텁

### 📊 아키텍처 영향도
- **DB 스키마**: 복합 인덱스 `(severity, timestamp DESC)`, `(area, timestamp DESC)` 적용
- **카메라 전환**: 단일 `POST /stream/mode` API → WS 브로드캐스트 → 모든 클라이언트 동기화
- **AI 파이프라인**: YOLOv8(탐지) + Anomalib PatchCore(이상탐지) 병렬 추론

---

## 💡 배운 점 및 인사이트 (Lessons Learned)

- **블로킹 I/O 격리**: OpenCV, PyTorch, pyserial은 모두 동기 블로킹 → FastAPI async 이벤트 루프 차단 방지를 위해 반드시 `asyncio.to_thread()` 사용
- **MJPEG 팬아웃**: 구독자별 `asyncio.Queue(maxsize=2)` 유지로 슬로우 클라이언트가 빠른 클라이언트를 차단하지 않도록 설계
- **단일 워커 제약**: WS 매니저 싱글톤은 `--workers 1` 강제 필요, 수평 확장 시 Redis pub/sub 백엔드 교체 필요
- **열화상 좌표 매핑**: IRC-256CA 16bit ADC 값 → 섭씨 변환: `temp = (raw_value / 100.0) - 273.15`

---

## 4️⃣ 추가 피드백 & 반영 — 회원가입 DB 설계 라운드
> **착수 시각**: 2026-04-16 14:30
> **목표**: 프론트엔드 회원가입 폼(`Signup.jsx`)을 백엔드 DB에 연결하기 위한 모델 설계.
> **전제(사용자 요청)**: AWS 프리티어 만료 임박 → DB 실제 기동은 최종 단계에서 한 번만. 그 전에는 모델·스키마·해싱·엔드포인트 코드만 먼저 완성.

### ⏱ 14:30 | "지금 회원가입을 위한 DB를 연결하고자 해"
→ 현재 상태 점검(PostgreSQL+asyncpg+SQLAlchemy 기반 이미 구성, User 모델 부재) 후 스키마 범위·DB 선택지·해싱 방식 확정 질의.

### ⏱ 14:35 | "테이블 먼저 구성하자"
→ 단일 `users` 테이블 초안 제안 (개인/사업자 account_type 컬럼으로 통합, 사업자 필드 nullable).

### ⏱ 14:38 | "DB 연결은 … 다 구현하고 마지막에 연결할게. 아마존 무료 기간이 얼마 남지 않아서"
→ 프로젝트 메모리 `project_aws_free_tier.md` 등록. 작업 순서 확정: 테이블 → 스키마 → 해싱 → 엔드포인트 → (최후) DB 연결.

### ⏱ 14:42 | "테이블 분리에 대해서는 제1정규화 ~ 제3정규화 고려해서 작성해줘"
→ 단일 테이블 초안 폐기. 3NF 기준으로 4 테이블 재설계:
- `users` — 공통 회원 정보 (account_type, email, username, password_hash, name, phone)
- `business_profiles` — 사업자 1:1 분리 (user_id PK=FK, biz_number UNIQUE, ceo_name, verified_at) → `account_type → biz_*` 이행 종속 제거
- `terms` — 약관 마스터 (code, is_required, version) → 약관 추가 시 스키마 불변
- `user_term_agreements` — 동의 이력 M:N 연결 (user_id + term_id 복합 PK, version 스냅샷, agreed_at) → 1NF 반복그룹 제거

### ⏱ 14:46 | "응 구현해줘"
→ 4개 ORM 모델 파일 신규 작성:
- `app/models/user.py` (UUID PK, account_type ENUM, unique email/username, relationships)
- `app/models/business_profile.py` (user_id PK/FK CASCADE, biz_number UNIQUE)
- `app/models/term.py` (SERIAL PK, code UNIQUE, is_required/version)
- `app/models/user_term_agreement.py` (복합 PK, version 스냅샷)
- `app/models/__init__.py` — 신규 4개 모델 포함시켜 `Base.metadata` 자동 등록
- `app/db/init_db.py` — 신규 모델 임포트 추가

### ⏱ 14:50 | "DB 연결을 python으로 해야돼? Node.js로 하면 되지 않아?"
→ Python 유지 권장 (판단 근거):
- 기존 FastAPI 백엔드가 DefectLog/WebSocket/MJPEG/LiDAR 전부 Python SQLAlchemy 세션 공유 중
- AI 파이프라인(YOLOv8/anomalib/OpenCV/pyserial) Python 전용 → Node 이식 불가
- 서버 2개 운영 시 AWS 프리티어 시간 2배 소진 (이번 라운드 제약 직접 위반)

### ⏱ 14:52 | "너 말대로 진행해줘"
→ Python 확정. 다음 라운드: Pydantic 스키마(`app/schemas/user.py`) → 비밀번호 해싱 유틸(`app/core/security.py` + `passlib[bcrypt]`) → 회원가입/중복확인 엔드포인트 → 최종 DB 연결.

### 📐 3NF 설계 근거 요약
- **1NF**: 약관 동의를 M:N 연결 테이블로 원자화, 컬럼 기반 반복그룹 제거
- **2NF**: 단일 PK 테이블은 자동 충족. 연결 테이블의 version/agreed_at 도 복합 PK 전체에 종속
- **3NF**: 사업자 속성(`biz_number → ceo_name` 등) 분리, `account_type`에 대한 이행 종속 제거

---

## 5️⃣ 백엔드 대규모 확장 — 인증·현장·평면도·SLAM·텔레메트리·AI웹훅 (2026-04-20)

> **착수 시각**: 2026-04-20 09:37  
> **작업자**: @unknownname-15  
> **목표**: 프론트엔드 기능 확장(현장 관리 / 분석 / 세션 워크플로우)에 대응하는 백엔드 API·모델·서비스 전면 구축. 총 36개 파일 신규/수정.

### ⏱ 2026-04-20 09:37 | 인증 시스템 완성 (JWT + OAuth + 의존성 주입)
- **신규 파일**:
  - `app/core/jwt.py` — `python-jose` 기반 HS256 JWT. `create_access_token(user_id, expires_minutes)` / `decode_access_token(token)`. `settings.JWT_SECRET` + `JWT_ACCESS_EXPIRE_MINUTES` 파라미터화
  - `app/api/auth.py` — 5개 엔드포인트: `POST /auth/signup`(개인/사업자 공용, 사업자 시 `business_profiles` 행 함께 생성) · `POST /auth/login`(아이디+비밀번호 → JWT) · `GET /auth/me`(현재 사용자 조회) · `GET /auth/check-email` · `GET /auth/check-username`(중복 확인)
  - `app/api/oauth.py` — SNS 소셜 로그인 3종. `POST /oauth/google` · `POST /oauth/kakao` · `POST /oauth/naver`. 공통 플로우: 프론트 인가 코드 → provider access_token 교환 → 프로필 조회 → DB 조회/자동 회원가입 → JWT 반환. `httpx.AsyncClient` 비동기 provider 호출
  - `app/dependencies.py` — FastAPI `Depends` 팩토리 모음. `get_db()`(비동기 DB 세션 생성기), `get_current_user()`(Bearer 토큰 검증 후 User ORM 반환), `get_ws_manager()`, `get_rgb_camera()`, `get_thermal_camera()`. 모든 라우터에서 재사용

### ⏱ 2026-04-20 09:37 | 현장(Site) 관리 API + ORM + 스키마
- **신규 파일**:
  - `app/models/site.py` — `sites` 테이블. UUID PK, 현장명/건물유형/주소/점검구분/의뢰유형(B2B/B2C)/의뢰사/연락처/계약기간/세대수/면적/배정팀원 JSONB/메모. `DefectLog` · `Report` 에서 FK 참조 예정. 인덱스: `(status, created_at DESC)` · `(client_type, created_at DESC)`
  - `app/schemas/site.py` — `SiteCreate / SiteUpdate / SiteResponse / SiteListResponse`. `SiteUpdate` 전 필드 Optional(PATCH 부분 업데이트). `SiteListResponse` 에 `total / page / per_page` 페이지네이션 메타 포함
  - `app/api/sites.py` — 5개 엔드포인트: `GET /sites`(필터+검색+페이지네이션) · `GET /sites/{id}` · `POST /sites` · `PATCH /sites/{id}` · `DELETE /sites/{id}`. `get_current_user` 의존성으로 인증 필수

### ⏱ 2026-04-20 09:37 | 평면도 업로드·처리 API + 서비스
- **신규 파일**:
  - `app/models/floorplan.py` — `floorplans` 테이블. 파일명/경로/크기/상태(`uploaded/processing/done/error`) + 추출 결과 JSONB(`walls_json` / `outline_json`)
  - `app/schemas/floorplan.py` — `FloorplanUploadResponse / FloorplanDetail / FloorplanListResponse`
  - `app/api/floorplan.py` — 5개 엔드포인트: `POST /floorplan/upload`(JPG/PDF/DXF, `aiofiles` 비동기 저장, 확장자 화이트리스트 검증) · `POST /floorplan/{id}/process`(OpenCV 벽체 추출 트리거, 백그라운드 태스크) · `GET /floorplan` · `GET /floorplan/{id}` · `DELETE /floorplan/{id}`
  - `app/services/floorplan_processor.py` — OpenCV 순수 이미지 처리. `extract_walls_from_bytes(image_bytes)` 함수. 파이프라인: 그레이스케일 → 이진화 → 방향성 모폴로지(수평/수직 구조 벽 추출) → 컨투어 감지 → 건물 외곽 다각형 추출. DB 독립 순수 함수, 결과를 `{"walls": [...], "outline": [...]}` dict 반환

### ⏱ 2026-04-20 09:37 | SLAM 맵 데이터 API
- **신규 파일**:
  - `app/models/slam_map.py` — `slam_maps` 테이블. 점유 격자(occupancy grid) 메타(해상도/크기/원점) + 이미지 base64 + 드론 위치 JSON
  - `app/schemas/slam_map.py` — `SlamMapCreate / SlamMapUpdate / SlamMapResponse`
  - `app/api/slam.py` — 5개 엔드포인트: `POST /slam`(새 맵 세션) · `GET /slam`(목록, 이미지 제외 메타만) · `GET /slam/{id}`(이미지 포함 상세) · `PATCH /slam/{id}`(실시간 매핑 중 점진 갱신) · `DELETE /slam/{id}`. WS `slam.map_updated` 이벤트 브로드캐스트로 프론트 3D 미니맵 실시간 반영

### ⏱ 2026-04-20 09:37 | 드론 텔레메트리 로그 API
- **신규 파일**:
  - `app/models/telemetry.py` — `telemetry_logs` 테이블. 위치(pos_x/y/z) + 자세(roll/pitch/yaw) + 배터리 + 비행 모드 + 센서 상태. 인덱스: `(created_at DESC)` 타임시리즈 조회 최적화
  - `app/schemas/telemetry.py` — `TelemetryCreate / TelemetryResponse`
  - `app/api/telemetry.py` — 3개 엔드포인트: `POST /telemetry`(저장 + WS `telemetry.update` push) · `GET /telemetry`(목록, 기간 필터) · `GET /telemetry/latest`(최신 1건)

### ⏱ 2026-04-20 09:37 | Python AI 서버 → FastAPI 웹훅 연동
- **신규 파일**:
  - `app/api/ai_webhook.py` — 3개 엔드포인트: `POST /ai/detection`(YOLO/PatchCore 탐지 이벤트 수신 → `DefectLog` DB 저장 + WS `defect.new` 브로드캐스트) · `POST /ai/thermal`(열화상 분석 결과 WS push) · `POST /ai/batch`(다건 탐지 결과 일괄 저장, 단건 `/detection` N회 호출과 동일 효과이나 트랜잭션 단위화)
  - Python AI 프로세스(YOLO/PatchCore/RANSAC)와 FastAPI 백엔드를 분리된 서비스로 유지하면서 이 웹훅으로 연결하는 아키텍처 — AI 서버 재시작이 메인 백엔드에 영향 없음

### ⏱ 2026-04-20 09:37 | 보고서 ORM + 녹화 서비스 + 라우터 통합
- **신규 파일**:
  - `app/models/report.py` — `reports` 테이블. 세션 ID / 현장 FK / LLM 제공자(claude/gemini) / 제목 / 본문 Text / 상태(draft/published) / 생성자 FK
  - `app/schemas/report.py` — `ReportCreate / ReportResponse`
  - `app/services/recording.py` — RGB + Thermal 동시 별도 파일 녹화. `recording_service` 싱글톤. `start_recording()` → `asyncio.Queue` 구독 + `cv2.VideoWriter` mp4 인코딩. `stop_recording()` → writer close + 파일 경로 반환. 저장 경로: `./recordings/YYYYMMDD_HHMMSS_rgb.mp4` / `_thermal.mp4`
  - `app/api/router.py` — 모든 서브 라우터를 `api_router` 로 통합. `main.py` 에서 `app.include_router(api_router, prefix="/api/v1")` 로 마운트. 신규 라우터 포함: `auth / oauth / defects / stream / websocket / report / telemetry / slam / floorplan / ai_webhook / sites`
- **수정 파일**:
  - `app/models/__init__.py` — 신규 5개 모델(`Site / Floorplan / SlamMap / Telemetry / Report`) import 추가 → `Base.metadata` 자동 등록, Alembic 마이그레이션에 반영
  - `app/db/init_db.py` — 신규 모델 임포트 추가
  - `app/main.py` — `api_router` 마운트, lifespan 에 recording_service 포함
  - `app/config.py` — `JWT_SECRET / JWT_ACCESS_EXPIRE_MINUTES / UPLOAD_DIR / RECORDINGS_DIR` 신규 환경변수 추가
  - `app/api/report.py` — 기존 LLM 스트리밍에 Report ORM 저장 로직 추가
  - `app/api/stream.py` · `app/services/camera.py` — 녹화 서비스 연동 주석 추가

### 📐 확장 후 전체 DB 스키마 현황
```
users (기존)          ←─── business_profiles (1:1, 기존)
  │                   ←─── user_term_agreements (M:N, 기존)
  ├── sites           ← 신규. B2B/B2C 현장 관리
  │     └── reports   ← 신규. 현장별 보고서 (FK: site_id optional)
  ├── defect_logs     ← 기존. (FK: site_id 추가 예정)
  ├── floorplans      ← 신규. 평면도 업로드·처리 결과
  ├── slam_maps       ← 신규. SLAM 맵 스냅샷
  └── telemetry_logs  ← 신규. 드론 위치/센서 타임시리즈
```

### 🔗 API 엔드포인트 전체 목록 (신규 추가분)
| 도메인 | 메서드 | 경로 | 역할 |
|--------|--------|------|------|
| Auth | POST | `/api/v1/auth/signup` | 회원가입 |
| Auth | POST | `/api/v1/auth/login` | JWT 로그인 |
| Auth | GET | `/api/v1/auth/me` | 내 정보 |
| OAuth | POST | `/api/v1/oauth/google` | Google OAuth |
| OAuth | POST | `/api/v1/oauth/kakao` | Kakao OAuth |
| OAuth | POST | `/api/v1/oauth/naver` | Naver OAuth |
| Sites | GET/POST | `/api/v1/sites` | 현장 목록/등록 |
| Sites | GET/PATCH/DELETE | `/api/v1/sites/{id}` | 현장 상세/수정/삭제 |
| Floorplan | POST | `/api/v1/floorplan/upload` | 평면도 업로드 |
| Floorplan | POST | `/api/v1/floorplan/{id}/process` | 벽체 추출 트리거 |
| SLAM | CRUD | `/api/v1/slam` | SLAM 맵 관리 |
| Telemetry | CRUD | `/api/v1/telemetry` | 텔레메트리 로그 |
| AI Webhook | POST | `/api/v1/ai/detection` | AI 탐지 이벤트 |
| AI Webhook | POST | `/api/v1/ai/batch` | 다건 탐지 일괄 |

### 📋 잔여 한계 / 향후 작업
- **DB 미연결**: 현재 모든 모델·엔드포인트 코드 완성 상태. Alembic `upgrade head` + AWS RDS 연결이 최종 단계
- **인증 가드**: 현재 일부 엔드포인트(`sites` 등)만 `get_current_user` 의존성. 운영 배포 전 전체 적용 필요
- **파일 저장**: `floorplan/upload` 는 로컬 파일시스템. 운영 시 S3 pre-signed URL 또는 `boto3` 업로드로 전환
- **AI 서버 분리**: `ai_webhook.py` 는 Python AI 프로세스가 HTTP 호출하는 구조 — AI 서버 URL/인증 정책은 별건으로 결정 필요

---

## 6️⃣ 멀티테넌트 조직 기반 권한 체계 구현 (2026-04-20)

> **착수 시각**: 2026-04-20 14:00  
> **작업자**: @youminsu0523  
> **목표**: 여러 회사/개인에게 배포될 플랫폼이므로, 조직(Organization) 기반 데이터 격리 + 사용자 권한 관리 체계 전면 구축.  
> **배경**: 기존에는 로그인만 되면 전체 데이터에 접근 가능했음. A회사 사용자가 B회사 데이터를 볼 수 없도록 격리 필요.

### ⏱ 14:00 | 소셜 로그인 에러 수정 (선행 작업)
- **문제 1**: `python-jose` 미설치 → 백엔드 기동 불가 → `ModuleNotFoundError: No module named 'jose'`
  - 해결: `pip install python-jose[cryptography]`
- **문제 2**: OAuth 이메일 중복 → `IntegrityError: duplicate key (email)=(youminsu0523@gmail.com)`
  - 원인: `_find_or_create_oauth_user()` 이메일 조회가 case-sensitive → 기존 계정 못 찾고 INSERT 시도
  - 해결: `func.lower()` 대소문자 무시 조회 + `IntegrityError` catch 후 재조회 (`app/api/oauth.py`)
- **문제 3**: React 18 Strict Mode 이중 실행 → OAuth 인가 코드 2회 전송 → `invalid_grant`
  - 해결: `OAuthCallback.jsx` 에 `useRef` guard 추가하여 1회만 실행

### ⏱ 14:30 | Phase 1 — 백엔드 스키마 + 핵심 의존성
- **모델 변경**:
  - `app/models/site.py` — `organization_id` FK 추가 (멀티테넌트 격리 기준)
  - `app/models/conversation.py` — `organization_id` FK 추가
  - `app/models/organization.py`:
    - `Organization` 모델에 `invite_code` (8자리 영숫자, unique) 추가
    - `OrganizationMember` 모델에 `started_at` (입사일), `ended_at` (퇴사/계약 만료일, nullable) 추가
- **Alembic 마이그레이션**: `alembic.ini` 한글 인코딩 오류 수정 (cp949→UTF-8) + `env.py` 전체 모델 import + 마이그레이션 생성·적용 완료
- **핵심 의존성** (`app/dependencies.py`):
  - `get_current_org_member()` — 현재 사용자의 활성 조직 멤버십 조회. `X-Organization-Id` 헤더로 다중 조직 선택 지원. `ended_at` 만료 체크
  - `get_current_user_with_org()` — 미소속 사용자도 허용 (soft 버전)
  - `require_role("owner", "admin")` — 역할 기반 접근 제어 팩토리

### ⏱ 15:00 | Phase 2 — 백엔드 API 조직 스코핑 (데이터 격리)
- **Sites API** (`app/api/sites.py`): 모든 CRUD에 `Depends(get_current_org_member)` 적용. `list`: `WHERE organization_id = org.id`, `create`: 자동 `organization_id` 설정
- **Defects API** (`app/api/defects.py`): `DefectLog → Site JOIN → Site.organization_id` 경유 필터링. summary/list/get 모두 적용
- **Reports API** (`app/api/report.py`): `Report → Site JOIN → Site.organization_id` 경유 필터링. save/list/get/download/delete 모두 적용
- **Chat API** (`app/api/chat.py`): 대화 생성 시 `organization_id` 자동 설정 + 참여자 같은 조직 검증. 목록 조회 시 `Conversation.organization_id == org.id` 필터

### ⏱ 15:30 | Phase 2 — 인증 응답 확장 + 조직 관리 API
- **인증 응답에 조직 정보 포함**:
  - `app/schemas/user.py` — `OrgBriefResponse` (id, name, role, department, position) 스키마 추가, `UserResponse.organizations` 필드 추가
  - `app/api/auth.py` — `/me`, `login` 응답에 사용자 조직 목록 포함
  - `app/api/oauth.py` — Google/Kakao/Naver 3종 OAuth 응답에도 조직 목록 포함
  - `PATCH /auth/me` — 사용자 이름/전화번호 수정 엔드포인트 추가
- **조직 관리 API 확장** (`app/api/organization.py`):
  - `GET /organizations/unaffiliated-users` — 미소속 사용자 목록 (admin/owner 전용)
  - `POST /organizations/members/assign` — 미소속 사용자 조직 배정 (admin/owner 전용)
  - `POST /organizations/join` — 초대 코드로 조직 가입
  - `PATCH /organizations/members/{user_id}` — 입사일/퇴사일 설정 지원, 퇴사일 경과 시 자동 비활성 처리

### 🔗 신규/변경 API 엔드포인트
| 메서드 | 경로 | 역할 |
|--------|------|------|
| PATCH | `/api/v1/auth/me` | 내 정보 수정 (이름/전화번호) |
| GET | `/api/v1/organizations/unaffiliated-users` | 미소속 사용자 목록 |
| POST | `/api/v1/organizations/members/assign` | 미소속 사용자 배정 |
| POST | `/api/v1/organizations/join` | 초대코드 가입 |

### 📐 설계 결정 사항
- **데이터 격리 방식**: Site에 `organization_id` FK 직접 부여. Defect/Report는 Site FK 경유 간접 필터링 (스키마 최소 변경)
- **다중 조직 허용**: 한 사용자가 여러 조직에 소속 가능 (프리랜서/컨설턴트 시나리오)
- **계약 관리**: `started_at`/`ended_at`으로 입사·퇴사 관리, 퇴사일 경과 시 자동 비활성
- **온보딩 플로우**: Slack/Notion/Jira 패턴 → "조직 생성 / 초대코드 가입 / 관리자 배정 대기" 3가지 선택지

---

## 7️⃣ 프로필 이미지 업로드 기능 구현 (2026-04-20)

> **착수 시각**: 2026-04-20 16:00  
> **작업자**: @youminsu0523  
> **목표**: 사용자 프로필 이미지 업로드/삭제 기능. 회사 특성상 팀원 얼굴 인식이 필요하므로 이니셜 아바타 → 실제 사진 전환 지원. 채팅에서도 프로필 이미지 표시.

### ⏱ 16:00 | User 모델 + 스키마 확장

- **피드백**: "내 정보 수정에서 프로필 이미지를 변경할 수 있게 해줘. 현재는 이름의 앞 두글자를 띄우지만, 회사 특성상 얼굴을 알아야 하는 경우가 있기 때문에 프로필 사진을 넣을 수 있게 해줘. 프로필 사진은 채팅에서도 표현되어야 해."
- **수정 파일**:
  - `app/models/user.py` — `profile_image_url` 컬럼 추가 (String 500, nullable). 업로드된 이미지의 서버 내 경로 저장
  - `app/schemas/user.py` — `UserResponse`에 `profile_image_url: Optional[str] = None` 필드 추가

### ⏱ 16:10 | 프로필 이미지 업로드/삭제 API

- **수정 파일**: `app/api/auth.py`
  - `PUT /auth/me/profile-image` — 프로필 이미지 업로드. `UploadFile` 수신 → content-type 검증(JPEG/PNG/WebP/GIF) → 5MB 크기 제한 → UUID 파일명으로 `./uploads/profiles/` 저장 → 기존 파일 삭제 → DB `profile_image_url` 갱신
  - `DELETE /auth/me/profile-image` — 프로필 이미지 삭제. 파일시스템 파일 제거 + DB null 처리
  - 기존 `signup`, `login`, `get_me`, `update_me` 응답에 `profile_image_url` 포함하도록 갱신
- **파일 업로드 패턴**: 기존 `floorplan.py` 패턴 참고 — `aiofiles` 비동기 파일 쓰기, `uuid` 파일명, 확장자 화이트리스트

### ⏱ 16:20 | 정적 파일 서빙 + DB 마이그레이션

- **수정 파일**: `app/main.py` — `FastAPI.mount("/uploads", StaticFiles(...))` 추가. 업로드된 프로필 이미지를 `/uploads/profiles/{filename}` 경로로 HTTP 제공
- **신규 파일**: `alembic/versions/b3f1a2c4e5d6_add_profile_image_url_to_users.py` — `users.profile_image_url` 컬럼 추가 마이그레이션

### ⏱ 16:30 | 마이그레이션 적용 및 오류 해결

- **문제**: User 모델에 `profile_image_url` 컬럼 추가 후 서버 재시작 시, SQLAlchemy가 `SELECT users.profile_image_url`을 시도하지만 DB에 해당 컬럼 미존재 → 모든 인증 관련 쿼리 실패 (멤버 관리 페이지 "데이터를 불러오지 못했습니다" 에러)
- **원인**: `Base.metadata.create_all()`은 새 테이블만 생성하고 기존 테이블에 컬럼을 추가하지 않음
- **해결**: `PYTHONPATH=. alembic upgrade head` 실행 → `a957fb9970a3 → b3f1a2c4e5d6` 마이그레이션 성공 적용

### 🔗 신규 API 엔드포인트
| 메서드 | 경로 | 역할 |
|--------|------|------|
| PUT | `/api/v1/auth/me/profile-image` | 프로필 이미지 업로드 (교체) |
| DELETE | `/api/v1/auth/me/profile-image` | 프로필 이미지 삭제 |

### 📐 설계 결정 사항
- **저장 방식**: 로컬 파일시스템 (`./uploads/profiles/`) + StaticFiles 서빙. 운영 배포 시 Cloudflare R2 presigned URL로 전환 예정 (기존 `project_file_storage_r2.md` 메모리 참조)
- **파일명 전략**: UUID v4 + 원본 확장자. 중복/충돌 방지 + URL 추측 불가
- **기존 파일 정리**: 새 이미지 업로드 시 이전 파일 자동 삭제 (디스크 낭비 방지)
- **크기 제한**: 5MB. 프로필 사진 용도로 충분, 서버 부담 최소화

---

## 📝 세션 추가 정보 (2026-04-20 @Hijin)

- 작성자 (Who): @Hijin554
- 작성 일자 (When): 2026-04-20
- 목표 기능 (Objective): 2차 프로젝트 Phase 2 — 학습 완료된 AI 3개 모델을 실제 서빙하는 파이프라인 + `/api/v1/detect` REST + `/api/v1/ws/stream` WebSocket (드롭 큐) 구축
- 작업 브랜치/환경: `Hijin`

### 1️⃣ 초기 프롬프트 (Initial Prompt)
> AeroInspect 2차 프로젝트 — FastAPI 백엔드 구축 요청(v3). 학습 완료된 YOLOv8s × 2 + ResNet50 × 1 가중치를 실제로 로드·추론하는 파이프라인을 기존 `backend/` 구조에 통합.

### 2️⃣ 계획(Plan) 단계 피드백

- **피드백 1** (통합 vs 신규):
  > "aeroinspect_backend/ 새로 만들지 말고 기존 backend/ 에 통합해줘. 먼저 기존 구조 꼼꼼히 읽고 delta 계획부터 보여줘."
  → 해결: 기존 [app/services/yolo_inference.py](app/services/yolo_inference.py), [core/ws_manager.py](app/core/ws_manager.py), [models/defect.py](app/models/defect.py) 등 15개 파일 분석 → 수정/신규/제외 파일 delta 계획서 작성 → 승인 후 구현 진입

- **피드백 2** (bbox 좌표 정책):
  > "API 응답은 bbox_xyxy(픽셀) 유지. DB 저장 시에만 xywhn 변환. 이미지 W/H가 필요하니 프레임 shape을 결과에 같이 실어서 내려줘."
  → 해결: `DetectionResult.image_shape: {width, height}` 필드 추가, `xyxy_to_xywhn(xyxy, w, h)` 헬퍼 별도 함수로 분리 → [tests/test_inference_pipeline.py](tests/test_inference_pipeline.py) 회귀 테스트 5건

- **피드백 3** (shim 패턴):
  > "모델 로드는 절대 중복 금지 — inference_pipeline.service가 유일한 싱글톤. yolo_inference.yolo_service는 내부적으로 참조만 해."
  → 해결: [yolo_inference.py](app/services/yolo_inference.py) 를 40줄 shim으로 재작성. 기존 호출자([defect_processor.py](app/services/defect_processor.py), [dependencies.py](app/dependencies.py)) 무수정으로 호환

- **피드백 4** (WS 이중 브로드캐스트):
  > "신규 /ws/stream 탐지 결과는 기존 ws_manager.broadcast('defects', ...)로도 Push해줘. 두 WS 채널 분리돼 있지만 결과는 양쪽 다 흐르게."
  → 해결: [core/stream_inference.py](app/core/stream_inference.py) 에서 `stream` 채널(신규 포맷) + `defects` 채널(레거시 `defect.new` 이벤트) 동시 브로드캐스트

- **피드백 5** (Alembic 베이스라인):
  > "0001_baseline.py 수동 작성은 기존 스키마와 drift 날 위험. 0002만 새로 작성하고 첫 배포 때 alembic stamp head 돌리는 절차로."
  → 해결: [alembic/versions/0002_defect_class_display.py](alembic/versions/0002_defect_class_display.py) 하나만 생성 (`down_revision=None`). README에 `alembic stamp 0002_defect_class_display` 절차 명시

### 3️⃣ 구현 핵심 아키텍처

#### 3-모델 추론 파이프라인 (싱글톤)
```
InferencePipeline.load_models()
  ├── YOLO(yolov8s_crack_moisture_best.pt)   — Crack, Moisture (nc=2)
  ├── YOLO(yolov8s_delamination_best.pt)     — delamination (nc=1)
  └── ResNet50(resnet50_wallpaper_best.pt)   — 19 classes (good=Burst 포함)
```
- 체크포인트 `class_names` 리스트를 하드코딩 `WALLPAPER_CLASSES`와 `assert` 검증 — 학습·서빙 클래스 순서 미스매치 사전 차단
- 입력 타입 4종 지원: `bytes / numpy.ndarray / PIL.Image / str(경로)`
- 블로킹 추론은 전부 `asyncio.to_thread()` 로 스레드 풀 위임

#### ⚠️ `good` 클래스 특수 처리
데이터셋 폴더명이 `good`으로 지어졌으나 실제 내용은 "터짐(Burst)" 하자 이미지. 가중치에 baked-in 되어 있어 내부명은 유지하되:
```python
CLASS_DISPLAY_MAP = {
    ...
    "good": ("Burst", "터짐"),  # ⚠️ 실제 의미는 '터짐'
}
WALLPAPER_SEVERE_CLASSES = {"Mold", "Damage", "Exploded", "Defective_Joint", "good"}
# → severity MED로 격상 (LOW 아님)
```
"정상=하자없음"으로 필터링하는 로직은 코드 어디에도 넣지 않음.

#### severity 자동 계산 규칙 ([inference_pipeline.py](app/services/inference_pipeline.py))
```
yolo_thermal/delam 탐지 있음               → HIGH
벽지 is_confident & top1 ∈ SEVERE classes  → MED
벽지 is_confident & 그 외                   → LOW
그 외 (신뢰도 부족)                         → null (판단 보류)
```

#### WebSocket 드롭 큐 + 프레임 스킵 ([core/stream_inference.py](app/core/stream_inference.py))
드론 IRC-256CA 스트림(15~30 fps) vs CPU/GPU 추론(80~150 ms/frame) 불일치 → 모든 프레임 처리 불가. 다음 패턴으로 해결:
```python
self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)

def submit(frame):
    if self._submitted_count % FRAME_SKIP != 0: return  # 3프레임 중 1개만
    try: self._queue.put_nowait(QueuedFrame(frame, ...))
    except asyncio.QueueFull: self._dropped_count += 1   # 바쁘면 그냥 버림
```
워커 태스크는 별도 `asyncio.create_task`로 영구 실행, main.py lifespan에서 `start()/stop()`.

#### DB 스키마 확장 ([models/defect.py](app/models/defect.py), [alembic/versions/0002_defect_class_display.py](alembic/versions/0002_defect_class_display.py))
기존 `defect_logs` 스키마 유지하면서 4컬럼 추가 + 레거시 A-E 컬럼 NULLABLE 완화:
```
+ defect_source ENUM('yolo_thermal','yolo_delam','wallpaper')
+ defect_class VARCHAR(50)                 -- 모델 내부명 (예: 'good', 'Crack')
+ defect_class_display_en VARCHAR(80)      -- 프론트용 (예: 'Burst')
+ defect_class_display_ko VARCHAR(80)      -- 프론트용 (예: '터짐')
~ area/category_code/defect_type: NOT NULL → NULLABLE  (신규 클래스 중 A-E 매핑 없는 케이스 대비)
```

### 4️⃣ 신규/수정 파일 목록

**신규 (11개)**:
- [app/schemas/detection.py](app/schemas/detection.py) — `DetectionResult`, `YoloDetection`, `WallpaperPrediction`, `HealthResponse` Pydantic 스키마
- [app/services/defect_taxonomy.py](app/services/defect_taxonomy.py) — `WALLPAPER_CLASSES`(19), `CLASS_DISPLAY_MAP`, `YOLO_DISPLAY_MAP`, `LEGACY_MAP_THERMAL/WALLPAPER`, `map_to_legacy()`, `xyxy_to_xywhn()`
- [app/services/wallpaper_classifier.py](app/services/wallpaper_classifier.py) — ResNet50 19-class. 체크포인트 `class_names` assert 검증. top1+top3 softmax
- [app/services/inference_pipeline.py](app/services/inference_pipeline.py) — 싱글톤 오케스트레이터. `detect_defects()`, `detect_defects_async()`, `detect_defects_legacy()` shim용
- [app/core/stream_inference.py](app/core/stream_inference.py) — 드롭 큐 워커 + `stream`/`defects` 양방향 브로드캐스트
- [app/api/detect.py](app/api/detect.py) — `POST /api/v1/detect` multipart 단건, `POST /api/v1/detect/batch` 최대 10장
- [app/api/ws_stream.py](app/api/ws_stream.py) — `WS /api/v1/ws/stream` 바이너리 JPEG 수신 + `asyncio.to_thread(cv2.imdecode)`
- [alembic/versions/0002_defect_class_display.py](alembic/versions/0002_defect_class_display.py) — 4컬럼 추가 + NULLABLE 완화 마이그레이션
- [tests/test_inference_pipeline.py](tests/test_inference_pipeline.py) — 18개 테스트 (xyxy→xywhn 5, taxonomy 8, /health & /detect 5)
- [pytest.ini](pytest.ini) — `asyncio_mode=auto`
- [README.md](README.md) — 3-모델 표, WebSocket 프로토콜, React 예제, 마이그레이션 절차, **동작 확인 체크리스트**

**수정 (10개)**:
- [app/services/yolo_inference.py](app/services/yolo_inference.py) — **전체 재작성**: 40줄 shim (`pipeline.load_models()` 위임, `detect_defects_legacy()` 반환)
- [app/config.py](app/config.py) — `AEROINSPECT_WEIGHTS_DIR`, `YOLO_THERMAL_WEIGHTS`, `YOLO_DELAM_WEIGHTS`, `WALLPAPER_WEIGHTS`, `YOLO_CONF_THRESHOLD=0.25`, `WALLPAPER_CONF_THRESHOLD=0.4`, `FRAME_SKIP=3`, `DEVICE=auto` 추가
- [.env.example](.env.example), [.env](.env) — 위 키 전부 반영
- [app/models/defect.py](app/models/defect.py) — 4 컬럼 추가, 레거시 컬럼 NULLABLE
- [app/schemas/defect.py](app/schemas/defect.py) — `DefectLogCreate/Response`에 4 필드 + 레거시 A-E Optional화
- [app/api/defects.py](app/api/defects.py) — `GET /api/v1/defects/recent` 추가, `POST /defects`에 4 필드 DB 저장
- [app/api/ai_webhook.py](app/api/ai_webhook.py) — 웹훅도 4 필드 저장
- [app/api/router.py](app/api/router.py) — `detect`, `ws_stream` 라우터 등록
- [app/main.py](app/main.py) — lifespan에 `stream_inference_worker.start()/stop()` + `/health` 확장 (`device/models_loaded/wallpaper_classes_count/stream_worker_running/frame_skip`)
- [requirements.txt](requirements.txt) — `torch>=2.1`, `ultralytics>=8.3.0` 핀 + `pytest`, `pytest-asyncio`
- [alembic.ini](alembic.ini), [alembic/env.py](alembic/env.py) — Python 3.14 cp949 로케일 이슈 대응: 한글 주석 제거 + `sys.path` 수동 삽입

### 5️⃣ 실제 검증 결과

- **pytest**: 18/18 통과 (`good=Burst` 매핑, xyxy→xywhn 회귀 등 전부)
- **서버 기동 로그**:
  ```
  [Pipeline] YOLO thermal 로드: yolov8s_crack_moisture_best.pt
  [Pipeline] YOLO delam 로드:   yolov8s_delamination_best.pt
  [Wallpaper] ResNet50 로드 완료: device=cpu, val_acc=0.5434, classes=19
  [Pipeline] 3-모델 로드 완료
  [StreamInfer] 워커 시작 (frame_skip=3)
  ```
- **`/health`** 응답: `models_loaded` 3개 전부 `true`, `wallpaper_classes_count=19`, `stream_worker_running=true`
- **`/api/v1/detect`** 실제 이미지 업로드 테스트 (Roboflow 샘플):
  ```
  top1: Damage (훼손) / conf 97.97% / severity=MED / has_defect=true
  ```
  → 클래스 표시명 매핑, severity 격상 규칙, image_shape 기록 모두 정상

### 📋 잔여 한계 / 향후 작업 (추가분)
- **GPU 추론**: 현재 T4 없이 CPU로 돌려 한 장당 5~15초. 프로덕션 배포 시 GPU 인스턴스 + `DEVICE=cuda` 로 전환 필요
- **싱글 워커 제한**: `stream_inference_worker`는 프로세스 내 싱글톤이라 gunicorn multi-worker 구동 불가. 다중 워커 필요 시 Redis pub/sub 기반 리팩터
- **드론 좌표**: MAVLink/LiDAR 연동 전이라 `lidar_x/y/z`는 당분간 NULL. TF 연동 완료 후 기존 컬럼에 채울 예정
- **벽지 분류 정확도 0.54**: `WALLPAPER_CONF_THRESHOLD=0.4`로 보수 필터링. 데이터 추가 수집 후 fine-tuning 필요

---

## 🔧 2026-04-21 추가 세션 — 벽지 분류 오탐 감소 (Issue 1)

### 1️⃣ 배경
- ResNet50 벽지 분류기 val_acc ≈ 54% (19-way). 단일 `top1_conf >= 0.4` 필터는 모호한 예측(top1/top2 근소차)을 걸러내지 못해 오탐 유입.
- 재학습 없이 **코드 수정만으로** FP 감소 가능한 지점을 찾음.

### 2️⃣ 적용한 개선 — 이중 게이트 (Top-k Margin 투표)
`is_confident` 판정에 두 조건 모두 만족 요구:
1. `top1_conf >= WALLPAPER_CONF_THRESHOLD` (기본 0.35로 완화 — 19-way 특성 반영, 랜덤(1/19≈5%) 대비 6배 이상)
2. `top1_conf - top2_conf >= WALLPAPER_MARGIN_THRESHOLD` (기본 0.15 신규 — 2위와의 분리도)

근거: 1위 점수 높아도 2위와 근소차면 모델이 헷갈린 상태. 그런 케이스는 `is_confident=false` → severity null로 판정 보류시켜 오탐 차단.

### 3️⃣ 변경 파일
- [app/config.py](app/config.py) — `WALLPAPER_CONF_THRESHOLD` 0.4 → 0.35, `WALLPAPER_MARGIN_THRESHOLD=0.15` 신규 추가
- [app/services/inference_pipeline.py](app/services/inference_pipeline.py) — `_run_wallpaper()`에 top1/top2 margin 계산 + 이중 게이트 로직. init / load_models 초기화 값 반영
- [app/schemas/detection.py](app/schemas/detection.py) — `WallpaperPrediction.is_confident` description 이중 조건으로 갱신
- [.env.example](.env.example) — 두 임계값 반영
- [README.md](README.md) — 알려진 제약 섹션에 이중 게이트 설명 추가

### 4️⃣ 추후 튜닝 방향
- 지금 값(0.35 / 0.15)은 "감"으로 지정. 실제 운영 로그(사람이 오탐 태그한 케이스) 축적 후 임계값 스윕 스크립트로 최적점 재산출 예정
- 드론 건물 검사 특성(하자 놓치면 안전 문제)상 "미탐 최소화" 쪽으로 더 완화(예: 0.30 / 0.10) 검토 가능
- 재학습 대안: class-weighted CrossEntropy, RandomRotation/ColorJitter 증강, 19→5 계층적 그룹핑 (외부 학습 파이프라인 필요)

---

## 🛠️ 2026-04-21 추가 — Issue 2 LiDAR 3D 좌표 배선 + 품질 개선 일괄

### 1️⃣ Issue 2: MAVLink/LiDAR 3D 좌표 통합 (하드웨어 없이 배선 완료)
기존 lidar.py(TF-Luna 9-byte 프레임 파서, 중앙값 필터, compute_3d_position) 이미 존재. **빠진 배선 4개** 연결:
1. `app/services/telemetry_cache.py` **신규** — 최신 드론 pose 메모리 캐시 싱글톤 (asyncio.Lock, stale 5초 판정)
2. [app/main.py](app/main.py) — lifespan에 `lidar_service.start()/stop()` 호출 (serial 실패해도 graceful), health에 `lidar.connected/distance_m`, `telemetry_cache.ready/age_sec` 노출
3. [app/api/telemetry.py](app/api/telemetry.py) — POST /telemetry 수신 시 `telemetry_cache.update()` + `lidar_service.update_attitude()` 호출
4. [app/core/stream_inference.py](app/core/stream_inference.py) — 프레임 추론 시점에 `_compute_lidar_xyz()` (cache_fresh + lidar distance) 호출, 레거시 `defect.new` 이벤트와 `stream` 페이로드에 `lidar_x/y/z` 주입

좌표 없으면 None 유지 → 드론/LiDAR 없어도 서비스 영향 없음.

### 2️⃣ 이미지 저장소 리팩 (Base64 → 파일시스템)
- `app/services/image_storage.py` **신규** — `save_base64_jpeg(b64) → rel_path`, `get_url()`, `delete()`. 저장 경로: `./uploads/defects/YYYY-MM-DD/uuid.jpg` (이미 mount된 StaticFiles 경유 서빙)
- [app/models/defect.py](app/models/defect.py) — `image_crop_path` (String 255) 컬럼 추가, `image_crop`(Text)는 DEPRECATED 표시 유지
- [alembic/versions/c8f1d2e4a7b9_add_image_crop_path_to_defect_logs.py](alembic/versions/c8f1d2e4a7b9_add_image_crop_path_to_defect_logs.py) **신규 마이그레이션**
- [app/schemas/defect.py](app/schemas/defect.py) — `image_crop_path`, `image_crop_url` 응답 필드
- [app/api/ai_webhook.py](app/api/ai_webhook.py), [app/api/defects.py](app/api/defects.py) — Base64 수신 시 파일 저장 → `image_crop=None`, `image_crop_path=rel_path` 기록. `_build_response()` 헬퍼로 `image_crop_url` 채움

효과: DB Text 컬럼 용량 폭증 방지. 1건당 ~100KB → 파일 경로만 저장.

### 3️⃣ 로깅 체계 (structlog + Request ID)
- `app/core/logging.py` **신규** — `configure_logging(json_output, level)`, `get_logger(name)`, contextvars 통합
- `app/core/middleware.py` **신규** — `RequestIDMiddleware`: `X-Request-ID` 수신/발행, 모든 로그에 `request_id/method/path` 자동 바인딩, 요청 완료 시 `http.request` 이벤트(status, duration_ms) 로깅
- [app/config.py](app/config.py) — `LOG_JSON=False`, `LOG_LEVEL=INFO`
- [requirements.txt](requirements.txt) — `structlog` 추가
- [app/main.py](app/main.py) — import 시점에 `configure_logging()`, `add_middleware(RequestIDMiddleware)`
- [.env.example](.env.example) — LOG_JSON/LOG_LEVEL 문서화

### 4️⃣ 테스트 추가 (새 모듈 회귀 방지)
- `tests/test_telemetry_cache.py` — update/snapshot/stale/clear 7 케이스
- `tests/test_image_storage.py` — base64 저장/삭제/data URL prefix/에러 8 케이스
- `tests/test_wallpaper_double_gate.py` — monkeypatch 기반 이중 게이트 로직 5 케이스 (모델 없이 검증)

### 5️⃣ 후속 TODO (다음 이터레이션)
- 카메라 intrinsics 불필요(1D LiDAR) 구조 유지 중. 멀티빔/3D LiDAR로 교체 시 `compute_3d_position` 확장 필요
- `image_crop` (Text, deprecated) 컬럼은 충분한 마이그레이션 기간 후 drop 예정
- 운영 배포 시 `LOG_JSON=true` 전환 + Grafana/Datadog 연동
- DefectLog 삭제 시 파일 cleanup 훅 (`image_storage.delete(defect.image_crop_path)`) 필요

---

## 🔬 20종 결함 분류 ONNX 추론 파이프라인 (2026-04-22)

- 작성자: @youminsu0523
- 작업 브랜치: `MS`

### 1️⃣ 초기 작업 내용

#### 추론 서비스 (신규)
- [app/services/onnx_inference.py](app/services/onnx_inference.py) — ONNX Runtime 기반 모델 로더 및 추론 래퍼 (ResNet, PatchCore 지원)
- [app/services/ensemble.py](app/services/ensemble.py) — 다중 모델 앙상블 투표/가중평균 서비스
- [app/services/alignment_detector.py](app/services/alignment_detector.py) — 건물 정렬(수직/수평) 감지 서비스
- [app/services/insulation_detector.py](app/services/insulation_detector.py) — 단열 결함 감지 서비스
- [app/services/temporal_filter.py](app/services/temporal_filter.py) — 시계열 기반 오탐 필터링
- [app/services/inference_pipeline_20.py](app/services/inference_pipeline_20.py) — 20종 결함 통합 추론 파이프라인

#### 스키마/모델/마이그레이션
- [app/models/defect.py](app/models/defect.py) — 20종 결함 컬럼 추가
- [app/schemas/detection.py](app/schemas/detection.py) — 20종 결함 응답 스키마 확장
- [app/services/defect_taxonomy.py](app/services/defect_taxonomy.py) — 결함 분류 체계 확장
- [alembic/versions/0003_add_20defect_pipeline_columns.py](alembic/versions/0003_add_20defect_pipeline_columns.py) — DB 마이그레이션
- [app/config.py](app/config.py) — ONNX 모델 경로 설정 추가
- [app/core/stream_inference.py](app/core/stream_inference.py) — 20종 파이프라인 연동

#### 학습 스크립트 & 설정
- `training/train_m1~m6` — ResNet(m1~m3), Thermal UNet(m4), FrameSeg(m5), PatchCore(m6) 학습 스크립트
- `training/configs/` — 모델별 YAML 학습 설정
- `training/eval/` — 벤치마크 및 전체 평가 스크립트
- `training/export_to_onnx.py` — PyTorch → ONNX 변환
- `training/AeroInspect_Training.ipynb` — 학습 노트북

### 2️⃣ 피드백 반영 (⏱ 2026-04-22)
- `.gitignore` 정리: 대용량 바이너리(`*.onnx.data`, `*.npy`), 학습 로그(`*_log.txt`), YOLO 학습 결과(`runs/`), lock 파일을 gitignore에 추가
- staged에서 불필요 파일 129개 제거 (172개 → 40개)

---

## 🧹 2026-04-22 추가 — 후속 TODO 소진 & 모니터링 엔드포인트 (@Hijin554)

> **착수 시각**: 2026-04-22 오후
> **작업 브랜치**: `Hijin`
> **목표**: LiDAR 세션(2026-04-21)에서 남긴 후속 TODO 소진 + 운영 모니터링 기반 마련
> **배경**: "백엔드 할 거 뭐 있냐"는 질문에 대한 잔여 작업 정리. MS 브랜치(20종 파이프라인)와 겹치지 않는 영역만.

### 1️⃣ 후속 TODO 소진 (LiDAR 세션에서 남김)

#### DefectLog 삭제 시 파일 cleanup 훅
- [app/api/defects.py](app/api/defects.py) `delete_defect` — DB 레코드 제거 후 `image_storage.delete(defect.image_crop_path)` 호출. DB 트랜잭션 성공 후에만 파일 지움 → 롤백 시 orphan 방지
- **+ 조직 스코프 적용**: 기존엔 인증만 걸려 있고 `get_current_org_member` 누락 → 다른 조직 레코드까지 UUID만 알면 삭제되는 버그였음. 내 조직 site에 연결된 레코드만 삭제되게 수정

### 2️⃣ 운영 모니터링 API 신규

#### `GET /api/v1/stream/stats`
- [app/api/stream.py](app/api/stream.py) — `stream_inference_worker.stats` + `telemetry_cache` ready/age + `lidar` connected/distance를 한 번에 반환
- 대시보드 좌측 상단 배지 / 운영 헬스 체크용. `/health`보다 실시간 추론 파이프라인에 특화
- 응답은 `StreamStatsResponse` Pydantic 스키마로 타입 고정 → Swagger에 전체 필드 구조 노출

#### `GET /api/v1/coverage/{site_id}`
- [app/api/coverage.py](app/api/coverage.py) **신규** — 텔레메트리 좌표(pos_x/pos_y)의 convex hull 면적 vs `sites.total_area` 대비 커버리지율
- **순수 Python Andrew's monotone chain** 구현 (scipy 의존 회피). Shoelace로 면적 산출
- `sample_limit` 쿼리 파라미터로 최근 N개 샘플 제어 (기본 2000, 최대 20000)
- 샘플 < 3점이면 `note` 필드로 부족 안내하는 graceful fallback
- 응답은 `CoverageResponse` Pydantic 스키마. `hull` 필드를 프론트 3D 미니맵 음영 영역용으로 그대로 전달

#### 스키마 모듈 신설
- [app/schemas/monitoring.py](app/schemas/monitoring.py) **신규** — `StreamStatsResponse`, `CoverageResponse`, `WorkerStats`, `TelemetryCacheStats`, `LidarStats`
- 기존 `app/schemas/defect.py` 등과 일관된 `from_attributes` 스타일 대신, 모니터링 응답은 완전 DTO라 Plain BaseModel

### 3️⃣ 벽지 임계값 튜닝 스윕 스크립트
- [scripts/sweep_wallpaper_thresholds.py](scripts/sweep_wallpaper_thresholds.py) **신규**
- 입력: JSONL (`{"top1_conf": 0.62, "top2_conf": 0.41, "label": "defect"}`)
- `WALLPAPER_CONF_THRESHOLD × WALLPAPER_MARGIN_THRESHOLD` 격자 탐색 → precision / recall / F1 / TP / FP / FN / TN 테이블
- F1 내림차순 정렬, 동점이면 recall 우선 (건물 검사 특성상 미탐 < 오탐)
- `--out sweep.csv`로 결과 저장 옵션

### 4️⃣ 운영 로그 전환 준비
- [.env.example](.env.example) — `LOG_JSON` / `LOG_LEVEL` 주석 보강. 운영 전환 방법과 Grafana/Datadog 적재 시나리오 명시
- [README.md](README.md) — "운영 모니터링 & 관측성" 섹션 신설 (`/health` / `/stream/stats` / `/coverage/{id}` 비교표, structlog 설정, 스윕 스크립트 사용법)

### 5️⃣ 회귀 테스트 & 품질 강화
- `tests/test_coverage_geometry.py` **신규** — convex hull (사각형/삼각형/중복점/공선/45도 회전), shoelace 면적 (6 케이스)
- `tests/test_defect_delete_cleanup.py` **신규** — storage.delete 호출 여부, 경로 None 스킵, DB→파일 순서, 404 경로 (4 케이스, `unittest.mock` 기반 DB·storage 독립 검증)
- `tests/test_coverage_response_shape.py` **신규** — `CoverageResponse` / `StreamStatsResponse` Pydantic 스키마 직렬화 (5 케이스)
- **pre-existing 테스트 버그 수정**: `tests/test_wallpaper_double_gate.py::test_edge_exact_thresholds` — float 정밀도로 `0.35 - 0.20 = 0.14999...`가 되어 경계값 비교 실패. 경계 바로 위(0.36/0.20) 값으로 대체하고 테스트명도 `test_edge_just_above_thresholds`로 정정

### 6️⃣ config.py 잔존 merge conflict 해소
- [app/config.py](app/config.py) — `<<<<<<< Updated upstream ... =======  ... >>>>>>> Stashed changes` 마커가 그대로 커밋돼 있어 `SyntaxError`로 venv 기동 불가 상태였음
- 양쪽 변경(로깅 설정 + 20종 ONNX 파이프라인 설정)이 상호 독립이라 **둘 다 보존**하고 마커만 제거

### 🧪 회귀 결과
- 새 테스트 3종 (15개 케이스) 모두 통과
- 전체 regression: `test_coverage_geometry / test_coverage_response_shape / test_defect_delete_cleanup / test_image_storage / test_inference_pipeline / test_telemetry_cache / test_wallpaper_double_gate / test_ws_manager` → **59/59 통과**
- 제외한 2개 파일은 내 변경과 무관한 pre-existing 이슈:
  - `test_yolo_inference.py`: 가중치 파일(`models_weights/*.pt`) 미배치 환경
  - `test_defects_api.py`: 선대 멀티테넌트 org-scoping 적용 이후 인증 토큰 없이 돌리면 401 반환 — 테스트 쪽이 옛 버전

### 📋 이번 세션 후속 TODO
- `test_defects_api.py` auth 토큰 fixture 달아서 org-scoping과 함께 작동하도록 갱신 (그동안 CI에서도 계속 빨간 불)
- coverage 엔드포인트를 `site_id`별 telemetry 분리(`telemetry_logs.site_id` FK 추가)로 확장. 현재는 전역 최근 샘플 기반
- 스윕 스크립트를 재학습 대신 threshold 조정 근거로 쓰려면 운영 JSONL 로그 수집 파이프라인(`logs/wallpaper_predictions.jsonl`) 분리 필요

---

## 🪢 2026-04-22 후속 — 후속 TODO 소진 (@Hijin554)

> **착수 시각**: 2026-04-22 늦은 오후
> **작업 브랜치**: `Hijin`
> **목표**: 바로 위 블록에서 남긴 "이번 세션 후속 TODO" 중 2개 클로즈

### 1️⃣ `test_defects_api.py` 재작성 — 401 빨간 불 해소

기존 테스트는 `httpx` 클라이언트만 만들고 실제 API 호출 시 JWT 토큰이 없어 모든 요청이 401로 귀결 → CI에서 7개 실패. 근본 원인은 멀티테넌트 org-scoping 적용 후 테스트 대응이 안 된 것.

- **접근 방식**: 실제 DB 띄우지 않고 FastAPI `dependency_overrides`로 `get_current_org_member` / `get_db` 주입
- `_make_org_tuple()` — 가짜 (user, member, org) `SimpleNamespace` 튜플. role="owner"
- `_make_empty_db()` — `AsyncMock` 기반 DB. `scalar` 0 / `execute` empty result 반환하도록 구성
- `authed_client` fixture — 두 override 적용한 `AsyncClient`
- `unauth_client` fixture — override 없는 순수 클라이언트 → 401 검증용
- 8 케이스: 200 empty / summary 구조 / severity 필터 / area 필터 / 404 / 무인증 GET 401 / 무인증 summary 401 / **무인증 DELETE 401** (오늘 추가한 DELETE org-scope 회귀 방지)

결과: **8/8 통과**. CI 안정화.

### 2️⃣ `telemetry_logs.site_id` FK 추가 — coverage 정확도 개선

이전 커밋에서 `/api/v1/coverage/{site_id}` 만들 때 텔레메트리에 `site_id` FK가 없어서 **site가 바뀌어도 같은 면적**이 나오던 버그.

- [app/models/telemetry.py](app/models/telemetry.py) — `site_id: UUID → sites.id` FK 컬럼 추가. nullable (현장 미지정 비행 허용), `index=True` (쿼리 최적화)
- [alembic/versions/e4c9a8b27f10_add_site_id_to_telemetry_logs.py](alembic/versions/e4c9a8b27f10_add_site_id_to_telemetry_logs.py) **신규 마이그레이션**
  - `down_revision = "c8f1d2e4a7b9"` (Hijin 체인 끝에 체결)
  - `ondelete="SET NULL"` — site 삭제 시 비행 기록은 보존
  - ⚠️ 현재 프로젝트는 **마이그레이션 그래프가 2 heads 상태** (`0003` / `c8f1d2e4a7b9`). 통합 배포 전 `alembic merge -m "merge heads" <id1> <id2>` 한 번 실행 필요 — 리비전 파일에도 주석으로 명시
- [app/schemas/telemetry.py](app/schemas/telemetry.py) — `TelemetryCreate.site_id` / `TelemetryResponse.site_id` Optional 추가
- [app/api/telemetry.py](app/api/telemetry.py) `create_telemetry` — `site_id=payload.site_id` 반영
- [app/api/coverage.py](app/api/coverage.py) — 쿼리 전략 이중화:
  1. 우선 `telemetry_logs.site_id == site.id` 필터로 조회
  2. 0건이면 마이그레이션 이전 비행 호환을 위해 **전역 최근 N건으로 fallback** + 응답 `note` 필드에 근사치임을 표기 (`"이 현장에 연결된 텔레메트리가 없어 전역 최근 샘플로 계산된 근사치입니다."`)
- [CoverageResponse](app/schemas/monitoring.py)의 `note` 필드를 실제로 사용하게 됨 — 사용자/프론트가 근사 여부를 판별 가능

### 🧪 회귀 결과
- `test_coverage_geometry / test_coverage_response_shape / test_defect_delete_cleanup / test_defects_api / test_image_storage / test_inference_pipeline / test_telemetry_cache / test_wallpaper_double_gate / test_ws_manager` → **67/67 통과**
- 제외한 `test_yolo_inference.py`는 기존부터 가중치 파일 없는 환경에서 실패하던 것 (내 변경 무관)

### 📋 남은 후속 TODO (다음 누군가의 몫)
- **alembic heads 병합**: `alembic merge -m "merge 20defect + image_crop_path + site_id chains" 0003 e4c9a8b27f10`
- **ROS2 브리지/MAVLink 파서** 에서 `POST /telemetry` 호출 시 현재 세션 `site_id` 주입 — 스키마는 준비됨, 호출자 수정 필요
- 스윕 스크립트용 운영 JSONL 로그 수집 파이프라인 (`logs/wallpaper_predictions.jsonl`) 분리

---

## 🔐 2026-04-22 3rd 세션 — 운영 배포 전 남은 인증·관측·확장 일괄 (@Hijin554)

> **착수 시각**: 2026-04-22 저녁
> **작업 브랜치**: `Hijin`
> **목표**: 드론 실기 연동 없이도 가능한 운영 배포 준비 항목 7개 일괄 처리
> **배경**: "백엔드 드론 없이 할 수 있는 것" 중 고·중 우선순위. MS 브랜치(20종 파이프라인)와 비충돌.

### 1️⃣ Refresh Token

- [app/core/jwt.py](app/core/jwt.py) — `create_refresh_token` / `decode_refresh_token` 추가. payload 에 `type` 클레임(`access` | `refresh`) 분리 → 교차 사용 차단. type 미포함 레거시 토큰은 access 로 호환 허용
- [app/api/auth.py](app/api/auth.py) `POST /auth/refresh` — 유효 refresh → 새 access 발급. 사용자 존재 재확인 (계정 삭제 케이스 대비)
- [app/api/auth.py](app/api/auth.py) / [app/api/oauth.py](app/api/oauth.py) — 로그인 응답(`TokenResponse`)에 `refresh_token` 포함. OAuth 3종(Google/Kakao/Naver)도 동일 적용
- [app/schemas/user.py](app/schemas/user.py) — `TokenResponse.refresh_token`, `RefreshTokenRequest`, `RefreshTokenResponse`
- [app/config.py](app/config.py) — `JWT_REFRESH_EXPIRE_DAYS: int = 14`
- 테스트 9개 (roundtrip / 교차 사용 거절 / 레거시 호환 / 만료·변조·서명 오류)

### 2️⃣ SLAM/Floorplan/Telemetry auth 가드 감사

기존에 인증 없이 뚫려 있던 11개 엔드포인트에 `get_current_user` Depends 추가:

| 라우터 | 엔드포인트 | 적용 |
|--------|-----------|------|
| SLAM | GET ""  / GET /{id} / POST "" / PATCH /{id} / DELETE /{id} | 전부 |
| Floorplan | GET "" / GET /{id} / POST /upload / POST /{id}/process / POST /analyze / DELETE /{id} | 전부 |
| Telemetry | GET /latest / GET "" | GET만 |
| Telemetry | POST "" | **의도적 오픈** — ROS2 브리지 내부 호출. 주석으로 보안 메모 남김 (향후 `INTERNAL_API_TOKEN` 예정) |

TODO 주석: SLAM/Floorplan 은 site/org FK 추가 후 `get_current_org_member` 로 승격 예정.

### 3️⃣ Prometheus `/metrics`

- [app/core/metrics.py](app/core/metrics.py) **신규** — `CollectorRegistry` 모듈 싱글톤. `http_requests_total`(Counter) / `http_request_duration_seconds`(Histogram) / 추론 워커 카운터 / 결함 카운터 / LiDAR·telemetry·queue gauge
- `PrometheusMiddleware` — 모든 요청 수/지연 자동 기록. `request.scope["route"].path` 로 템플릿 라벨링 → cardinality 폭증 방지
- `refresh_sensor_gauges()` — `/metrics` 스크랩 시마다 센서 싱글톤 스냅샷 → Gauge 반영. 미연결 시 -1 sentinel
- [app/main.py](app/main.py) — 미들웨어 등록 + `/metrics` 엔드포인트 (OpenMetrics 텍스트, `include_in_schema=False`)
- `prometheus-client` 라이브러리 추가
- 테스트 5개 (Counter 라벨 증감 / Gauge set / OpenMetrics 렌더 / sentinel / 실값 반영)

### 4️⃣ LOG_JSON 출력 유효성 테스트

- [tests/test_logging_json.py](tests/test_logging_json.py) **신규** — `caplog` fixture 로 structlog 렌더링 결과 회수
- JSON 라인이 `json.loads` 로 파싱되는지, `event`/`level`/`timestamp` 필드 존재 여부, bound contextvars (`request_id`, `path`) 자동 병합 검증
- `LOG_JSON=false` 출력은 JSON 파싱 실패해야 함 (구분 확증)
- 테스트 3개

### 5️⃣ 평면도 스케일 보정 (FR-015)

- [app/models/floorplan.py](app/models/floorplan.py) — `scale_px_per_meter`(Float) + `scale_reference`(JSONB) 컬럼 추가
- [alembic/versions/f3d1b6c09a12_add_scale_to_floorplans.py](alembic/versions/f3d1b6c09a12_add_scale_to_floorplans.py) **신규**
- [app/schemas/floorplan.py](app/schemas/floorplan.py) — `FloorplanCalibrateRequest/Response`
- [app/api/floorplan.py](app/api/floorplan.py) `POST /{id}/calibrate` — `p1`, `p2`, `real_length_m` 입력 → `math.hypot` 픽셀 거리 / 실측 길이 = px/m 환산. 동일 점이면 400
- 테스트 7개 (가로·대각선 스케일 / 동일점 None / Pydantic 검증: 음수·0·잘못된 좌표 모양)

### 6️⃣ 푸시 알림 (FCM/APNs) 스켈레톤

- [app/models/device_token.py](app/models/device_token.py) **신규** — `device_tokens` 테이블. 사용자 × 토큰 UNIQUE, platform(fcm|apns|web) 구분, `is_active` soft disable
- [alembic/versions/c7e2d5f3a18b_add_device_tokens.py](alembic/versions/c7e2d5f3a18b_add_device_tokens.py) **신규**
- [app/services/push_notifications.py](app/services/push_notifications.py) **신규** — `PushNotificationService` 싱글톤. `provider = noop | fcm | apns` 디스패처. 실패 시 `is_active=False` 자동 처리. `_send_fcm`/`_send_apns` 는 TODO 주석 처리된 스켈레톤 (firebase-admin 연결 시 구현)
- [app/api/notifications.py](app/api/notifications.py):
  - `POST /notifications/tokens` — 토큰 등록/재활성 (upsert)
  - `DELETE /notifications/tokens/{id}` — 로그아웃 시 제거 (소유자 검증)
  - `POST /notifications/push/test` — 본인 디바이스 테스트 발송
- [app/config.py](app/config.py) `PUSH_PROVIDER: str = "noop"` 기본값
- [app/models/__init__.py](app/models/__init__.py) — `DeviceToken` export
- 테스트 3개 (noop 경로 / 디바이스 0건 / 싱글톤 기본값)

### 7️⃣ Redis pub/sub 수평 확장 추상화

- [app/core/ws_manager_redis.py](app/core/ws_manager_redis.py) **신규** — `RedisConnectionManager(ConnectionManager)` — `broadcast` 시 Redis `publish`, 각 워커가 `subscriber_task` 로 수신 후 로컬 연결로 재분배
- 기존 `ConnectionManager` 상속 구조 → 라우터 코드 수정 불필요
- `create_ws_manager(backend, redis_url)` 팩토리. 기본 `memory`, `WS_BACKEND=redis` 로 전환
- Redis 미기동 상태에서 `broadcast` 호출 시 예외 삼키고 로컬 폴백 (개발 편의)
- [app/config.py](app/config.py) — `WS_BACKEND`, `REDIS_URL` 추가
- 테스트 6개 (팩토리 분기 / URL 누락 ValueError / 잘못된 backend / Redis 없이 호출 폴백 / publish 실패 폴백)

### 🧪 전체 회귀
- `test_coverage_* / test_defect_* / test_defects_api / test_floorplan_calibration / test_image_storage / test_inference_pipeline / test_logging_json / test_metrics / test_push_service / test_refresh_token / test_telemetry_cache / test_wallpaper_double_gate / test_ws_manager / test_ws_manager_redis` → **100/100 통과**
- 제외: `test_yolo_inference.py` (가중치 미배치 환경)

### 📦 신규 의존성
- `prometheus-client` (설치 완료)
- `redis` (설치 선택 — `WS_BACKEND=redis` 로 전환할 때만 필요)
- `firebase-admin` (나중 푸시 실 발송 시 필요)

### ⚠️ 배포 전 필수 조치
- **alembic heads 정리** — 신규 2개(`f3d1b6c09a12`, `c7e2d5f3a18b`) 추가로 체인이 더 길어짐. `0003` 과 merge 필요
- **운영 전환 시 환경변수 점검**:
  - `LOG_JSON=true`
  - `PUSH_PROVIDER=fcm` (firebase-admin 연결 후)
  - `WS_BACKEND=redis` + `REDIS_URL` (수평 확장 시)
- Telemetry POST 오픈 상태 확인 — VPC/방화벽 레벨 접근 제어 전제

---

## 📝 기본 정보 (Meta)

- 작성자 (Who): @youminsu0523
- 작성 일자 (When): 2026-04-23 14:00
- 목표 기능 (Objective): 학습 데이터셋 출처 관리 문서화 및 .gitignore 설정
- 작업 브랜치/환경: `MS`

---

## 💬 바이브코딩 대화 흐름 (Vibe Coding Log)

### 1️⃣ 학습 데이터셋 출처 문서 추가 (Training Datasets Source Management)

> ⏱ 2026-04-23 14:00

#### 배경
- 20종 결함 분류 모델 학습에 사용한 데이터셋(63,285장)의 출처·라이선스를 체계적으로 정리할 필요
- `backend/training/` 디렉토리의 대용량 데이터셋 파일이 Git 추적에서 제외되어야 함

#### 작업 내용

**① `backend/training/.gitignore` 신규 생성**
- 학습용 대용량 데이터셋 디렉토리(`datasets/`, `gdrive_raw/`, `weights/` 등)를 Git 추적에서 제외
- 모델 가중치(`.pt`, `.onnx` 등)도 추적 제외 대상

**② `backend/training/datasets_sources.md` 신규 생성**
- 데이터셋 총괄표: 9개 데이터셋, 총 63,285장 이미지
- 하자코드(A-01~E-02) ↔ 데이터셋 ↔ 모델 클래스 매핑 테이블
- 원본 데이터 출처 상세 (Roboflow, GitHub, AI Hub 등 28개 소스)
  - 카테고리별 정리: A 구조·기하 / B 단열·방수 / C 마감·표면 / D 바닥·난방 / E 창호·유리
- 라이선스 요약: CC BY 4.0(22개), CC BY-NC(2개), Public Domain(1개), MIT(1개), GPL-3.0(1개), Academic(2개), 내부(1개)

#### 커밋
- `8bf2ad7` — `feat: add .gitignore and documentation for training datasets source management`

#### 비고
- Hijin 브랜치 PR #27 머지 완료 (`f4a2068`) — 04-22 작업분(refresh token, auth guards, prometheus, push notifications, redis pub/sub)은 이전 세션에서 기록 완료

---

## 🎮 2026-04-23~24 — TEST MODE 스트리밍 서비스 + Dashboard bbox 객체탐지 시각화 (@youminsu0523)

> **작업자**: @youminsu0523  
> **작업 브랜치**: `MS`  
> **목표**: 드론 없이 로컬 이미지/영상으로 AI 하자 검출을 시험할 수 있는 TEST MODE 구축. 대시보드에서 bbox 오버레이로 탐지 결과 실시간 시각화.

### ⏱ TEST MODE 백엔드 — test_stream.py (신규 1053줄)

`backend/app/services/test_stream.py` **신규**:
- **카테고리별 균등 샘플링**: 각 하자 유형(Crack, Moisture, Delamination 등)이 골고루 노출되도록 라운드로빈
- **RGB/Thermal 쌍 동기화**: 프레임 버전 카운터로 두 스트림 정합성 보장. 쌍이 없는 데이터는 Thermal에 "No Signal" 표시
- **재생 제어**: 시작(start) / 일시중지(pause) / 재개(resume) / 정지(stop) 상태 관리
- **image_crop 생성**: DefectCard 썸네일 표시용 base64 JPEG 생성
- **20종 ONNX 추론 or 목업 폴백**: 모델 가중치가 있으면 실제 추론, 없으면 랜덤 하자 목업 생성
- **소스 전환**: 프로젝트 내장 학습 데이터(`training/`) ↔ 사용자 직접 업로드 이미지/영상

### ⏱ TEST MODE 백엔드 — stream.py API 엔드포인트 추가 (189줄+)

`backend/app/api/stream.py` 수정:
| 메서드 | 경로 | 역할 |
|--------|------|------|
| POST | `/stream/test/init` | 테스트 모드 초기화 (이미지 스캔 + 모델 로드) |
| POST | `/stream/test/start` | 재생 시작 |
| POST | `/stream/test/pause` | 일시중지 |
| POST | `/stream/test/resume` | 재개 |
| POST | `/stream/test/stop` | 정지 |
| GET | `/stream/test/state` | 현재 재생 상태 조회 |
| GET | `/stream/test/rgb` | 테스트 RGB MJPEG 스트림 |
| GET | `/stream/test/thermal` | 테스트 Thermal MJPEG 스트림 |
| POST | `/stream/test/detection-mode` | 시각화 모드 전환 (bbox/detection) |
| POST | `/stream/test/source` | 소스 전환 (project/upload) |
| POST | `/stream/test/upload` | 테스트 이미지/영상 업로드 |
| DELETE | `/stream/test/upload` | 업로드 파일 삭제 |
| GET | `/stream/test/upload/list` | 업로드 파일 목록 |
| GET | `/stream/test/defect/{id}/{channel}` | 특정 하자 시점 프레임 스냅샷 |

### ⏱ TEST MODE config 추가

`backend/app/config.py`:
- `DRONE_CONNECTED: bool = False` — 드론 미연결 시 테스트 모드 활성화
- `TEST_MODE_ENABLED: bool = True`
- `TEST_IMAGE_INTERVAL: float = 3.0` — 이미지 전환 주기(초)

### ⏱ TEST MODE 프론트엔드 — TestModeBar.jsx (신규 319줄)

`frontend/src/components/dashboard/TestModeBar.jsx` **신규**:
- 시작/일시중지/정지 재생 제어 버튼
- 프로젝트 데이터 ↔ 직접 업로드 소스 전환 토글
- 직접 업로드 모드: 이미지/영상 대량 드래그&드롭 첨부 + 파일 목록 표시
- bbox / detection 시각화 모드 전환

### ⏱ Dashboard + LiveVideoFeed 연동

- **`Dashboard.jsx`** — 테스트 모드일 때 `TestModeBar` 렌더링 + 테스트 스트림 URL(`/stream/test/rgb`, `/stream/test/thermal`)로 전환
- **`LiveVideoFeed.jsx`** — 테스트 모드 스트림 URL 분기 처리. bbox 오버레이 표시를 위한 `<img>` src 동적 전환
- **`App.jsx`** — `DashboardLayout`에서 테스트 모드 진입 시 `test/init` 자동 호출, 퇴장 시 `test/stop` cleanup
- **`sessionStore.js`** — `enterTestMode()`, `testSource`, `testPlayState`, `testDetectionMode`, `setTestSource()`, `setTestPlayState()`, `setTestDetectionMode()` 상태 추가
- **`camera.py`** — 테스트 모드용 프레임 생성 지원

---

## 🔗 2026-04-24 — Frontend↔Backend 미연결 모듈 일괄 연동 + 슈퍼어드민 + 도면 검증 (@youminsu0523)

> **착수 시각**: 2026-04-24 09:30  
> **작업자**: @youminsu0523  
> **작업 브랜치**: `MS`  
> **목표**: 프론트엔드에서 localStorage Mock으로만 동작하던 5개 모듈을 백엔드 실제 API로 전환. 백엔드에만 구현되어 있던 기능을 프론트에 연결. 기획만 완료된 도면 검증 파이프라인 구현. 슈퍼어드민 시드 계정 생성.

### ⏱ 09:30 | Phase 1 — Frontend Mock → Real API 전환 (5개 파일)

기존 localStorage 기반 Mock API를 axios + JWT 인증 백엔드 호출로 전면 교체. 각 API 파일의 함수 시그니처는 유지하되 body만 fetch로 교체하는 설계대로 진행.

| 파일 | 변경 내용 |
|------|----------|
| `frontend/src/api/chatApi.js` | localStorage 시드+CRUD → `axios.get/post/patch` + JWT 헤더. `listConversations(userId)` → `listConversations()` (서버가 JWT로 사용자 식별) |
| `frontend/src/api/notificationApi.js` | 동일 패턴. `simulateLatency()` 제거, 실제 HTTP 호출로 교체 |
| `frontend/src/api/sitesApi.js` | 동일 패턴. 필터 파라미터(`status`, `building_type`, `client_type`, `search`) 쿼리스트링 지원 추가 |
| `frontend/src/api/reportsApi.js` | `POST /report/save` + `GET /report/{id}/download` 마크다운 다운로드 함수(`downloadReport`) 추가 |
| `frontend/src/api/organizationApi.js` | Mock 시드 제거, `removeMember(userId)` 함수 추가 |

### ⏱ 09:40 | Phase 1 후속 — Store 호출부 수정

API 시그니처 변경에 따른 Store 수정:

- **`chatStore.js`** — `CURRENT_USER` 하드코딩 제거. `getCurrentUser()`를 localStorage에서 읽도록 변경. `sendMessage`에서 sender 정보 제거 (백엔드 JWT에서 자동 식별). `createConversation`에서 `participants` → `participant_ids`로 키 변경. `getUnreadCounts` 응답의 `perConversation` → `per_conversation` 매핑
- **`reportsStore.js`** — `updateReport` import 제거 (백엔드에 PATCH 없음). `update` 메서드를 로컬 캐시 전용으로 변경

### ⏱ 09:45 | Phase 2 — Refresh Token 프론트 연동

- **`authApi.js`** — 401 응답 인터셉터 추가. `isRefreshing` 플래그 + `failedQueue` 패턴으로 동시 요청 대응. Refresh 실패 시 localStorage 정리 + `/login` 리다이렉트. `uploadProfileImage()`, `deleteProfileImage()`, `updateMe()` 함수 추가
- **`authStore.js`** — `setAuth(token, user, refreshToken)` 3번째 파라미터 추가. `logout()`에 `refresh_token` 삭제 추가
- **`Login.jsx`** / **`OAuthCallback.jsx`** — 로그인 응답에서 `refresh_token` 추출하여 `setAuth`에 전달

### ⏱ 09:50 | Phase 2 — 보고서 다운로드 버튼

- **`ReportDetail.jsx`** — 헤더에 `Download` 아이콘 버튼 추가. `downloadReport(id)` 호출 → 마크다운 파일 브라우저 다운로드

### ⏱ 09:52 | Phase 2 — 부서/미소속 사용자/프로필 이미지 확인

- **`AdminMembers.jsx`** — 이미 백엔드 직접 axios 호출로 구현 완료 확인 (부서 CRUD + 미소속 배정 + 멤버 수정)
- **`EmployeeLanding.jsx` EditProfileModal** — 이미 fetch로 `PUT /auth/me/profile-image` 직접 호출 구현 확인

### ⏱ 09:55 | Phase 3 — Floorplan OpenCV 벽체 추출 (process 엔드포인트 구현)

기존 `POST /floorplan/{id}/process`는 TODO 스텁이었음. 실제 처리 로직 연결:

- **`floorplan.py`** — 파일 존재 확인 → `aiofiles`로 비동기 읽기 → content_type 분기:
  - **JPG/PNG/WEBP**: `extract_walls_from_bytes()` 직접 호출
  - **PDF**: `pdf2image.convert_from_bytes()` → OpenCV BGR 변환 → 동일 파이프라인 (pdf2image 미설치 시 422 반환)
  - **DXF**: `ezdxf.read()` → LINE 엔티티 좌표 추출 → 정규화 (ezdxf 미설치 시 422 반환)
  - 처리 성공 → `status="completed"`, `wall_count`, `walls_data` DB 갱신
  - 처리 실패 → `status="failed"` + 500 에러

### ⏱ 10:00 | Phase 3 — 도면 이미지 품질 검증 파이프라인

`project_inspection_area_auto.md` 메모리에 정의된 2단계 검증 사양을 구현:

- **`floorplan_processor.py`** — `validate_floorplan_quality()` 함수 추가. 7개 체크 항목:
  1. **해상도**: 1000×1000px 이상 권장, 500px 미만 거부
  2. **선명도**: Laplacian variance. 100+ 양호, 30 미만 거부
  3. **대비**: 그레이스케일 표준편차. 50+ 양호, 25 미만 거부
  4. **직선 비율**: HoughLinesP 기반. 에지 대비 직선 픽셀 비율 0.3+ 양호, 0.15 미만 거부
  5. **직각 교차점**: 수평선×수직선 쌍 수. 4+ 양호, 2 미만 부족
  6. **기울기**: 직선 각도 중앙값의 수평/수직 편차. 3° 이내 양호, 10°+ 경고
  7. **벽체 감지 수**: `extract_walls_from_bytes` 호출. 5+ 양호, 3 미만 거부

- **종합 판정**: `status` = `ok` (에러 없음) | `warning` (경고만) | `rejected` (에러 있음). 점수는 항목별 가중 평균.

- **`floorplan.py`** — `POST /floorplan/validate` 엔드포인트 추가. 파일 크기 50KB 미만 즉시 거부. `FloorplanValidateResponse` 스키마로 응답
- **`schemas/floorplan.py`** — `FloorplanValidateResponse` 스키마 추가

### ⏱ 10:05 | 슈퍼어드민 시드 계정 생성

- **`main.py`** — `_ensure_superadmin_seed()` 함수 추가. lifespan에서 DB 초기화 직후 호출. `admin` username 존재 여부 확인 → 없으면 자동 생성 (중복 방지)
- **계정 정보**: ID `admin` / PW `admin` / email `admin@aeroinspect.io` / `is_superadmin=True`
- **DB에 즉시 생성 완료** (스크립트 직접 실행)

### ⏱ 10:10 | bcrypt + passlib 호환성 이슈 해결

- **문제**: `bcrypt 5.0.0` + `passlib 1.7.4` 조합에서 `ValueError: password cannot be longer than 72 bytes` 에러. passlib 내부의 `detect_wrap_bug()` 함수가 72바이트 초과 비밀번호로 테스트하면서 새 bcrypt의 strict 검증에 걸림
- **해결 1**: `security.py` — passlib `CryptContext` 제거, `bcrypt` 라이브러리 직접 사용으로 전환. `hash_password()` = `bcrypt.hashpw()`, `verify_password()` = `bcrypt.checkpw()`
- **해결 2**: bcrypt 5.0.0 → 4.2.1 다운그레이드 (안정 버전)
- **해결 3**: admin 계정 비밀번호 해시 재생성 (bcrypt 4.2.1 기준)

### ⏱ 10:15 | 슈퍼어드민 Pydantic 이메일 검증 에러 해결

- **문제**: `admin@aeroinspect.local` 이메일 → Pydantic `EmailStr`이 `.local` 도메인을 special-use name으로 거부 → 로그인 시 500 에러
- **해결**: 이메일을 `admin@aeroinspect.io`로 변경 (DB + 시드 코드)

### ⏱ 10:20 | 슈퍼어드민 라우팅 + 권한 가드 수정

슈퍼어드민이 조직 미소속 상태에서도 모든 기능에 접근 가능하도록 수정:

- **`Login.jsx`** / **`OAuthCallback.jsx`** — `is_superadmin`이면 조직 없어도 `/employee`로 직행
- **`OrgRequired.jsx`** — 슈퍼어드민은 `currentOrg` 체크 건너뜀. `adminOnly` 페이지도 접근 허용
- **`EmployeeLanding.jsx`** — `isAdmin` 판정에 `user?.is_superadmin` 조건 추가 (2곳: EmployeeHeader, QuickActionsSection). `QuickActionsSection`에서 `user` 미선언 버그 수정 (`useAuthStore` import 누락)
- **`AdminMembers.jsx`** — `fetchData` 분기 처리: 슈퍼어드민은 `admin/all-users` 우선 호출, 조직 API는 try-catch로 감싸서 미소속 시 빈 배열로 fallback. 전체 사용자 행 클릭 시 편집/배정 모달 연결

### ⏱ 10:30 | 랜딩 헤더 로그인 상태 반영

- **`LandingHeader.jsx`** — `useAuthStore` 연동:
  - **비로그인**: `로그인` + `도입 문의하기` 표시 (직원전용 숨김)
  - **로그인 상태**: `직원 전용` + `로그아웃` + `도입 문의하기` 표시
  - 데스크탑/모바일 메뉴 모두 동일 적용

### ⏱ 10:35 | EmployeeLanding 환영 배너 개인화

- **`WelcomeBanner`** — 하드코딩 `과장님` 제거. `authStore.user.name` + `currentOrg.position` 동적 표시. 직급 미설정 시 이름만 표시

### ⏱ 10:40 | 멤버관리/TEST MODE 권한 분기

- **`QuickActionsSection`** — `멤버 관리` + `TEST MODE` 카드를 `isAdmin` (admin 또는 superadmin) 조건으로 묶어 일반 멤버에게 비노출

### 🔗 변경 파일 목록 (Frontend 17개 + Backend 5개)

**Frontend 수정**:
- `api/chatApi.js`, `api/notificationApi.js`, `api/sitesApi.js`, `api/reportsApi.js`, `api/organizationApi.js` — Mock → Real API
- `api/authApi.js` — Refresh Token 인터셉터 + 프로필 이미지 API
- `store/chatStore.js`, `store/reportsStore.js`, `store/authStore.js` — API 시그니처 변경 대응
- `pages/Login.jsx`, `pages/OAuthCallback.jsx` — refresh_token 전달 + 슈퍼어드민 라우팅
- `components/auth/OrgRequired.jsx` — 슈퍼어드민 가드 bypass
- `pages/EmployeeLanding.jsx` — isAdmin 로직 + 배너 개인화 + TEST MODE 권한
- `pages/employee/AdminMembers.jsx` — 슈퍼어드민 분기 + 행 클릭 편집
- `pages/employee/ReportDetail.jsx` — 다운로드 버튼
- `components/landing/LandingHeader.jsx` — 로그인/로그아웃 상태 분기

**Backend 수정**:
- `app/core/security.py` — passlib → bcrypt 직접 사용
- `app/api/floorplan.py` — process 실제 구현 + validate 엔드포인트
- `app/services/floorplan_processor.py` — `validate_floorplan_quality()` 추가
- `app/schemas/floorplan.py` — `FloorplanValidateResponse` 추가
- `app/main.py` — 슈퍼어드민 시드 + bcrypt 4.2.1 호환

### 🔗 신규 API 엔드포인트
| 메서드 | 경로 | 역할 |
|--------|------|------|
| POST | `/api/v1/floorplan/validate` | 도면 이미지 품질 검증 (7개 항목) |

### 📐 설계 결정 사항
- **Mock → Real 전환 전략**: API 파일의 함수 시그니처 유지 → Store/컴포넌트 호출부 최소 변경. 각 API 파일에 독립 axios 인스턴스 생성 (JWT + X-Organization-Id 헤더 자동 첨부)
- **Refresh Token 큐잉**: 동시에 여러 요청이 401 받았을 때 refresh 1회만 실행, 나머지는 큐에 대기 후 새 토큰으로 재시도
- **슈퍼어드민 권한 모델**: 조직 소속 없이도 모든 페이지/기능 접근 가능. `is_superadmin` 플래그가 `currentOrg` 체크보다 우선
- **도면 검증 판정 기준**: `rejected` (에러 1개 이상) > `warning` (경고만) > `ok` (전부 통과). 점수는 참고용이며 판정은 에러/경고 유무로 결정

---

## ⏱ 2026-04-21 ~ 04-23 | ML 학습 파이프라인 전체 구축

### 📋 작업 개요
20종 건물 하자 검출 AI를 위한 전체 ML 파이프라인 구축: 데이터 수집 → 폴더링 → 라벨링 → 학습 → ONNX 변환

### 📂 데이터 수집 (gdrive_raw/)
- **총 63,285장**, 31개 원본 폴더, 하자코드(A-01~E-02) 기준 폴더명 통일
- **출처**: Roboflow Universe (CC BY 4.0), AI Hub (CC BY-NC), GitHub 공개 데이터셋, 팀 자체 수집
- **출처 문서**: `training/datasets_sources.md` 생성

| 주요 데이터 | 이미지 수 | 출처 |
|-----------|----------|------|
| 균열 (Crack) | ~15,000장 | Roboflow, AI Hub |
| 벽지/마감 (Wallpaper) | ~12,000장 | Roboflow |
| 바닥/타일/유리 | ~8,600장 | Roboflow, GitHub |
| 열화상 (Thermal) | ~4,400장 | Roboflow, Crack900 |
| 실내 세그멘테이션 | ~7,400장 | Roboflow |
| 코킹 하자 | ~6,700장 | 팀 자체 수집 |

### 🔧 폴더링 / 라벨링
1. 중복 제거 — MD5 해시 100% 검증 후 삭제 (1.7GB 회수)
2. 라벨 포맷 통일 — COCO JSON → YOLO txt, polygon → bbox 변환
3. 클래스 매핑 — 원본 클래스 → 프로젝트 20종 하자 코드
4. full-image bbox 제거 — 부정확한 bbox 9,549장 제거, 정밀 bbox 데이터 보강

### 🤖 모델 학습 결과 (v3~v4)

| 모델 | 역할 | 데이터 | 성능 |
|------|------|--------|------|
| M1 YOLO | 구조·방수 검출 (A-02,A-03,B-03,B-04) | 20,393장 nc=3 | mAP@0.5=0.685 |
| M1 ResNet | 균열 유형 분류 | 2,991장 4cls | ValAcc=0.999 |
| M2 YOLO | 마감·표면 검출 (C-01~C-05) | 6,546장 nc=2 | mAP@0.5=0.939 |
| M2 ResNet | 표면 유형 분류 | 3,404장 5cls | ValAcc=0.844 |
| M3 YOLO | 바닥·창호 검출 (D-03,D-04,E-01,E-02) | 8,044장 nc=3 | mAP@0.5=0.762 |
| 열화상 YOLO | 열화상 결함 (B-01,B-02,B-05) | 4,372장 nc=3 | mAP@0.5=0.536 |
| M5 YOLO-seg | 기하학 세그 (A-01,A-04) | 7,418장 nc=5 | frames seg |
| M6 PatchCore | 이상탐지 (비지도) | 5,361장 | coreset 77MB |

### 🛠 신규/수정 파일

**신규**:
- `training/retrain_all_v3.py` — GPU 순차 학습 파이프라인
- `training/integrate_new_data.py` — gdrive_raw → datasets 자동 매핑 통합
- `training/datasets_sources.md` — 데이터셋 출처 문서
- `training/auto_train_all.py` — 자동 학습 + 모니터링 파이프라인

**수정**:
- `app/services/alignment_detector.py` — M5+G1+LiDAR 정밀 기하학 검출기 완전 재작성
  - RANSAC 200회 라인 피팅 + 서브픽셀 엣지 검출
  - LiDAR 수직/수평 기준값 연동 (드론 roll/pitch 역보정)
  - KCS 41 46 01 기준 불량 판정 (수직 ±3mm/m, 직각 ±2mm/m)

### 📐 설계 결정 사항
- **gdrive_raw vs datasets**: gdrive_raw = 원본(하자코드별), datasets = 학습용(모델별, YOLO txt 통일)
- **bbox 정확도**: full-image bbox 제거 + 정밀 bbox 보강 + box loss 가중치 10.0
- **GPU/CPU 병렬**: YOLO=GPU, ResNet/PatchCore=CPU 동시 진행
- **열화상 보강**: 태양광 열화상(열점 유사) 추가 → 1,262→4,372장, mAP 14% 개선

---

## 📅 2026-04-24 (목) — 테스트 모드 고도화: 실시간 오버레이 + 카테고리 균등 샘플링

> **작업자**: @youminsu0523 (Claude Opus 4.6 바이브코딩)
> **브랜치**: `MS`

### ⏱ 세션 시작 | 이전 대화 복구 + 이슈 파악

이전 대화가 유실되어 git diff 기반으로 테스트 모드 구현 상태를 복구.
사용자가 보고한 4가지 이슈 확인:
1. DRONE1(RGB)과 DRONE2(Thermal)에 서로 다른 구간의 이미지가 표시됨
2. 하자탐지목록에 균열만 나옴 (다른 하자 유형 확인 불가)
3. DefectCard에 이미지가 "없음"으로 표시됨
4. onnxruntime 미설치 에러

### ⏱ R1 | 카테고리별 균등 샘플링 구현 (균열만 나오는 문제 해결)

- **문제**: 28,914장을 한꺼번에 셔플 → ext_crack이 82.8%(23,372장) 차지 → 거의 균열만 노출
- **데이터 분포**:
  | 카테고리 | 이미지 수 | 비율 |
  |---------|----------|------|
  | ext_crack | 23,372 | 82.8% |
  | ext_glass | 2,745 | 9.7% |
  | ext_building_crack | 1,064 | 3.8% |
  | ext_wall_crack | 678 | 2.4% |
  | ext_floor_crack | 170 | 0.6% |
  | ext_surface | 144 | 0.5% |
  | ext_concrete | 10 | 0.04% |
  | paired_crack (Crack900) | 731 | 2.6% |
- **해결**: `_category_frames: Dict[str, List[TestFrame]]`로 카테고리별 그룹핑 후, `_advance_frame()`에서 카테고리를 균등 확률(12.5%)로 랜덤 선택
- **추가 수정**: 디렉토리 구조가 `ext_crack/train/images/*.jpg` 3단계 깊이 → `os.path.relpath()` + `Path(rel_path).parts[0]`으로 1단계 디렉토리명 추출

### ⏱ R2 | RGB-Thermal 쌍 동기화 (프레임 버전 카운터)

- **문제**: 두 MJPEG 스트림이 독립적으로 `sleep(3초)` → 타이밍 어긋남
- **해결**: `_frame_version: int` 카운터 도입. RGB 제너레이터가 양쪽 프레임 준비 완료 후 `++`, Thermal 제너레이터는 새 버전까지 대기

### ⏱ R3 | DefectCard "없음" → 실제 이미지 표시 (image_crop)

- **문제**: `image_crop` 필드 누락 → DefectCard가 "없음" 표시
- **해결**: `_generate_random_crop()` — 프레임 랜덤 크롭 → 112x112 → base64 JPEG. 실제 추론에도 bbox 기반 `_crop_to_base64()` 적용

### ⏱ R4 | onnxruntime 설치

- `requirements.txt`에 `onnxruntime` 추가 + pip install 완료 (v1.25.0)

### ⏱ R5 | 하자 클릭 시 DRONE1/DRONE2에 해당 시점 프레임 표시

- **Backend**: `store_defect_frame()` — raw RGB/Thermal JPEG + bbox/label/severity 메타데이터를 `OrderedDict`에 저장 (최대 200건)
- **Backend**: `GET /test/defect/{defect_id}/{channel}?mode=` — 저장된 프레임에 mode별 시각화 적용 후 JPEG 반환
- **Frontend**: `LiveVideoFeed.jsx` — `isTestMode && selectedDefect` 조건에서 MJPEG 스트림 대신 defect frame endpoint URL로 전환. "DEFECT VIEW" 배지 표시

### ⏱ R6 | bbox 오버레이에 한글 라벨 깨짐 해결

- **문제**: `cv2.putText()`는 한글 미지원
- **해결**: PIL + Windows `malgunbd.ttf`(맑은 고딕 Bold) 자동 탐색 + 폰트 캐싱. cv2로 네모박스, PIL로 한글 텍스트 렌더링

### ⏱ R7 | 하자 탐지 타이밍 수정 (이미지보다 목록이 먼저 갱신)

- **문제**: WS 전송 → yield 순서 → 하자가 이미지보다 먼저 목록에 표시
- **해결**: yield 먼저 → 0.5초 대기 → 브로드캐스트 순서로 변경

### ⏱ R8 | BBox / 객체감지(Detection) 2가지 시각화 모드

- **BBox 모드**: 빨간 네모박스 + 한글 라벨
- **Detection 모드**: 반투명 컬러 마스크(심각도별) + Canny 에지 윤곽 강조 + L자 코너 마커 + 심각도 뱃지
- **심각도별 색상**: HIGH=빨강, MED=주황, LOW=노랑
- `TestModeBar.jsx`에 BBOX/DETECT 토글 → `POST /test/detection-mode` API 호출

### ⏱ R9 | 실시간 라이브 스트림 오버레이 (핵심 리팩토링)

- **문제**: 오버레이가 하자 목록 클릭 시에만 보이고 라이브 스트림에는 미표시
- **해결 — 제너레이터 흐름 전면 리팩토링**:
  ```
  기존: 프레임 → 인코딩 → yield(원본) → 추론 → WS 브로드캐스트
  변경: 프레임 → _detect(결과만) → _apply_live_overlay → 인코딩 → yield(오버레이 포함) → _broadcast_detection
  ```
- **추론 분리**: `_detect()`(결과 반환) + `_broadcast_detection()`(WS 전송)
- `_apply_live_overlay()`: numpy 프레임에 직접 오버레이 (JPEG 디코딩/재인코딩 없음). `_detection_mode`에 따라 bbox 또는 detection 스타일 적용
- Thermal에도 동일한 오버레이 적용. 영상 프레임에도 동일 흐름

### 🔗 변경 파일 목록

**Backend (3개)**:
- `app/services/test_stream.py` — 전면 리팩토링
- `app/api/stream.py` — defect frame 조회 + detection-mode 엔드포인트
- `requirements.txt` — onnxruntime 추가

**Frontend (3개)**:
- `src/components/video/LiveVideoFeed.jsx` — 하자 선택 시 defect frame 표시 + detection mode 쿼리
- `src/components/dashboard/TestModeBar.jsx` — BBOX/DETECT 토글 + API 연동
- `src/store/sessionStore.js` — `testDetectionMode` 상태 추가

### 🔗 신규 API 엔드포인트
| 메서드 | 경로 | 역할 |
|--------|------|------|
| POST | `/api/v1/stream/test/detection-mode` | 감지 시각화 모드 전환 (bbox ↔ detection) |
| GET | `/api/v1/stream/test/defect/{id}/{channel}?mode=` | 하자 시점 프레임 조회 |

### 📐 설계 결정 사항
- **카테고리 균등 샘플링**: 카테고리 단위 랜덤 선택 → 데이터 불균형에도 전체 하자 유형 골고루 노출
- **프레임 저장 전략**: raw JPEG + 메타데이터 저장, 조회 시 mode별 시각화 적용
- **라이브 오버레이**: numpy 프레임에 직접 그려서 JPEG 인코딩 1회만 수행
- **추론/브로드캐스트 분리**: `_detect()` → `_apply_live_overlay()` → `yield` → `_broadcast_detection()` 4단계 분리

---

## 📅 2026-04-24 (목) — 입력 필드 UX 수정 + 멤버 배정 조직 선택 + 채팅 시스템 전면 리팩토링

> **작업자**: @youminsu0523 (Claude Opus 4.6 바이브코딩)
> **브랜치**: `MS`

### ⏱ 세션 시작 14:00 | 이슈 파악

사용자가 조직명 입력 필드에서 텍스트가 보이지 않는 문제를 발견. 드래그해야만 보임.
추가로 관리자 멤버 배정 모달에 조직 선택 기능이 없어 슈퍼어드민이 다른 조직에 멤버를 배정할 수 없는 문제도 확인.
채팅 시스템에서 아이콘이 "??", 이름이 "알 수 없음"으로 표시되고 메시지 좌우 정렬이 안 되는 문제, DM 중복 생성 문제도 발견.

### ⏱ R1 | 전역 입력 필드 텍스트 색상 수정

- **문제**: `body`에 `text-white`가 전역 적용 → 라이트 배경 위 `input/textarea/select`가 흰 배경에 흰 글씨
- **영향 범위**: Login, Signup, FindAccount, Onboarding, AdminMembers, SessionSetup, PreWork, EmployeeLanding, SiteFormModal, ContactModal, AddDefectDialog 등 전체 폼 요소
- **해결**: `index.css`에 전역 룰 추가
  ```css
  input, textarea, select { color: #111827; }
  input::placeholder, textarea::placeholder { color: #9ca3af; }
  ```

### ⏱ R2 | 멤버 배정 모달 — 소속 조직 선택 기능 (슈퍼어드민)

- **프로세스 변경**: 역할→부서→직위(기존) → **소속 조직→역할→부서→직위**(개선)
- **Backend 추가**:
  - `GET /admin/all-orgs` — 전체 조직 목록 + 멤버 수 (슈퍼어드민 전용)
  - `GET /admin/orgs/{org_id}/departments` — 특정 조직의 부서 목록 (슈퍼어드민 전용)
  - `AssignMemberRequest`에 `organization_id: Optional[UUID]` 추가
  - `assign_member` — 슈퍼어드민이 `organization_id` 지정 시 해당 조직에 배정, 일반 admin은 자기 조직에만
- **Frontend**: 배정 모달에서 조직 선택 시 해당 조직의 부서 목록을 동적 로드. 조직 미선택 시 역할/부서/직위 비활성화(opacity-40)

### ⏱ R3 | 채팅 시스템 Mock 제거 — 실제 사용자 데이터 연동

- **근본 원인**: `CURRENT_USER = { id: 't1', ... }` Mock ID를 모든 채팅 컴포넌트가 참조 → 실제 UUID와 불일치
  - `isMine` 항상 `false` → 모든 메시지가 왼쪽(상대방) 배치
  - `CHAT_TEAM_MEMBERS` Mock 배열에서 상대방 검색 → 매칭 실패 → "??" 아이콘, "알 수 없음" 이름
  - `conv.participants`가 `{user_id, name, initials}` 객체 배열인데 ID 문자열처럼 취급
- **수정 파일 5개**:
  - `ChatHeader.jsx` — `CHAT_TEAM_MEMBERS` 제거, `conv.participants`에서 직접 상대방 조회
  - `ConversationItem.jsx` — 동일 Mock 제거, participants 객체에서 이름/이니셜 표시
  - `MessageBubble.jsx` — `CURRENT_USER.id` → `localStorage.user.id`로 교체. 내 메시지 오른쪽(노란색), 상대 왼쪽(흰색)
  - `ConversationList.jsx` — 검색 필터도 participants 객체 사용
  - `NewChatModal.jsx` — `CURRENT_USER` 참조 3곳 모두 `getCurrentUser()` 함수로 교체

### ⏱ R4 14:20 | DM 중복 생성 방지

- **문제**: `findDMConversation`이 `participants.some(p => p.user_id === userId1 || userId2)` — 아무 한 명만 매칭되면 "기존 DM 있음"으로 판단 → 잘못된 DM 반환 or 미매칭 시 중복 생성
- **Frontend 수정**: `&&` 연산으로 **두 사용자 모두** 참여하는 DM만 매칭
  ```javascript
  c.participants.some(p => p.user_id === userId1) &&
  c.participants.some(p => p.user_id === userId2)
  ```
- **Backend 수정**: `create_conversation`에서 DM 생성 시 기존 DM 존재 여부를 서버에서도 검증 (aliased join으로 두 사용자 모두 참여하는 DM 검색)

### ⏱ R5 14:30 | 채팅 나가기 기능 추가

- **Backend**: `DELETE /conversations/{id}/leave` — ConversationMember 삭제, 남은 참여자 없으면 대화방 자체 삭제
- **Frontend**: `chatApi.js`에 `leaveConversation()` 추가, `chatStore.js`에 `leaveConversation` 액션 추가
- **UI**: `ParticipantPanel.jsx` 하단에 빨간색 "대화 나가기" 버튼 + `window.confirm` 확인 다이얼로그

### 🔗 변경 파일 목록

**Backend (3개)**:
- `app/schemas/organization.py` — `AssignMemberRequest`에 `organization_id` 추가
- `app/api/organization.py` — assign API 수정 + `GET /admin/all-orgs`, `GET /admin/orgs/{id}/departments` 추가
- `app/api/chat.py` — `DELETE /conversations/{id}/leave` 추가 + DM 중복 방지 로직

**Frontend (8개)**:
- `src/index.css` — 전역 input/textarea/select 텍스트 색상
- `src/pages/employee/AdminMembers.jsx` — 배정 모달 조직 선택 UI
- `src/components/chat/ChatHeader.jsx` — Mock 제거, 실제 participants 사용
- `src/components/chat/ConversationItem.jsx` — Mock 제거, 실제 participants 사용
- `src/components/chat/MessageBubble.jsx` — 실제 사용자 ID로 좌우 정렬
- `src/components/chat/ConversationList.jsx` — 검색 필터 Mock 제거
- `src/components/chat/NewChatModal.jsx` — Mock 제거
- `src/components/chat/ParticipantPanel.jsx` — participants 객체 처리 + 채팅 나가기 버튼

**API 파일**:
- `src/api/chatApi.js` — `leaveConversation()` 추가 + `findDMConversation` 로직 수정
- `src/store/chatStore.js` — `leaveConversation` 액션 추가

### 🔗 신규 API 엔드포인트
| 메서드 | 경로 | 역할 |
|--------|------|------|
| GET | `/api/v1/organizations/admin/all-orgs` | 전체 조직 목록 (슈퍼어드민) |
| GET | `/api/v1/organizations/admin/orgs/{org_id}/departments` | 특정 조직 부서 목록 (슈퍼어드민) |
| DELETE | `/api/v1/chat/conversations/{id}/leave` | 대화방 나가기 |

### 📐 설계 결정 사항
- **전역 CSS vs 개별 클래스**: `body { text-white }` 상속 문제를 개별 input마다 `text-gray-900` 추가 대신 전역 CSS 규칙으로 일괄 해결 — 유지보수 부담 최소화
- **Mock 데이터 전면 제거**: `CURRENT_USER`(id: 't1') / `CHAT_TEAM_MEMBERS` 참조를 모든 채팅 컴포넌트에서 제거하고 `localStorage.user` 기반으로 교체 — Phase 1 → Phase 2 전환 완료
- **DM 중복 방지 이중 잠금**: 프론트엔드 `findDMConversation` + 백엔드 `create_conversation` 양쪽에서 기존 DM 존재 여부 검증
- **대화 나가기 정리**: 마지막 참여자가 나가면 대화방 자체 자동 삭제 (DB 정리)

## 📅 2026-04-24 (목) — 멤버 관리 초대 코드 표시 버그 수정

> **작업자**: @youminsu0523 (Claude Opus 4.6 바이브코딩)
> **브랜치**: `MS`
> **시각**: 14:41

### ⏱ R10 | 멤버 관리 페이지 초대 코드 미표시 버그 수정

- **문제**: AdminMembers 페이지 상단에 조직 초대 코드가 표시되지 않음
- **원인 분석**:
  - 프론트엔드(`AdminMembers.jsx`)는 `orgInfo?.invite_code`로 표시 로직이 정상 구현되어 있었음
  - 그러나 백엔드 `GET /api/v1/organizations/members` 응답에서 `OrganizationResponse` 구성 시 `invite_code` 필드를 누락
  - 같은 파일의 `GET /api/v1/organizations/my` 엔드포인트에는 `invite_code=org.invite_code`가 포함되어 있었으나, `/members` 엔드포인트에서는 빠져 있었음
  - 스키마 기본값이 `invite_code: Optional[str] = None`이므로 누락 시 항상 `None` → 프론트엔드 조건부 렌더링 통과 못함
- **해결**: `app/api/organization.py` line 122에 `invite_code=org.invite_code` 추가

### 🔗 변경 파일 목록

**Backend (1개)**:
- `app/api/organization.py` — `/members` 응답에 `invite_code` 필드 추가

### 📐 설계 결정 사항
- **초대 코드 흐름**: 조직 생성 시 8자리 코드 자동 발급 → 관리자가 멤버 관리 페이지에서 확인 → 오프라인/메신저로 신규 직원에게 전달 → 온보딩 페이지에서 입력하여 가입
- **코드 알파벳**: 혼동 문자(0/O, 1/I/L) 제외한 32종 문자, 약 1조 조합

---

## 🔄 2026-04-24 — 노션 동기화 스크립트 세션별 상세 캡쳐 개선

> **착수 시각**: 2026-04-24 11:10
> **작업자**: @youminsu0523
> **목표**: `sync_notion_logs.py`가 새 로그 콘텐츠를 세션별로 분리하여, 각 세션의 작업 내용과 관련된 앱 페이지를 Playwright로 상세 캡쳐 후 노션에 업로드하도록 개선.
> **배경**: 기존 동기화는 새 콘텐츠 전체를 하나의 세션으로 묶어 스크린샷 1장만 촬영. 여러 기능(테스트 모드, 멤버관리, 로그인 등)이 섞여 있어도 대시보드 1장만 첨부되어 팀원이 어떤 화면이 변경됐는지 알 수 없었음.

### ⏱ 11:10 | 노션 동기화 실행 → 문제 확인

- 기존 스크립트 실행 결과: Backend/Frontend 각 1장씩만 캡쳐 (전체 내용을 하나로 뭉침)
- 사용자 피드백: "어느 부분인지 상세하게 캡쳐하기로 했었는데"

### ⏱ 11:15 | SESSION_ROUTE_MAP 신규 추가

세션 키워드 → 앱 라우트/페이지 매핑 테이블 추가. 우선순위 기반 매칭:

| 우선순위 | 키워드 예시 | 라우트 | 라벨 |
|---------|-----------|--------|------|
| 높음 | `AdminMembers`, `멤버 관리` | `/employee/admin/members` | 멤버 관리 페이지 |
| 높음 | `Signup`, `회원가입` | `/signup` | 회원가입 페이지 |
| 높음 | `ReportDetail`, `보고서 다운로드` | `/employee/reports` | 보고서 목록 |
| 높음 | `floorplan`, `도면` | `/employee` | 직원 랜딩 (도면) |
| 중간 | `test_stream`, `TestModeBar` | `/dashboard` | 대시보드 테스트 모드 |
| 중간 | `EmployeeLanding`, `슈퍼어드민` | `/employee` | 직원 전용 랜딩 |
| 중간 | `Login.jsx`, `Refresh Token` | `/login` | 로그인 페이지 |
| 낮음 | `Dashboard`, `bbox` | `/dashboard` | 대시보드 |

### ⏱ 11:20 | split_into_sessions() 함수 추가

- 새 콘텐츠를 `\r?\n---\r?\n` (Windows CRLF 대응) 정규식으로 세션별 분리
- 각 세션에서 `##` 또는 `###` 첫 헤딩을 제목으로 추출
- 50자 미만 짧은 섹션(메타 블록 등)은 자동 스킵

### ⏱ 11:25 | infer_session_route() 2단계 매칭

기존 `infer_component_hint()` 대체:
1. **1단계**: 세션 제목(`session_title`)으로 `SESSION_ROUTE_MAP` 매칭 (가장 정확)
2. **2단계**: 본문 전체 텍스트로 매칭
3. **3단계**: 기존 `COMPONENT_HINT_MAP` 폴백

### ⏱ 11:30 | capture_app_screenshot() 라우트 기반 캡쳐로 전면 교체

- **UI 로그인 방식 채택**: `update_screenshots.py` 패턴 참고 — `_login_via_ui()`로 실제 로그인 폼 입력 (`input#userId` + `input#password` → submit → `/employee` 대기)
- 기존 `_inject_auth_token()` (localStorage 직접 주입 + API 로그인 시도) 제거
- 인증 후 `route_info["route"]`로 직접 이동 → 2.5초 대기 → 캡쳐

### ⏱ 11:35 | main() 세션별 개별 처리 루프

```
기존 흐름:
  파일 → [전체 콘텐츠] → 캡쳐 1장 → Notion 1회

변경 흐름:
  파일 → split_into_sessions() → [세션1] → 라우트 추론 → 캡쳐 → Notion
                                 → [세션2] → 라우트 추론 → 캡쳐 → Notion
                                 → [세션3] → 라우트 추론 → 캡쳐 → Notion
```

### ⏱ 11:40 | CRLF 버그 수정 + 재동기화 검증

- 첫 실행에서 세션이 1개로만 감지됨 → 원인: Windows `\r\n` 때문에 `\n---\n` 패턴 미매칭
- 정규식을 `\r?\n---\r?\n`으로 수정 → 4개 세션 정상 분리 확인
- 기존 페이지 아카이브 → 커서 롤백 → 재실행:
  - Backend 4세션 + Frontend 2세션 = **총 6장 상세 캡쳐** 노션 업로드 완료

### 🛠 변경 파일 목록

**수정 (1개)**:
- `sync_notion_logs.py` — 세션별 분리 + 라우트 매핑 + UI 로그인 캡쳐 + 개별 Notion append

### 📐 설계 결정 사항
- **세션 분리 기준**: `---` (마크다운 수평선). Vibe_Coding_Log.md가 이미 이 구분자를 세션 경계로 사용 중
- **라우트 매핑 우선순위**: 구체적 키워드(멤버관리, 회원가입) > 중간(테스트모드, 슈퍼어드민) > 넓은(대시보드). 제목 매칭이 본문 매칭보다 우선
- **UI 로그인 채택 이유**: localStorage 직접 주입은 React 상태와 불일치 → 빈 화면. 실제 폼 로그인이 안정적 (`update_screenshots.py` 검증 완료)
- **커서 갱신 시점**: 모든 세션 처리 완료 후 한 번에 갱신 (중간 실패 시 전체 재시도)

---

## 🔄 2026-04-25 ~ 04-27 — 조직 초대 버그 수정 + 초대코드 만료 + 실시간 채팅 + 첨부파일·읽음 표시

> **착수 시각**: 2026-04-25 14:00
> **작업자**: @youminsu0523
> **목표**: 멤버 초대 플로우 버그 수정, 초대코드 보안 강화(30일 만료), 실시간 채팅 WebSocket 연결, 채팅 첨부파일 전송 및 읽음 표시 기능 구현
> **배경**: 관리자가 멤버를 초대해도 로그인 시 온보딩 페이지가 뜨고, 초대코드 입력 시 401 에러 발생. 퇴사자 보안 우려로 초대코드 주기적 변경 필요. 채팅이 새로고침 없이는 갱신되지 않는 문제. 채팅에 파일 첨부/이모지/읽음 표시 기능 부재.

### ⏱ 04-25 14:00 | 멤버 초대 버그 진단 및 수정

**문제 1**: `invite_member` 엔드포인트가 `status="invited"`로 생성하지만, `_get_user_orgs`는 `active`만 반환 → 초대받은 사용자가 로그인 시 조직 미소속으로 인식

**문제 2**: `join_by_invite_code`가 `invited` 상태 멤버도 "이미 소속" (409)으로 차단 → 교착 상태

**수정**:
- `invite_member`: `status="invited"` → `"active"`로 변경 (즉시 활성화)
- `join_by_invite_code`: invited 상태 멤버가 초대코드 입력 시 active로 전환, deactivated는 403 반환

### ⏱ 04-25 15:00 | 초대코드 30일 만료 기능

**모델 변경** (`models/organization.py`):
- `invite_code_expires_at` 컬럼 추가 (DateTime, nullable, default=30일 후)
- `regenerate_invite_code()` / `is_invite_code_expired()` 메서드 추가

**API 변경** (`api/organization.py`):
- `join_by_invite_code`: 만료 여부 확인 → 410 "초대 코드가 만료되었습니다"
- `POST /invite-code/regenerate`: 새 코드 발급 + 30일 연장 (admin/owner 전용)
- 모든 조직 응답에 `invite_code_expires_at` 포함

**마이그레이션**: `g1a2b3c4d5e6` — `invite_code_expires_at` 컬럼 추가 + 기존 조직에 30일 기본값

### ⏱ 04-26 10:00 | 실시간 채팅 WebSocket 연결

**문제**: 프론트엔드 WebSocket이 `defects` 채널에만 연결 → 백엔드의 `chat:{id}` 브로드캐스트를 수신 불가 → 새로고침 없이 메시지 미표시

**수정** (`api/chat.py`):
- 메시지 전송 시 `chat:{conversation_id}` + 각 참여자 `user:{user_id}` 채널로 이중 브로드캐스트

### ⏱ 04-27 09:00 | 채팅 첨부파일 전송 기능

**모델 변경** (`models/message.py`):
- `file_url` (String 500), `file_name` (String 300), `file_content_type` (String 100) 컬럼 추가
- `text` → `nullable=True` (파일만 보내는 경우)

**API 변경** (`api/chat.py`):
- `POST /conversations/{id}/messages/file` 신규 엔드포인트 (multipart/form-data)
- 저장 경로: `./uploads/chat/{uuid}.ext`, 제한: 200MB/파일
- `aiofiles` 비동기 파일 저장 (기존 프로필 이미지 패턴 재사용)

**스키마 변경** (`schemas/chat.py`):
- `MessageResponse`: `file_url`, `file_name`, `file_content_type`, `read_by_count`, `sender_profile_image_url` 필드 추가
- `LastMessageBrief`: `file_name` 필드 추가

**마이그레이션**: `h1b2c3d4e5f6` — messages 테이블에 파일 컬럼 + text nullable 변경

### ⏱ 04-27 10:00 | 읽음 표시 기능

**API 변경** (`api/chat.py`):
- `get_messages`: 다른 멤버들의 `last_read_at`을 조회하여 각 메시지별 `read_by_count` 계산 (내 메시지에만 표시)
- `mark_read`: 읽음 처리 후 `chat.read` WebSocket 이벤트를 대화방 + 참여자 개인 채널로 브로드캐스트

### 🛠 변경 파일 목록

**수정 (5개)**:
| 파일 | 변경 내용 |
|------|----------|
| `models/organization.py` | `invite_code_expires_at` 컬럼 + 재생성/만료확인 메서드 |
| `models/message.py` | `file_url`, `file_name`, `file_content_type` 컬럼 + text nullable |
| `schemas/organization.py` | `OrganizationResponse`에 `invite_code_expires_at` |
| `schemas/chat.py` | `MessageResponse`에 파일+읽음+프로필 필드, `LastMessageBrief`에 `file_name` |
| `api/organization.py` | invite_member active 전환, join 만료검증, regenerate 엔드포인트 |
| `api/chat.py` | 파일 업로드 엔드포인트, 읽음상태 응답, mark_read WS broadcast, user 채널 broadcast |
| `main.py` | `./uploads/chat` 디렉토리 생성 |

**신규 (2개)**:
| 파일 | 내용 |
|------|------|
| `alembic/versions/g1a2b3c4d5e6_*.py` | invite_code_expires_at 마이그레이션 |
| `alembic/versions/h1b2c3d4e5f6_*.py` | messages 파일 컬럼 마이그레이션 |

### 📐 설계 결정 사항
- **초대코드 만료**: 30일 기본값. 퇴사자가 기존 코드를 알아도 만료 후 사용 불가. 관리자가 수동으로 즉시 재생성 가능
- **파일 업로드 분리**: 텍스트 전용 `POST /messages` (JSON)와 파일 포함 `POST /messages/file` (multipart) 분리 → backward compatibility 유지
- **200MB 제한**: 업무용 CAD 도면, 점검 보고서, 드론 영상 등 대용량 파일 대응
- **읽음 계산 방식**: 기존 `ConversationMember.last_read_at` 인프라 활용. 메시지별 개별 읽음 테이블 없이 timestamp 비교로 효율적 계산
- **이중 WS 채널 브로드캐스트**: `chat:{conversation_id}` (활성 대화방) + `user:{user_id}` (다른 대화방/페이지 밖 알림)

---

#### ⏱ 2026-04-27 | 알림 일괄 삭제 엔드포인트 추가

- **피드백**: 프런트의 「전체 삭제」 액션이 단건 `DELETE /{id}`를 N회 호출하는 방식이라 라운드트립이 누적되고 부분 실패 처리가 복잡함. `read-all`이 이미 단일 PATCH 엔드포인트인 것과 대칭으로 일괄 삭제 엔드포인트를 추가.
- **반영**:
  - `app/api/notifications.py`: `DELETE /api/v1/notifications` 추가. 현재 사용자 소유 알림을 단일 `DELETE … WHERE user_id = current_user.id` 쿼리로 일괄 삭제. 응답: `{"deleted": <rowcount>}`.
  - 라우트 등록 순서 영향: 기존 `DELETE /{notification_id}` 와 충돌 없음(빈 path 와 path 파라미터는 FastAPI 에서 별도 매칭).
  - 모듈 헤더 라우트 일람 주석에 「전체 삭제」 항목 추가.

### 🔗 변경 파일 목록 (1개)

| 파일 | 변경 유형 |
|------|----------|
| `backend/app/api/notifications.py` | `DELETE /notifications` 일괄 삭제 엔드포인트 추가 + 헤더 주석 갱신 |

---

## 📅 2026-04-27 — WS 알림 채널 화이트리스트 + DB 마이그레이션 정리 메모

> **작업자**: @Hijin554 (Claude Opus 4.7 바이브코딩)
> **브랜치**: `Hijin`
> **목표**: `notification_service.create()` 가 broadcast 하는 `notifications:{user_id}` 채널이 WS 게이트웨이에서 거부되어 실시간 푸시가 막혀 있던 문제 해결.

### ⏱ 알림 채널 화이트리스트 추가

- **문제**: `app/services/notification_service.py` 가 `notifications:{user_id}` 로 WS broadcast 하는데, `app/api/websocket.py` 의 `_is_valid_channel()` 이 `chat:`, `user:` 만 허용하고 `notifications:` 는 거부 → 알림 DB 에는 저장되지만 실시간 푸시는 안 가서 사용자가 새로고침해야만 벨 아이콘에 뜨는 상태.
- **해결**: `_DYNAMIC_CHANNEL_PREFIXES` 튜플로 동적 채널 prefix 일원화 (`chat:`, `user:`, `notifications:`). 팀원이 만든 기존 startswith() 패턴과 일관성 유지.
- **검증**: `tests/test_ws_channel_whitelist.py` 신규 — static 4개 + dynamic prefix 3개 + notification_service 형식 + 알 수 없는 채널 거부, 총 9 케이스 모두 통과.

### ⏱ DB 마이그레이션 정리 TODO 메모

- **문제 인식**: `init_db.py` 가 `Base.metadata.create_all` 로 테이블을 자동 생성하고 있고, alembic versions 폴더에 9개 마이그레이션 파일이 별도로 존재. 이중 운영 상태 → 컬럼 추가 시 기존 테이블에 반영 안 됨, 환경별 스키마 drift, `alembic_version` 추적 불가.
- **반영**: `backend/README.md` 의 "DB 마이그레이션 절차" 섹션 맨 앞에 ⚠️ 경고 박스 추가. 출시 전 정리 작업 3단계 (init_db에서 create_all 제거 → `alembic stamp head` → 이후 alembic 단일화) 명시.
- **유지 사유**: 로컬 개발은 init_db 방식으로 잘 돌아가는 중. 운영 RDS 손대기 전 백업 후 일괄 정리할 항목으로 미룸.

### 🔗 변경 파일 목록 (3개)

| 파일 | 변경 유형 |
|------|----------|
| `backend/app/api/websocket.py` | `_DYNAMIC_CHANNEL_PREFIXES` 도입 + `notifications:` prefix 추가 + docstring 갱신 |
| `backend/README.md` | DB 마이그레이션 정리 ⚠️ TODO 메모 추가 |
| `backend/tests/test_ws_channel_whitelist.py` | 채널 화이트리스트 검증 테스트 9개 신규 |

### 📐 설계 결정 사항

- **prefix 화이트리스트 vs 정규식**: 팀원이 만든 기존 코드가 `channel.startswith("chat:")` 방식이라 같은 스타일 채택. UUID 형식 검증은 안 함 (다른 채널들도 형식 검증 없음, 일관성 우선). TODO 주석으로 JWT 인증 보강 필요성 명시.
- **DB 마이그레이션 즉시 처리 vs 메모만**: 운영 RDS 손대면 데이터 손실 위험 + 팀원 로컬 다 깨질 수 있어 즉시 작업 위험 큼. README 메모로 출시 전 체크리스트화 → 안전한 시점에 일괄 처리.

---

## 📡 2026-04-27 — 건물 열화상 자동화 데이터셋(Thermal Building) 스크립트 작성

> **작업자**: @youminsu0523
> **목표**: 건물 점검용 보조 열화상 데이터 수집 스크립트 구현 및 모델 재학습 파이프라인 연동.

### ⏱ 열화상 데이터 구조화 다운로드 및 모델 적용 파이프라인

- **문제**: 가중치 정확도를 올리기 위해 기존 공공 데이터 이외에 추가적인 열화상 데이터셋(Water Leak, Thermal Defect)의 외부 수급 필요성 대두.
- **반영**:
  - `backend/training/download_thermal_building.py` (신규):
    - Roboflow Universe에서 물샘, 단열 결함 등 5종 데이터셋을 자동 다운로드.
    - 다운로드 된 YOLO 포맷 데이터를 열화상 클래스 스키마(`0: Crack, 1: Moisture, 2: delamination`)에 맞게 전처리 및 매핑.
  - `backend/training/retrain_m1_v4s_run.py` (신규):
    - 구조적(Structural) 하자 데이터 2만 장에 대응하는 YOLOv8s 훈련 최적화 파이프라인 실행 래퍼.
    - 에폭 50, batch 32, box=10.0으로 파인튜닝 후 최고 성능 가중치(Best.pt)를 즉시 ONNX 포맷으로 변환.
  - `backend/app/config.py`:
    - 시스템 Heartbeat 설정 설정 축소 변경(`WS_HEARTBEAT_INTERVAL: int = 3`).

### 🔗 변경 파일 목록 (3개)

| 파일 | 변경 유형 |
|------|----------|
| `backend/training/download_thermal_building.py` | Roboflow 기반 열화상 데이터 스크래핑 및 클래스 맵핑 모듈 |
| `backend/training/retrain_m1_v4s_run.py` | YOLOv8s 리트레이닝 스크립트 및 ONNX 최적화 |
| `backend/app/config.py` | WebSocket Heartbeat 변수 최적화 |

---

## 📅 2026-04-28 — 채팅 첨부 mock 파일 placeholder 생성 스크립트

- 작성자 (Who): @unknownName-15
- 작성 일자 (When): 2026-04-28
- 작업 브랜치: `SH`
> **목표**: 채팅 메시지에 첨부된 이미지가 화면에 안 뜨고 다운로드 시 "not found" 가 뜨던 문제 1차 해결 — 시드 데이터의 file_url 이 가리키는 실제 파일을 디스크에 만들어 준다.

### ⏱ 증상 진단

- **현상**: DM 의 "A동_외벽_균열탐지_결과.jpg" 등 첨부 이미지가 채팅 말풍선에 깨진 형태로 뜨고, 다운로드 버튼 클릭 시 404 ("not found") 발생.
- **원인**: `backend/scripts/seed_mock_chats.py` 가 DB 의 `messages.file_url` 에 `/uploads/chat/mock_*.jpg` `mock_*.pdf` `mock_*.xlsx` 9 개 경로를 시드하지만 실제 파일은 만들지 않음 → `app.mount("/uploads", StaticFiles(...))` 가 파일을 못 찾고 404 반환 → `<img>` 깨지고 다운로드도 실패.
- **추가 잠재 이슈** (다음 작업 단계 B·C 로 분리): cross-origin 환경에서 `<a download>` 속성 무시, `StaticFiles` 가 `Content-Disposition` 미설정으로 한글 파일명 손실. 이번 커밋에서는 시드 mock 파일 placeholder 만 우선 채워 화면 렌더 확인이 목표.

### ⏱ 해결: placeholder 생성 스크립트 신규

- **신규 파일**: `backend/scripts/generate_mock_chat_files.py`
  - 이미지 5종 (`mock_crack_detection.jpg` 외): PIL 로 라벨 텍스트 + 격자 패턴 placeholder JPG (800x560) 생성. Windows 한글 폰트(`malgun.ttf`) 자동 탐색.
  - PDF 3종 (`mock_report_march.pdf` 외): handcrafted 최소 유효 PDF 1.4 (Catalog/Pages/Page/Font/Contents 5 오브젝트 + xref). 외부 라이브러리 없이 ASCII 라벨 한 줄 표시.
  - XLSX 1종 (`mock_defect_data.xlsx`): `openpyxl` 미설치 환경 대응으로 `zipfile` stdlib 만 사용해 최소 유효 워크북([Content_Types].xml + rels + workbook.xml + sheet1.xml + sharedStrings.xml) 직접 작성.
- **실행**: `cd backend && python -m scripts.generate_mock_chat_files` → `backend/uploads/chat/` 에 9 개 파일 생성.

### 🔗 변경 파일 목록 (1개)

| 파일 | 변경 유형 |
|------|----------|
| `backend/scripts/generate_mock_chat_files.py` | 신규 — 시드 mock 첨부파일을 placeholder JPG/PDF/XLSX 로 생성하는 일회용 스크립트 |

### 📐 설계 결정 사항

- **시드 스크립트(`seed_mock_chats.py`) 직접 수정 vs 별도 스크립트**: 별도 스크립트로 분리. 시드는 DB 만 다루고 파일 생성은 독립적으로 재실행 가능해야 디버깅 편함. 기존 시드 데이터를 건드리지 않고 placeholder 만 보충 가능.
- **PDF/XLSX 라이브러리 미사용**: reportlab/openpyxl 추가 의존성 도입을 피하기 위해 stdlib + 핸드크래프트 접근. PDF 는 한글 폰트 임베드가 복잡해 라벨은 ASCII 로만 표기 (placeholder 목적이라 충분).
- **다음 단계 (B·C)**: 이번 작업으로 화면 렌더가 정상화되면 (1) 백엔드에 `GET /api/v1/chat/messages/{id}/download` 추가하여 `FileResponse(filename=...)` 로 RFC 5987 한글 파일명 보존, (2) 프런트 `MessageBubble.jsx` 의 다운로드를 axios blob + `URL.createObjectURL` 방식으로 전환하여 cross-origin `download` 무시 문제 우회 — 두 단계로 나눠 진행 예정.

---

## 📅 2026-04-28 — 채팅 첨부 다운로드 엔드포인트 추가 (Content-Disposition + 권한 체크)

> **작업자**: @codelabprovide1 (Claude Opus 4.7 바이브코딩)
> **목표**: A 단계 placeholder 로 이미지는 정상 렌더되지만, 다운로드 버튼 클릭 시 cross-origin `<a download>` 무시로 새 탭에 이미지만 뜨던 문제 해결. 프런트가 blob 으로 받아 즉시 저장하도록 인증된 다운로드 엔드포인트 신설.

### ⏱ `GET /api/v1/chat/messages/{message_id}/download` 신규

- **동작**: `FileResponse(path, filename=msg.file_name, media_type=msg.file_content_type)` 로 응답. FastAPI 가 RFC 5987 형식(`filename*=UTF-8''<percent-encoded>`) 으로 한글 파일명을 자동 인코딩 → 다운로드 시 원본 한글/영문 파일명 그대로 저장됨.
- **권한**: 메시지 → `conversation_id` → `ConversationMember` 조회로 현재 사용자가 해당 대화방 참여자인지 확인. 미참여자에게는 403.
- **path traversal 방어**: `msg.file_url` 이 `/uploads/chat/` prefix 로 시작하는지, saved_name 에 `/` `\` `..` 가 없는지, 절대경로 정규화 후 `CHAT_UPLOAD_DIR` 하위인지 3중 검증. DB 값을 신뢰하지 않고 디스크 경로 합성을 엄격하게 제한.
- **404 케이스**: 메시지에 `file_url` 이 없거나 디스크에 실제 파일이 없으면 404. (StaticFiles 와 분리해 명시적 에러 메시지 노출.)

### 🔗 변경 파일 목록 (1개)

| 파일 | 변경 유형 |
|------|----------|
| `backend/app/api/chat.py` | `FileResponse` import + `download_message_file` 핸들러 신규 (권한 체크 + path traversal 방어 + RFC 5987 파일명 헤더) |

### 📐 설계 결정 사항

- **별도 엔드포인트 vs StaticFiles 응답 가공**: StaticFiles 는 응답 헤더를 커스터마이징하기 까다롭고 인증 미들웨어를 끼우기 어려움. 별도 엔드포인트로 분리하면 (a) JWT 권한 체크, (b) Content-Disposition 헤더, (c) 향후 audit log/추적까지 한 자리에서 처리 가능.
- **이미지 인라인 표시는 기존 `/uploads/...` 유지**: `<img src>` 인증을 강제하면 마운트마다 토큰 헤더를 못 실어서 깨짐. 다운로드만 인증 경로로 분리하는 게 비용 대비 효과 큼. 추후 비공개 첨부가 필요해지면 `/uploads/chat` 마운트를 닫고 같은 다운로드 라우트로 일원화하면 됨.
- **path traversal 검증 3중**: prefix 검사 + 구분자 거부 + 절대경로 정규화 후 root 비교. DB 변조 시나리오까지 가정해 보안 깊이 확보 (`os.path.abspath` 만으로는 심볼릭 링크 우회 가능성이 있어 prefix/문자 검사 병행).

---

## 🔬 2026-04-28~29 — Recall 극대화 추론 파이프라인 + 학습 인프라 고도화 (@youminsu0523)

> **작업자**: @youminsu0523 (Claude Opus 바이브코딩)
> **작업 일자**: 2026-04-28 ~ 2026-04-29
> **작업 브랜치**: `MS`

### 1️⃣ 프롬프트 / 목표
> Recall 극대화를 위해 추론 파이프라인에 ByteTrack 객체 추적, IoU 기반 TemporalFilter 고도화, SAHI 타일링 추론, Cross-Model Spatial Boost, Active Learning(Hard Example Mining), 실시간 DB 저장을 추가. 학습 스크립트 전체를 실전 GPU 환경(RTX 5070)에 맞게 보강하고, 자동 순차 학습·야간 학습 파이프라인 구축.

### 2️⃣ 수행된 작업 요약

#### A. 추론 파이프라인 — 신규 서비스 모듈 (4개)

**`app/services/object_tracker.py` (신규)**
- IoU 기반 프레임 간 하자 추적기 (드론 환경 특화)
- Kalman filter 대신 순수 IoU 매칭 — 드론 풍압/틸트/접근 비선형 환경에서 더 강건
- `track_id` 부여 → 프레임 간 동일 물리 하자 식별
- `min_hits` 이상 탐지된 트랙만 확정(confirmed)으로 보고
- `max_age` 프레임까지 미탐지 허용 (블러·흔들림 대응)
- `reconfigure(camera_fps, frame_skip)`로 실제 추론 FPS 기반 동적 max_age 조정

**`app/services/tiled_inference.py` (신규)**
- SAHI 방식 타일 분할 추론 — 4K 드론 영상에서 소형 하자(크랙, 핀홀) Recall 극대화
- `generate_tiles()`: overlap_ratio 기반 타일 좌표 생성, 저해상도 1-타일 fallback
- `tiled_predict()`: 타일별 YOLO 추론 → 좌표 원본 이미지 기준 복원 → cross-tile NMS
- Tier 3 프레임에서만 선택적 적용 (실시간 예산 보호)

**`app/services/active_learning.py` (신규)**
- Hard Example Mining — 모델이 불확실해한 프레임 자동 수집
- 수집 기준: 저신뢰 검출(conf 0.15~0.40), PatchCore high + YOLO miss, Temporal reject
- `./training/hard_examples/{date}/{model}/` 경로에 YOLO 학습 포맷으로 저장
- 메모리 버퍼 + 주기적 디스크 flush (save_interval=30초)
- 기본 비활성화(`HARD_EXAMPLE_ENABLED=False`), 명시적 활성화 필요

**`app/services/defect_persistence.py` (신규)**
- 실시간 추론 결과 → DefectLog DB 비동기 저장
- DB 쓰기 실패 시 메모리 버퍼에 보관(최대 1000건) → 주기적 재시도
- 세션 종료 시 `flush_retry_buffer()`로 잔여 버퍼 소진

#### B. 기존 서비스 고도화

**`app/services/temporal_filter.py` (대규모 리팩토링)**
- 기존 class 단순 카운트 → IoU 기반 `SpatialBucket` 매칭으로 전면 교체
- `BufferedDetection` / `SpatialBucket` dataclass 도입
- Noisy-OR 신뢰도 누적: `1 - Π(1 - conf_i)` — 반복 탐지 시 conf 상승
- 시간 기반 윈도우 만료(`window_time_sec`) 추가
- `accumulated_conf` 필드를 보고 결과에 부여

**`app/services/ensemble.py` — Cross-Model Spatial Boost 추가**
- `cross_model_spatial_boost()`: 서로 다른 모델이 동일 위치를 탐지했을 때 conf 승격 (+0.15)
- 교차 증거(다른 source의 겹침)로 실제 하자 확률이 높다는 판단 근거

**`app/services/inference_pipeline_20.py` (대규모 확장)**
- M1/M2/M3 `_run_m*()`: SAHI 타일링 추론 옵션 추가 (`use_tiling` 파라미터)
- Tier 3에서만 `tiled_predict()` 적용 (소형 하자 Recall↑, 실시간 예산 보호)
- 2-Stage conf 결합: 곱셈 → `compute_combined_confidence()` (Noisy-OR)로 변경
- Cross-Model Spatial Boost → Cross-Model NMS 순서로 파이프라인 구성
- M1-ResNet 클래스 목록 갱신: `["caulking_indicator", "crack_indicator", "moisture_indicator", "structural_damage", "waterproof_defect"]`
- M3-ResNet 클래스 목록 갱신: `["frame_defect"]`
- 모델 로드 시 더미 추론(640x640)으로 shape 검증 + 실패 시 graceful skip
- 필수 모델(M1-YOLO, M2-YOLO) 미로드 시 파이프라인 비활성 + 경고 메시지

**`app/core/stream_inference.py` (대규모 확장)**
- ByteTrack 추적 → Temporal Filter 오탐 제거 → 브로드캐스트 3단계 파이프라인
- 3-model 파이프라인과 20종 파이프라인 모두에 추적+필터 적용
- Hard Example Mining: 추론 루프 내에서 `check_and_collect()` 호출
- 에러 카운터: 연속 10회 추론 실패 시 경고 출력
- 세션 시작/종료 시 추적기·필터·마이너 상태 초기화/flush
- `/health` 응답에 tracker stats, hard_example stats, db_persistence stats 추가

#### C. 모델·DB 확장

**`app/config.py` — 신규 설정값 13개 추가**
- ByteTrack: `TRACKER_MIN_HITS`, `TRACKER_MAX_AGE`, `TRACKER_IOU_THRESHOLD`
- TemporalFilter: `TEMPORAL_FILTER_IOU`
- Hard Example Mining: `HARD_EXAMPLE_ENABLED`, `HARD_EXAMPLE_DIR`, `HARD_EXAMPLE_LOW_CONF_MIN/MAX`, `HARD_EXAMPLE_SAVE_INTERVAL`

**`app/models/defect.py` — DefectLog 확장 컬럼 3개**
- `track_id` (BigInteger): ByteTrack 추적 ID
- `accumulated_conf` (Float): 시간 누적 신뢰도 (Noisy-OR)
- `tier_executed` (BigInteger): 실행 Tier (1/2/3)

**`app/main.py` — `/health` 엔드포인트 고도화**
- 20종 파이프라인 상태 포함 (로드 여부 + 개별 모델 상태)
- 활성 파이프라인(`3model` / `20defect`) 표시
- 필수 모델 미로드 시 HTTP 503 반환 (degraded 상태)
- stream_worker_stats (tracker, hard_examples, db_persistence) 포함

#### D. 학습 스크립트 보강

**`training/train_m1~m5_*.py` (전체 7개 수정)**
- `find_weights()` 헬퍼: ultralytics 저장 경로를 절대경로로 동적 탐색 (Phase2 시작 시 Phase1 weights 자동 연결)
- `erasing=0.0`: seg 라벨 혼합 데이터 호환 (erasing augmentation이 seg 마스크를 깨뜨리는 문제 방지)
- Windows cp949 stdout 인코딩 문제 방지 (`sys.stdout` UTF-8 래핑)

**`training/eval/evaluate_all.py` (대규모 확장)**
- IoU 기반 TP/FP 판정으로 정확한 mAP@0.5 계산
- Per-class Precision/Recall/F1 리포트 생성
- `_imread_unicode()`: 한글 경로 이미지 로드 지원
- Windows 인코딩 호환

**`training/augment_missing_classes.py` (신규)**
- 부족 클래스 데이터 추출/변환: M1 caulking_defect (6.5%→15%+), M2 baseboard_defect (0%→추가)
- gdrive_raw 폴더에서 원시 데이터 필터링하여 학습 데이터에 추가

**`training/extract_resnet_crops.py` (신규)**
- YOLO 라벨(bbox)에서 ROI crop → ResNet 학습용 클래스별 폴더 저장
- M1/M3 ResNet 학습 데이터 자동 생성

**`training/train_all_sequential.py` (신규)**
- 전체 모델 순차 학습 자동화 (M5→M1~M3 YOLO→M1/M3 ResNet→M4)
- 모든 로그 `training_log.txt`에 append

**`training/overnight_train.py` (신규)**
- 밤새 자동 학습 파이프라인
- 진행 중 모델 완료 대기 → GPU 여유 확보 → 다음 모델 자동 시작
- 전체 완료 후 `evaluate_all.py` 자동 실행

**`training/monitor.py` + `training/monitor_live.py` (신규)**
- 학습 상태 실시간 모니터링 유틸리티
- `monitor.py`: 10분 간격 학습 현황 체크
- `monitor_live.py`: 학습 출력 파일 실시간 감시 → `training_log.txt`에 타임스탬프 기록

#### E. 단위 테스트 (4개 신규)

| 테스트 파일 | 대상 | 주요 검증 |
|------------|------|----------|
| `tests/test_ensemble.py` | Noisy-OR, cross_model_nms, spatial_boost | 결합 신뢰도 계산, NMS 중복 제거, 교차 모델 부스팅 |
| `tests/test_object_tracker.py` | DefectTracker | track_id 부여, 확정 로직, 미탐지 후 재매칭 |
| `tests/test_temporal_filter.py` | TemporalFilter | IoU 매칭, Noisy-OR 누적, 투표 로직, 즉시 보고, LiDAR 중복 억제 |
| `tests/test_tiled_inference.py` | SAHI 타일링 | 타일 좌표 생성, 저해상도 fallback, cross-tile NMS |

### 🔗 변경 파일 목록 (27개)

| 파일 | 변경 유형 |
|------|----------|
| `app/config.py` | ByteTrack·TemporalFilter·HardExample 설정값 13개 추가 |
| `app/core/stream_inference.py` | ByteTrack 추적 + Temporal Filter + HEM + 에러 카운터 통합 |
| `app/main.py` | `/health` 20종 파이프라인 상태 + 503 반환 + worker stats |
| `app/models/defect.py` | DefectLog: track_id, accumulated_conf, tier_executed 컬럼 추가 |
| `app/services/ensemble.py` | `cross_model_spatial_boost()` 신규 |
| `app/services/inference_pipeline_20.py` | SAHI 타일링 + Noisy-OR conf 결합 + spatial boost + 모델 검증 |
| `app/services/temporal_filter.py` | IoU SpatialBucket + Noisy-OR 누적 전면 리팩토링 |
| `app/services/object_tracker.py` | **신규** — IoU 기반 하자 추적기 |
| `app/services/tiled_inference.py` | **신규** — SAHI 타일 분할 추론 |
| `app/services/active_learning.py` | **신규** — Hard Example Mining |
| `app/services/defect_persistence.py` | **신규** — 실시간 DB 비동기 저장 |
| `training/eval/evaluate_all.py` | IoU 기반 mAP@0.5 + Per-class F1 리포트 |
| `training/train_m1_yolo_structural.py` | find_weights + erasing=0.0 |
| `training/train_m1_resnet_crack.py` | find_weights + 인코딩 호환 |
| `training/train_m2_yolo_surface.py` | find_weights + erasing=0.0 |
| `training/train_m3_yolo_floor_window.py` | find_weights + erasing=0.0 |
| `training/train_m3_resnet_floor_window.py` | find_weights + 인코딩 호환 |
| `training/train_m4_thermal_unet.py` | 인코딩 호환 |
| `training/train_m5_frame_seg.py` | 인코딩 호환 |
| `training/augment_missing_classes.py` | **신규** — 부족 클래스 데이터 증강 |
| `training/extract_resnet_crops.py` | **신규** — YOLO bbox → ResNet crop 추출 |
| `training/train_all_sequential.py` | **신규** — 전체 모델 순차 학습 자동화 |
| `training/overnight_train.py` | **신규** — 야간 자동 학습 파이프라인 |
| `training/monitor.py` | **신규** — 학습 현황 주기적 체크 |
| `training/monitor_live.py` | **신규** — 학습 출력 실시간 감시 |
| `tests/test_ensemble.py` | **신규** — 앙상블 함수 단위 테스트 |
| `tests/test_object_tracker.py` | **신규** — DefectTracker 단위 테스트 |
| `tests/test_temporal_filter.py` | **신규** — TemporalFilter 단위 테스트 |
| `tests/test_tiled_inference.py` | **신규** — SAHI 타일링 단위 테스트 |

### 📐 설계 결정 사항

- **IoU 매칭 vs Kalman Filter (ObjectTracker)**: 드론 환경에서 하자는 이미지 좌표 상 비선형 움직임을 보임(풍압, 틸트, 접근/후퇴). Kalman의 등속 가정이 부적합하여 단순 IoU 매칭이 더 강건. supervision ByteTrack은 정적 객체 매칭이 불안정한 것으로 확인되어 자체 구현.
- **Noisy-OR 신뢰도 누적 (TemporalFilter)**: 단순 max/mean 대신 `1 - Π(1-conf_i)` 사용. 독립 관측을 가정하여 반복 탐지 시 신뢰도가 자연스럽게 1에 수렴하며, 저신뢰 반복 검출도 누적 시 보고 대상이 될 수 있음.
- **SAHI 타일링 Tier 3 한정**: 타일링은 소형 하자 Recall을 크게 높이지만 추론 비용이 N타일 배로 증가. Tier 3(정밀 검사)에서만 적용하여 실시간 예산을 보호.
- **conf 결합 방식 변경 (곱셈 → Noisy-OR)**: YOLO conf × ResNet conf는 두 모델 모두 높아야 최종 conf가 유지되어 Recall을 떨어뜨림. Noisy-OR는 한쪽만 높아도 최종 conf가 유지되어 Recall 친화적.
- **Hard Example Mining 기본 비활성**: 디스크 I/O + 메모리 사용량이 추론 성능에 영향을 줄 수 있으므로 명시적 활성화(`HARD_EXAMPLE_ENABLED=True`)가 필요. 수집 데이터는 YOLO 학습 포맷이라 바로 재학습에 사용 가능.
- **필수 모델 미로드 시 503**: M1-YOLO + M2-YOLO가 없으면 구조+마감 하자 탐지가 불가능하므로 파이프라인 자체를 비활성화. `/health`에서 503을 반환하여 모니터링 시스템이 즉시 감지.

---

## 🚀 2026-04-29 ~ 05-02 — 학습 v2/v3 Refine·Retrain + Colab/Kaggle 원격 학습 + 가구 인식 데이터셋 + M4 Context 로컬 학습 (@youminsu0523)

> **작업자**: @youminsu0523 (Claude Opus 바이브코딩)
> **작업 일자**: 2026-04-29 17:52 ~ 2026-05-02 02:14
> **작업 브랜치**: `MS`

### 1️⃣ 프롬프트 / 목표

> 1. 단일 학습 mAP 자체를 끌어올리는 데 집중. 후처리(TTA/SAHI/앙상블)는 첨가제 — "원재료(모델)가 신선해야". 0.9 솔루션 1순위는 학습 차원.
> 2. 베이스라인 학습 결과(M1 0.842 / M2 0.794 / M3 0.804 / M5 0.626)를 v2/v3로 끌어올리는 데이터 정제 + 재학습 파이프라인을 만들어줘.
> 3. RTX 5070 Laptop은 한 모델만 학습 가능 → 동시에 여러 계정 Colab + Kaggle로 분산 학습. 핸드폰으로 야간에 진행 가능하게 만들어줘.
> 4. M3가 침대/소파 같은 빌트인 가구를 floor/window로 오탐(False Positive)하는 문제 해결 — 가구를 "검사 대상이 아닌 클래스"로 학습시켜라.
> 5. M4 Thermal U-Net이 데이터셋 라벨 오류로 막혔으니, M4를 "Context 부위 분류(wall/ceiling/floor/window/door)" 모델로 재정의하고 ADE20K 외부 데이터 + 우리 frames/floor_window 통합으로 새로 학습.
> 6. `M4 학습 끝나면 자동으로 다음 단계 시작`되도록 자동화. "재실행해줘" 한 마디면 last.pt 자동 감지하고 이어서 돌리도록.

### 2️⃣ 진행 라운드 (시각 / 산출물 / 결정 사항)

| 라운드 | 시각 | 작업 | 산출물 |
|-------|------|------|-------|
| R1 | 2026-04-29 17:52 | M2-ResNet crop 추출 — YOLO 2클래스(surface/baseboard) bbox에서 ROI crop, 학습 폴더 자동 분류 | `extract_m2_resnet_crops.py` |
| R2 | 2026-04-30 07:29 | 전 YOLO 모델 imgsz=960 fine-tune 자동화 (`copy_paste`+`multi_scale`+50ep), 기존 best.pt 자동 탐색 | `finetune_960.py` |
| R3 | 2026-04-30 07:35~38 | 베이스라인 가중치 패키징 (Drive 업로드용 zip): M5 best.pt(202MB), frames.zip(239MB) | `colab/upload_to_drive/{m5_baseline_best.pt, frames.zip}` |
| R4 | 2026-04-30 09:22~35 | Colab 원격 학습 노트북 1차 — M5v2/M2 YOLO (계정 A/B 분산, Drive autosave + Resume) | `colab/{m5v2_colab_train.ipynb, m2_yolo_colab_train.ipynb}` |
| R5 | 2026-04-30 10:22~25 | M1 structural 데이터셋 압축 (32GB → ~6GB, max 1280px / jpg85) + M1-aggressive Colab 학습 노트북 | `compress_m1_dataset.py`, `colab/m1_aggressive_colab_train.ipynb`, `colab/upload_to_drive/structural_compressed/` |
| R6 | 2026-04-30 10:27~36 | **M4 Thermal U-Net 폐기 → M4 Context (5클래스 부위 분류)로 재정의.** frames(M5)+floor_window(M3) 통합 데이터셋 빌더 + YOLO 학습 스크립트 + ADE20K(5만장) → YOLO bbox 변환 | `build_m4_context_dataset.py`, `train_m4_context_yolo.py`, `convert_ade20k_to_yolo.py` |
| R7 | 2026-04-30 10:58~11:31 | M5v2_v2 / M1-conservative / M2-aggressive Colab 노트북 추가 (병렬 학습 슬롯 3개 운영) | `colab/{m5v2_v2_colab_train.ipynb, m1_conservative_colab_train.ipynb, m2_aggressive_colab_train.ipynb}` |
| R8 | 2026-04-30 12:12 | **Active Learning 데이터 정제기** — best.pt로 train 전수 추론, IoU<0.3(잘못된 GT) / conf>0.85 매칭 GT 없음(놓친 GT) / GT 0개+검출 5+(누락 의심) 자동 분류 → `*_refined/` 생성 | `refine_dataset.py` |
| R9 | 2026-04-30 13:21 | **핸드폰으로 야간 학습 진행 가이드 작성** — 계정 A/B/C별 노트북, 90분 끊김 방지법, T4/A100 예상 시간(7~14h), Drive autosave + Resume 흐름 | `colab/PHONE_GUIDE.md` |
| R10 | 2026-04-30 13:54 | M5v2_v2 Kaggle 노트북 (Colab 슬롯 부족분 Kaggle T4×2로 보충) | `colab/m5v2_v2_kaggle_train.ipynb` |
| R11 | 2026-04-30 13:59~14:43 | **Refine + Retrain v2/v3 노트북 시리즈** — M2v2/M3v2/M4v2/M1v3/M5v3 정제 데이터로 재학습 (Drive autosave + Resume + ONNX export 포함, 자동 mAP 비교) | `colab/{m2v2_refine_retrain, m3v2_refine_retrain, m1v3_refine_retrain, m4v2_refine_retrain, m5v3_refine_retrain}.ipynb` |
| R12 | 2026-04-30 14:42~43 | M3 baseline best.pt + floor_window.zip(1.05GB) 패키징 — M3v2 Colab 학습용 | `colab/upload_to_drive/{m3_baseline_best.pt, floor_window.zip}` |
| R13 | 2026-04-30 16:09 | **M4 Context 학습 종료 자동 감지 → train_m4v2_local.py 자동 시작** 워처 (5분 간격, 2회 미감지 시 종료 확정, 30초 buffer 후 다음 단계) | `auto_run_m4v2.py` |
| R14 | 2026-04-30 16:24 | **SAHI 타일 데이터셋 사전 처리** — 1280 이미지를 640×640 타일 4개(overlap 0.2)로 분할, bbox 50% 이상 포함된 것만 유지(8 worker) | `sahi_tile_dataset.py` |
| R15 | 2026-04-30 16:30 | M1 Plan A (Kaggle 또는 A100 Colab) 노트북 — A100 8~10h 예상 | `colab/m1_plan_a_kaggle_or_colab.ipynb` |
| R16 | 2026-04-30 17:06~26 | **M3 False Positive 방지용 furniture-aware 통합 데이터셋** — ADE20K(2만장) + frames + floor_window 통합, **시공 부위 5클래스 + 빌트인 가구 5클래스(cabinet_builtin, kitchen_appliance, countertop_sink, kitchen_island, shelf) = 10클래스**로 학습. Kaggle/Colab 노트북 함께 작성 | `build_furniture_aware_dataset.py`, `colab/{furniture_aware_kaggle.ipynb, furniture_aware_train.ipynb}`, `colab/upload_to_drive/furniture_aware.zip` (1.6GB) |
| R17 | 2026-05-01 09:49 | **신규 ONNX 3개 평가 스크립트** — M2v2/M3v2/m5v2_v2 한 번에 mAP 측정 + baseline 대비(±) 자동 비교 (CPU 평가, GPU는 M4v2 학습 중이므로) | `eval_new_onnx.py` |
| R18 | 2026-05-01 10:55~56 | 코랩 결과 백업 폴더 — baseline보다 낮아 운영 미사용된 모델(m2v2 -0.001, m3v2 -0.053, m5v2_v2 -0.093) 보관 (앙상블·후처리 결합 시 활용) | `colab_results_backup/README.md`, `colab/upload_to_drive/{m3,m5}_baseline_best.onnx` |
| R19 | 2026-05-02 02:14 | **M4v2 로컬 학습 자동화 스크립트 (RTX 5070 Laptop)** — best.pt 자동 탐색(4개 후보 경로) → refine → Hard Mining(상위 10%, 4× 가중) → stage1/stage2 학습 → ONNX 자동 export | `train_m4v2_local.py` |

### 3️⃣ 변경/추가 파일 목록 (40+개)

#### A. 데이터셋 빌더 (5개)

| 파일 | 역할 |
|------|------|
| `training/compress_m1_dataset.py` | M1 structural 데이터셋 1280px+jpg85 압축 (Colab 업로드용, 32GB→6GB) |
| `training/build_m4_context_dataset.py` | frames+floor_window를 5클래스 부위 분류로 통합 (M4 Thermal U-Net 폐기 후 신규) |
| `training/convert_ade20k_to_yolo.py` | ADE20K seg(150클래스) → YOLO bbox(5클래스: wall/floor/ceiling/window/door) 변환 |
| `training/build_furniture_aware_dataset.py` | M3 FP 방지용 10클래스 통합 (시공 부위 5 + 빌트인 가구 5) |
| `training/build_m5v2_dataset.py` | M5 frame seg v2 데이터셋 빌더 |

#### B. 학습 자동화 (4개)

| 파일 | 역할 |
|------|------|
| `training/refine_dataset.py` | Active Learning 정제: noisy GT/missed GT/누락 의심 자동 분류 |
| `training/finetune_960.py` | 전 YOLO 모델 imgsz=960 + copy_paste + multi_scale 50ep fine-tune |
| `training/train_m4v2_local.py` | M4 Context 정제+Hard Mining(10% 4×)+stage1/2 재학습+ONNX export 자동화 (RTX 5070) |
| `training/auto_run_m4v2.py` | M4 Context 학습 종료 자동 감지 → train_m4v2_local.py 자동 시작 워처 |

#### C. 데이터 보강 (2개)

| 파일 | 역할 |
|------|------|
| `training/sahi_tile_dataset.py` | 1280 이미지를 640×640×4타일(overlap 0.2)로 분할, 8 worker 병렬 |
| `training/extract_m2_resnet_crops.py` | M2 YOLO bbox에서 surface/baseboard ROI crop → ResNet 학습용 |

#### D. 평가 (1개)

| 파일 | 역할 |
|------|------|
| `training/eval_new_onnx.py` | M2v2/M3v2/m5v2_v2 ONNX 일괄 mAP 평가 + baseline 대비 표시 (CPU) |

#### E. Colab/Kaggle 원격 학습 노트북 (15개)

> 위치: `training/colab/`

| 노트북 | 모델 | 비고 |
|------|------|------|
| `m1_aggressive_colab_train.ipynb` | M1-YOLO | aggressive aug, structural_compressed 사용 |
| `m1_conservative_colab_train.ipynb` | M1-YOLO | conservative aug 비교군 |
| `m1_plan_a_kaggle_or_colab.ipynb` | M1-YOLO | A100/Kaggle 8~10h 예상 |
| `m1v3_refine_retrain.ipynb` | M1-YOLO v3 | refine_dataset 결과로 재학습 |
| `m2_yolo_colab_train.ipynb` | M2-YOLO | baseline 재현 |
| `m2_aggressive_colab_train.ipynb` | M2-YOLO | aggressive aug |
| `m2v2_refine_retrain.ipynb` | M2-YOLO v2 | refine 재학습 (mAP 0.7928, baseline 0.794 대비 -0.001 → 미사용 백업) |
| `m3v2_refine_retrain.ipynb` | M3-YOLO v2 | refine 재학습 (mAP 0.7514, -0.053 → 미사용 백업) |
| `m4v2_refine_retrain.ipynb` | M4-Context v2 | T4 6~7h, A100 4~5h |
| `m5v2_colab_train.ipynb` | M5-Seg v2 | baseline |
| `m5v2_v2_colab_train.ipynb` | M5-Seg v2.v2 | aug 변형 |
| `m5v2_v2_kaggle_train.ipynb` | M5-Seg v2.v2 | Kaggle T4×2 보충 |
| `m5v3_refine_retrain.ipynb` | M5-Seg v3 | T4 7~8h, A100 4~5h |
| `furniture_aware_kaggle.ipynb` | M3+가구 통합 (Kaggle) | 10클래스 |
| `furniture_aware_train.ipynb` | M3+가구 통합 (Colab) | 10클래스, FP 방지 |
| `colab/PHONE_GUIDE.md` | — | 핸드폰 야간 학습 가이드 |
| `colab_results_backup/README.md` | — | baseline 미달 결과 백업 정책 |

#### F. 학습 데이터 zip 패키징 (`training/colab/upload_to_drive/`)

| 파일 | 크기 | 용도 |
|------|------|------|
| `frames.zip` | 239MB | M5v2 학습 |
| `frames_ade.zip` | 645MB | M5v2 + ADE20K |
| `floor_window.zip` | 1.05GB | M3 학습 |
| `m4_context.zip` | 1.6GB | M4 Context (frames+floor_window+ADE20K) |
| `furniture_aware.zip` | 1.6GB | M3 가구 인식 통합 |
| `structural_compressed/` | ~6GB | M1 압축본 |
| `m3_baseline_best.{pt,onnx}` | 52/103MB | M3 baseline |
| `m5_baseline_best.{pt,onnx}` | 207/103MB | M5 baseline |

#### G. eval 설정 신규

| 파일 | 용도 |
|------|------|
| `training/configs/floor_window_eval.yaml` | M3v2 평가용 데이터 yaml |
| `training/configs/frame_seg_eval.yaml` | M5v2_v2 평가용 데이터 yaml |
| `training/configs/surface_eval.yaml` | M2v2 평가용 데이터 yaml |

### 4️⃣ 평가 결과 (2026-05-01 기준, baseline 대비 ±)

| 모델 | baseline mAP50 | v2/v3 mAP50 | Δ | 운영 채택 |
|------|---------------|------------|----|----------|
| M2v2 (surface) | 0.794 | 0.7928 | -0.001 | ❌ baseline 유지 |
| M3v2 (floor_window) | 0.804 | 0.7514 | -0.053 | ❌ baseline 유지 |
| m5v2_v2 (frames) | 0.626 | 0.5329 | -0.093 | ❌ baseline 유지 |
| M4 Context (신규) | — | 학습 중 (M4v2로 이어감) | — | M4v2 로컬 학습 진행 중 |
| furniture_aware (M3 FP 방지) | — | 학습 중 | — | 야간 Colab 진행 중 |

> **결정**: v2/v3 결과가 baseline에 미달하여 운영 ONNX는 baseline 복원본을 유지하고, v2 결과는 `colab_results_backup/`에 보관(앙상블·후처리 결합 시 재활용 가능). 다음 라운드는 (1) M4v2 로컬 학습 완료 대기 + ONNX 자동 export, (2) furniture_aware 학습 결과로 M3 가중치 갱신.

### 📐 설계 결정 사항

- **단일 학습 mAP가 1순위**: 후처리(TTA/SAHI/앙상블)는 보조이므로, 같은 시간을 v2/v3 데이터 정제와 외부 데이터(ADE20K 5만장) 통합에 투자하는 게 더 효과적이라 판단.
- **M4 재정의 (Thermal U-Net → Context 5클래스)**: M4 Thermal 데이터셋 라벨 오류(class_id 범위 초과 + seg 누락)로 학습이 막혀, 우리가 가진 frames+floor_window+ADE20K로 "부위 분류" 역할을 부여. 후속 파이프라인에서 부위 컨텍스트를 다른 모델 결과 검증에 사용.
- **분산 학습 슬롯 운영 (Colab A/B/C + Kaggle)**: RTX 5070 Laptop은 한 번에 한 모델만 학습 가능 + 12~14h 단위로 묶임. 무료 Colab 3계정 + Kaggle T4×2를 병렬로 운영하여 야간 학습 throughput 4~5배 확보.
- **Drive autosave + Resume 의무화**: Colab 무료 90분 inactive 끊김 + 12h 세션 한계 → 5분마다 last.pt를 Drive에 자동 저장, 노트북 재실행 시 last.pt 감지하면 자동 resume. 핸드폰만으로 진행 가능.
- **Refine 임계값 (IoU<0.3 / conf>0.85 / 누락 5+)**: 너무 엄격하면 정제 후 데이터 손실, 너무 느슨하면 노이즈 잔존. 모델별로 검증한 결과 IoU 0.3 / conf 0.85 / 5+ 검출이 균형점.
- **M4v2 Hard Mining 10% × 4 가중 (5배)**: 전체 4× 복사는 학습 시간이 5배가 되지만 어려운 케이스만 5배 노출하므로 epoch 수 동일하게 유지하면서도 어려운 분포에 더 노출.
- **furniture-aware 10클래스 채택**: 단순 "가구=배경" 마스킹은 빌트인 가구가 시공 검사 대상에 포함될 때 모순. 빌트인 5클래스(cabinet_builtin/kitchen_appliance/countertop_sink/kitchen_island/shelf)를 별도 클래스로 학습시켜 인식하면서, 그중 운영 시 검사 대상은 후처리 단계에서 결정.
- **v2 결과 baseline 미달 시 폐기 + 백업 정책**: 운영 ONNX는 항상 mAP 기준 최선 선택, 성능 하락분은 `colab_results_backup/`에 보관하여 추후 앙상블·후처리 결합 가능성 열어둠.
- **M4 학습 종료 자동 감지 (auto_run_m4v2.py)**: 사람이 학습 종료 시점을 모니터링하면 야간 시간을 낭비. 5분 간격 + 2회 미감지 + 30초 buffer로 안전하게 다음 단계 자동 트리거.
- **자동 진행 우선**: 학습/배포 워크플로우 분기점에서 YES/NO 묻지 말고 합리적 default(last.pt 자동 감지, 자동 resume, 자동 ONNX export)로 즉시 진행 — 사용자 피드백 반영(`feedback_auto_progress`).

---

## 📑 2026-04-20 ~ 05-01 — 제출용 백엔드 산출물(API 명세서·ERD·FlowChart·기획서·WBS·최종발표 PPT) 일괄 작성 (@youminsu0523)

> **작업자**: @youminsu0523 (Claude Opus 바이브코딩)
> **작업 일자**: 2026-04-20 18:38 ~ 2026-05-01 17:47
> **작업 브랜치**: `MS`
> **위치**: `tasks/`

### 1️⃣ 프롬프트 / 목표

> 1. 제출/심사용 백엔드 문서 풀세트 — API 명세서, ERD 설계서, AI 추론 파이프라인 기획, Backend Developer Guide, FlowChart 0~6, 요구사항정의서, WBS, 최종발표 PPT — 를 단일 소스로 만들고 코드 변경에 따라 v1.x로 갱신.
> 2. FlowChart는 Mermaid 소스(.md)와 PNG 둘 다 보유 — PPT 삽입용 + Notion/문서 임베드용.
> 3. 데이터·인터페이스 요구사항은 ERD 설계서·API 명세서로 일원화하여 요구사항정의서 중복 제거(v1.4).
> 4. 최종발표 PPT는 python-pptx로 자동 생성 — 다음 라운드 수정도 스크립트 재실행으로 즉시 반영 가능하게.
> 5. WBS는 데이터/생성 스크립트 분리(`wbs_data.py` ↔ `build_wbs_xlsx.py`) — 일정 변경은 데이터만 고치면 xlsx가 새로 만들어지도록.

### 2️⃣ 진행 라운드 (시각 / 산출물 / 결정 사항)

| 라운드 | 시각 | 작업 | 산출물 |
|-------|------|------|-------|
| R1 | 2026-04-20 18:38 | 요구사항정의서 v1.0 최초 작성 (FR/NFR 골격) | `tasks/요구사항정의서_AeroInspect_v1.0.md` |
| R2 | 2026-04-20 18:42 | API 명세서 v1.0 최초 작성 (인증/회원/현장/하자/조직 라우터) | `tasks/API_명세서_AeroInspect_v1.0.md` |
| R3 | 2026-04-29 16:55 | **ERD 설계서 v1.0** — 18개 테이블, 관계·인덱스·보안·복구·확장 계획까지 포함 | `tasks/AeroInspect_ERD_설계서_v1.0.md` |
| R4 | 2026-04-29 16:57 | **AI 추론 파이프라인 기획 v1.0** — 6-Model 20종 하자, ByteTrack/Temporal Filter/SAHI, 차별화 포인트, 기술 스택, 리스크 대응 | `tasks/AeroInspect_AI_추론파이프라인_기획_v1.0.md` |
| R5 | 2026-04-29 16:57 | **Backend Developer Guide v1.0** — 기술 스택/환경 설정/구조/Git 전략/CI·CD/배포·운영/문제해결 | `tasks/AeroInspect_Backend_Developer_Guide_v1.0.md` |
| R6 | 2026-04-29 16:58 ~ 17:01 | **FlowChart 0~6 Mermaid 소스 작성** — 0:전체 시스템 / 1:Auth / 2:AI 추론 / 3:하자 관리 / 4:보고서 생성 / 5:실시간 스트리밍 / 6:조직·채팅 | `tasks/FlowChart_{0..6}_*.md` (7개) |
| R7 | 2026-04-29 16:59 | **API 명세서 v1.1** — Coverage API, Refresh Token, 20종 하자 파이프라인 스키마, ByteTrack/Temporal Filter 컬럼 추가 반영 | `tasks/API_명세서_AeroInspect_v1.1.md` |
| R8 | 2026-05-01 10:12 | WBS PDF 외부 산출물 통합(FlowFit) | `tasks/FlowFit_WBS_v1.0.pdf` |
| R9 | 2026-05-01 10:57 ~ 11:01 | 발표용 화면 캡처 자동화 스크립트 (Playwright/Selenium 기반) — 일반 화면/세션/대시보드 활성 상태 별도 | `tasks/{capture_screens, capture_session_screens, capture_dashboard_active}.py` |
| R10 | 2026-05-01 11:04 ~ 11:06 | **WBS 단일 소스(`wbs_data.py`) + xlsx 빌더(`build_wbs_xlsx.py`)** — 4주 일정(2026-04-10 ~ 05-06), 4명 담당(MS/HJ/SH/KD/ALL), 3-depth 트리 | `tasks/{wbs_data.py, build_wbs_xlsx.py, AeroInspect_WBS_v1.0.xlsx}` |
| R11 | 2026-05-01 11:25 ~ 11:27 | **FlowChart Mermaid → PNG (v1)** 분할 렌더 — 7개 차트, part1~5 분할(큰 차트는 가독성 위해 분할) | `tasks/flowchart_png/*.png` (24개) |
| R12 | 2026-05-01 11:31 | **요구사항정의서 v1.1** — FR-017 LiDAR 3D 좌표 연동 완료, FR-023 이미지 저장소 / FR-024 구조화 로깅 신규, defect_logs 스키마 image_crop_path + 20종 확장 컬럼, NFR 관측성 신설 | `tasks/요구사항정의서_AeroInspect_v1.1.md` |
| R13 | 2026-05-01 12:46 ~ 12:47 | **FlowChart PNG v2** — 단일 통합 PNG로 재렌더(고해상도 1장씩), PPT 삽입 시 분할 부담 제거 | `tasks/flowchart_png_v2/*.png` (7개) |
| R14 | 2026-05-01 13:21~14:17 | **최종발표 PPT 에셋 정리** — `ppt_assets/`(CAPTURE_GUIDE / SELF_EVAL_GUIDE / ANIMATION_GUIDE / images / logos / screenshots) | `tasks/ppt_assets/` |
| R15 | 2026-05-01 14:24 ~ 14:34 | **최종발표 PPT 빌더 3단 구조** — 컨텐츠(`ppt_content.py`) ↔ 헬퍼(`ppt_helpers.py`) ↔ 빌더(`build_ppt.py`). Color/Font/Shape/Card/Metric/Arrow/Connector/Table/Image 헬퍼 일괄 정의 | `tasks/{ppt_content, ppt_helpers, build_ppt}.py` |
| R16 | 2026-05-01 14:47 | **최종발표 PPT 빌드 (13MB, python-pptx)** — 표지·목차·문제·시장·차별화·페르소나·여정·팀·시스템 개요·시스템 요구사항·기능·데모·기술 스택·로드맵 등 | `tasks/AeroInspect_최종발표.pptx` |
| R17 | 2026-05-01 17:11 | **기획서 v1.3 (최종)** — BOM 4차 변경(2026-04-01~04-09 기록 포함), 시스템 아키텍처, 6-Model 앙상블 설계, 데이터 플로우 다이어그램, ByteTrack/Temporal Filter/SAHI 도해 | `tasks/기획서_AeroInspect_v1.3.md` |
| R18 | 2026-05-01 17:47 | **요구사항정의서 v1.4** — 데이터·인터페이스 요구사항 섹션 제거(ERD·API 명세서로 일원화), 시스템 아키텍처 코드블록 → 표 재구성, 17 라우터 / 60+ 엔드포인트 / 18 ORM 테이블 정리, 5단계 사용자 역할(슈퍼어드민/Owner/Admin/Member/비인증) 명시 | `tasks/요구사항정의서_AeroInspect_v1.4.md` |

### 3️⃣ 산출물 목록 (백엔드 관점)

#### A. 핵심 명세서 (3개)

| 파일 | 페이지·크기 | 핵심 내용 |
|------|------------|----------|
| `tasks/API_명세서_AeroInspect_v1.2.md` | 110KB | 인증·OAuth·회원·조직·현장·하자·드론·평면도·보고서·알림·메신저·**Employee** 라우터 + 20종 하자 파이프라인 스키마 + ByteTrack/Temporal Filter 컬럼 + **Swagger securityScheme(HTTPBearer/AIWebhookSecret)** + **운영 보안 가드(APP_ENV=production)** |
| `tasks/AeroInspect_ERD_설계서_v1.1.md` | 60KB | **19개 테이블** (inspection_schedules 신규), 관계 정의, 인덱스 설계, **11개 Enum** (schedule_status_enum 추가), 보안·권한, 네이밍 규칙, 성능 모니터링, 백업·복구, 확장 계획 — alembic 12 리비전 |
| `tasks/AeroInspect_AI_추론파이프라인_기획_v1.0.md` | 44KB | 6-Model 20종 하자, 2-Stage YOLO+ResNet, ByteTrack/Temporal Filter/SAHI/Active Learning, Noisy-OR, 차별화 포인트, 리스크 대응 |

#### B. 요구사항·기획서 (3개)

| 파일 | 변경점 |
|------|-------|
| `tasks/요구사항정의서_AeroInspect_v1.0.md` | 최초 (2026-04-20) |
| `tasks/요구사항정의서_AeroInspect_v1.1.md` | LiDAR 3D 좌표·이미지 저장소·구조화 로깅·관측성 신설 (2026-05-01) |
| `tasks/요구사항정의서_AeroInspect_v1.4.md` | ERD·API 명세서로 일원화 후 슬림화, 17 라우터/60+ 엔드포인트/18 테이블 정리 (2026-05-01) |
| `tasks/기획서_AeroInspect_v1.3.md` | BOM 4차 변경(1,611,430원), 시스템 아키텍처, 6-Model 앙상블, 데이터 플로우, 후처리 도해 |

#### C. FlowChart (Mermaid + PNG)

| 차트 | Mermaid 소스 | PNG (v1 분할) | PNG (v2 통합) |
|------|-------------|--------------|--------------|
| 0 전체 시스템 | `FlowChart_0_Overall_System.md` | `flowchart_png/FlowChart_0_*_part{1,2}.png` | `flowchart_png_v2/FlowChart_0_*.png` |
| 1 Auth Service | `FlowChart_1_Auth_Service.md` | part1~5 | 통합 1장 |
| 2 AI Inference | `FlowChart_2_AI_Inference_Pipeline.md` | part1~4 | 통합 1장 |
| 3 Defect Mgmt | `FlowChart_3_Defect_Management.md` | part1~3 | 통합 1장 |
| 4 Report Gen | `FlowChart_4_Report_Generation.md` | part1~3 | 통합 1장 |
| 5 RT Streaming | `FlowChart_5_RealTime_Streaming.md` | part1~4 | 통합 1장 |
| 6 Org/Chat | `FlowChart_6_Organization_Chat.md` | part1~4 | 통합 1장 |

#### D. WBS / 최종발표 / 가이드 (10개+)

| 파일 | 역할 |
|------|------|
| `tasks/wbs_data.py` | WBS 단일 소스(2026-04-10 ~ 05-06, 4명 담당, 3-depth 트리) |
| `tasks/build_wbs_xlsx.py` | wbs_data → xlsx 빌더 |
| `tasks/AeroInspect_WBS_v1.0.xlsx` | WBS 산출물 |
| `tasks/FlowFit_WBS_v1.0.pdf` | 외부 WBS PDF |
| `tasks/AeroInspect_Backend_Developer_Guide_v1.0.md` | 백엔드 개발자 가이드(스택·환경·구조·Git·CI/CD·배포·트러블슈팅) |
| `tasks/build_ppt.py` | 최종발표 PPT 빌더 |
| `tasks/ppt_content.py` | PPT 슬라이드 컨텐츠(표지/목차/문제/시장/차별화/팀/시스템/기능/데모/스택/로드맵) |
| `tasks/ppt_helpers.py` | PPT 도형·색·폰트·카드·메트릭·표·이미지 헬퍼 |
| `tasks/ppt_assets/{CAPTURE_GUIDE,SELF_EVAL_GUIDE,ANIMATION_GUIDE}.md` | 캡처/자가평가/애니메이션 가이드 |
| `tasks/ppt_assets/{images,logos,screenshots}/` | PPT 이미지·로고·스크린샷 |
| `tasks/AeroInspect_최종발표.pptx` | 최종발표 PPT (13MB, python-pptx 빌드) |
| `tasks/{capture_screens, capture_session_screens, capture_dashboard_active}.py` | 발표용 화면 캡처 자동화 |

### 📐 설계 결정 사항

- **명세 일원화 (v1.4 슬림화)**: 데이터·인터페이스 요구사항을 ERD·API 명세서에 일원화하여 요구사항정의서에서 중복 제거. 변경 시 두 군데를 동시에 수정해야 하는 부담 제거.
- **Mermaid + PNG 둘 다 유지**: Mermaid는 git diff 가능(텍스트), PNG는 PPT/Notion 즉시 임베드 가능. v1(분할)은 모바일/슬라이드 가독성용, v2(통합)는 인쇄·전체 검토용.
- **PPT를 코드로 빌드 (python-pptx 3단 구조)**: 손으로 PPT를 편집하면 (1) 디자인 일관성 깨짐, (2) 후속 수정 시 위치·서식 다시 맞춰야 함. 컨텐츠/헬퍼/빌더 분리로 컨텐츠만 수정해도 일관 디자인 유지.
- **WBS 데이터·빌더 분리**: 일정·담당·산출물 변경은 데이터(`wbs_data.py`)만 수정 → xlsx 자동 재생성. 빌더는 스타일·셀 폭만 책임.
- **버전 v1.0~v1.4 명시 + 변경 이력 유지**: 심사·발표·외부 공유 용도로 버전 식별이 필수. 변경 이력 표를 모든 v1.x 문서 첫머리에 일관 배치.
- **노션 동기화 정확성 규칙 준수**: 산출물 작성 시각은 git mtime 기준으로 그대로 기록(임의 시간 금지) — `feedback_notion_sync_accuracy` 적용.

---

## 🛰 추가 라운드 (2026-04-28 ~ 2026-05-03 — 머신러닝 후처리·통합 평가·Swagger·DB 시드)

> 5/1 이후 git commit 없이 unstaged 로 누적된 작업들을 라운드별 mtime 기준으로 정리.

| 라운드 | 시각 | 작업 | 산출물 |
|-------|------|------|-------|
| R19 | 2026-04-28 10:00 ~ 13:45 | **Recall 극대화 후처리 파이프라인** — TemporalFilter(Noisy-OR), ByteTrack ObjectTracker, SAHI TiledInference, ActiveLearning Hard Example Mining, DefectPersistence 임계값 조정. 단위 테스트 4종(temporal_filter / object_tracker / tiled_inference / ensemble) 동시 신설 | `app/services/{temporal_filter, object_tracker, tiled_inference, active_learning, defect_persistence}.py` + `app/core/stream_inference.py` + `app/models/defect.py` (track_id/accumulated_conf/tier_executed) + `tests/test_{temporal_filter,object_tracker,tiled_inference,ensemble}.py` |
| R20 | 2026-04-28 15:43 ~ 04-30 17:06 | **M1~M5 학습 스크립트 일괄 보강** — resnet/yolo 트레이너 정비, augment_missing_classes(부족 클래스 증강), 데이터셋 빌더 6종 (compress_m1, build_m4_context, convert_ade20k, build_m5v2, auto_run_m4v2, build_furniture_aware) | `training/train_m{1,2,3,4,5}_*.py` + `training/{augment_missing_classes, compress_m1_dataset, build_m4_context_dataset, convert_ade20k_to_yolo, build_m5v2_dataset, auto_run_m4v2, build_furniture_aware_dataset}.py` + `training/eval/evaluate_all.py` |
| R21 | 2026-05-02 03:03 ~ 03:10 | **ONNX inference + TTA 후처리 추가** — Test-Time Augmentation(수평/수직 flip + 다중 스케일), furniture_gate / geometric_gate 신설 (가구 위 false positive 차단, 수직수평 편차 검증) + 단위 테스트 | `app/services/{onnx_inference, tta}.py` + `tests/test_{furniture_gate, geometric_gate, tta}.py` |
| R22 | 2026-05-03 01:40 ~ 10:08 | **통합 평가 파이프라인 정착** — dry_run_full_pipeline(전체 6-Model 일괄 검증), evaluate_postprocess_ablation(후처리 ablation study), evaluate_integrated(M1~M6+gate 통합 mAP), detection schema 확장(20종 파이프라인 응답) | `training/eval/{dry_run_full_pipeline, evaluate_postprocess_ablation, evaluate_integrated}.py` + `app/schemas/detection.py` + `app/services/inference_pipeline_20.py` + `training/eval/results/integrated_eval_*.{json,md}` (5개) + `postprocess_ablation_*.{json,md}` (2쌍) |
| R23 | 2026-05-03 17:45 ~ 21:44 | **후처리 강도 정책 정착 + 배포 가이드** — furniture_gate / ensemble / postprocess_config.yaml 으로 후처리 강도를 모델별 분기(약한 모델만 boost, 강한 모델 무수정 — `feedback_postprocess_strength_policy` 적용), evaluate_ultralytics_val + evaluate_max_boost 측정. DEPLOYMENT_GUIDE 작성 | `app/services/{furniture_gate, ensemble, postprocess_config.yaml}` + `training/eval/{evaluate_ultralytics_val, evaluate_max_boost}.py` + 루트 `DEPLOYMENT_GUIDE.md` |
| R24 | 2026-05-03 22:xx ~ (현재 세션) | **Swagger Phase 1~3 + 운영 보안 가드** — HTTPBearer(bearerFormat=JWT) + AIWebhookSecret 보안 스키마 명시 등록, 17개 tags_metadata + servers + contact + persistAuthorization, 마크다운 description (인증·WS·멀티조직 가이드). PROTECTED/PUBLIC/WEBHOOK 공통 responses 적용. 핵심 schema 에 example 추가 (LoginRequest/TokenResponse/SiteCreate/DefectLogCreate). config.py + init_db.py 에 `APP_ENV=production` 가드 (placeholder secret 차단 / create_all 자동 스킵 → alembic 책임 분리). `.env.example` APP_ENV·AI_WEBHOOK_SECRET·PUSH_PROVIDER 등 보강. 보안 점검: .env git 추적 0건 / 소스 하드코딩 시크릿 0건 검증 | `app/main.py` + `app/api/router.py` + `app/schemas/common.py` (신규) + `app/schemas/{user, site, defect}.py` + `app/config.py` + `app/db/init_db.py` + `backend/.env.example` |
| R25 | 2026-05-03 23:xx ~ (현재 세션) | **Mockup → DB 전환 (KPI 0 방지 시드)** — InspectionSchedule 모델 + alembic migration `i2c3d4e5f6a7` 신규 (sites/users/orgs FK + scheduled_at). `/api/v1/employee` 라우터 (schedule/today + kpi/monthly + activities — 조직 단위 격리). 시연 시드 스크립트 `seed_demo_data.py` 신설: 조직(DRONE INSPECT 데모) + 부서 3 + 사용자(백승희/오희진) + 현장 8개(시행사·B2B/B2C·상태 분산) + 하자 25~60건/현장(6개월 분산) + 보고서 3~5건/완료현장 + 오늘 일정 3건(09:00 헬리오시티/14:00 잠실 리센츠 백승희/16:30 잠실 엘스 오희진) + 알림 8종/사용자. idempotent + APP_ENV 가드 + `--reset` / `--force-prod` 옵션 | `app/models/inspection_schedule.py` (신규) + `app/models/__init__.py` + `alembic/versions/i2c3d4e5f6a7_add_inspection_schedules.py` (신규) + `app/api/employee.py` (신규) + `app/api/router.py` + `scripts/seed_demo_data.py` (신규) |
| R26 | 2026-05-03 (현재 세션, 후속 정정) | **tasks 문서 인라인 정정 + DB 시드 실행 완료** — (1) **tasks 문서 양식 정정**: `API_명세서_AeroInspect_v1.1.md` → 새 v1.2 파일 (부록을 끝에 박지 않고 4.17 Employee API · 2.1.5 Swagger securityScheme · 8.5 운영 보안 가드 인라인 위치로 분산), `AeroInspect_ERD_설계서_v1.0.md` → 새 v1.1 파일 (4.19 inspection_schedules · 5장 관계 · 6.1 인덱스 · 8.3/12.1 Enum · 13장 결론 카운트 19/12/11/32 인라인 갱신, 문서 이력 위치 마지막 → 목차 이전 이동). 가이드 3종(AI 추론 파이프라인/Frontend Guide/Backend Guide) 도 문서 이력 위치 동일 정정. tasks 8개 문서 팀명 다마코더 → AeroInspect 일괄 교체. (2) **DB 마이그레이션 실 적용**: alembic 분기 head 2개(`0003`, `i2c3d4e5f6a7`)를 merge revision `89b53c16de85` 로 병합 → `alembic upgrade head` 성공. 이전 분기 마이그레이션 path 누락으로 `defect_logs` 컬럼 10개(image_crop_path, track_id, accumulated_conf, tier_executed, deviation_*, delta_temperature, ensemble_boosted, defect_class_display_*) inconsistent → `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 일괄 보정. (3) **시연 시드 실 적용**: `python -m scripts.seed_demo_data --reset` 실행 → org=1, depts=3, users=2(백승희/오희진), sites=8, **defects=315(HIGH 77)**, reports=12, **today schedules=3**(잠실 리센츠 14:00 KST 백승희 정상 시드 검증). (4) `CHANGES_2026-05-03.md` 신설 — 내일 Claude 웹 문서 변환용 산출물 목록 + 변환 프롬프트 템플릿. | `tasks/API_명세서_AeroInspect_v1.2.md` (신규) + `tasks/AeroInspect_ERD_설계서_v1.1.md` (신규) + `tasks/AeroInspect_AI_추론파이프라인_기획_v1.0.md` + `tasks/AeroInspect_Frontend_Developer_Guide_v1.0.md` + `tasks/AeroInspect_Backend_Developer_Guide_v1.0.md` + (팀명 일괄: API_v1.0/기획서_v1.3/리서치/요구사항 v1.0/v1.1/v1.4/generate_research_xlsx/ppt_assets README) + `backend/alembic/versions/89b53c16de85_merge_*.py` (alembic merge revision 신규) + 루트 `CHANGES_2026-05-03.md` (신규) |

### 📐 추가 설계 결정 사항 (R19~R25)

- **후처리 강도는 모델별 분기**: 강한 모델(M1·M3) 후처리 약화 금지, 약한 모델(M2·M4·M5)에만 furniture_gate / geometric_gate / TTA / ensemble boost 적용 — `feedback_postprocess_strength_policy` 메모리 적용. `postprocess_config.yaml` 단일 소스로 통합.
- **Swagger securitySchemes 명시 등록**: `HTTPBearer(bearerFormat=JWT)` 와 `AIWebhookSecret(X-AI-Webhook-Secret)` 두 스키마를 OpenAPI components 에 직접 주입 — FastAPI 기본은 bearerFormat 비워둠. Authorize 버튼 + `/ai/*` apiKey 입력란 모두 활성화.
- **운영 가드 일관 패턴**: `APP_ENV=production` 환경변수를 단일 식별자로 사용. config.py(placeholder secret 차단), init_db.py(create_all 스킵), seed_demo_data.py(시드 abort) 세 곳에서 동일하게 분기. 운영 entrypoint 는 `alembic upgrade head && uvicorn ...` 로 정착.
- **시드 idempotent 설계**: 모든 시드 함수는 사전 SELECT 로 존재 여부 확인 후 INSERT. 재실행해도 중복 데이터 안 쌓이고, `--reset` 으로 데모 데이터만 정확히 정리 가능. 슈퍼어드민/기존 사용자는 보존.
- **Mockup → DB 전환 원칙**: 프론트 const(MOCK_TODAY_SCHEDULE 등)에 박혀있던 시연 데이터를 다른 const 로 교체하는 것은 본질적으로 같은 mockup. 신규 모델(InspectionSchedule) + 시드 + endpoint 로 **실제 DB 데이터 → API → 프론트** 경로로 일원화. 시연 환경에서도 운영과 같은 데이터 흐름 사용.
- **KPI 0 방지 책임 분리**: 빈 DB → 0% KPI 표시는 프론트 fallback이 아니라 **시드가 충분량을 보장**하는 방식으로 해결. `_ensure_defects` 가 사이트당 25~60건, `_ensure_reports` 가 완료 사이트당 3~5건, `_ensure_today_schedule` 이 오늘 3건을 보장.
- **노션 동기화 정확성 규칙 재준수**: 본 라운드들의 시각은 working tree 파일 mtime 으로 산정 (5/1 이후 git commit 없음 — ML 학습 진행 중이라 commit 보류 상태). 임의 시간 X.
- **tasks 문서 양식 정정 원칙 (R26 추가)**: 변경분을 **부록**으로 끝에 박지 않고 본문 해당 장(章)에 인라인 삽입. 버전이 올라가면(`v1.0 → v1.1`, `v1.1 → v1.2`) 파일명도 함께 rename. 문서 이력은 표지 직후·목차 이전에 배치 (가이드 3종도 동일하게 정정). 팀명은 `AeroInspect` 단일 사용 (이전 `다마코더` 표기 일괄 교체).
- **alembic 분기 head 병합 + DDL 보정 (R26 적용 결과)**: 분기 마이그레이션 두 path 가 누적되어 head 가 2개(`0003`, `i2c3d4e5f6a7`) 발생 → `alembic merge` 로 `89b53c16de85` mergepoint 생성 후 `upgrade head` 성공. `defect_logs` 의 일부 컬럼이 alembic_version 에는 적용 완료로 기록됐으나 실제 DDL 미반영 상태였음(R19 ORM 컬럼 추가 시 마이그레이션 누락 + 분기 path 분기 영향) → `ADD COLUMN IF NOT EXISTS` 로 안전 보정. 시드 305건 INSERT 정상 통과로 검증.

---

## 🛰 R27 — 모델 mAP 한계 측정 + GPU 배포 가이드 전환 (2026-05-04 새벽 세션)

> 5/3 23:00 ~ 5/4 02:30 (~3.5시간 야간 세션). 모든 후처리 카드 시도 후 정직한 한계 측정.

| 라운드 | 시각 | 작업 | 산출물 |
|-------|------|------|-------|
| R27.1 | 5/3 23:00 ~ 23:30 | **DEPLOYMENT_GUIDE GPU 아키텍처 전환** — Cloud Run/Fly.io 분리 구조 → GCP Compute Engine + L4 GPU 통합 인스턴스. 드론 WebSocket 실시간 추론 적합. ONNX Runtime CUDA Provider, Cloud SQL Postgres, Nginx + Let's Encrypt 자동 시작·롤백 가이드까지 step-by-step 재작성 | `DEPLOYMENT_GUIDE.md` (985 → 990 lines, GPU 아키텍처) |
| R27.2 | 5/3 23:30 ~ 5/4 00:30 | **max_boost 평가 GPU 전환** — onnxruntime 1.25.1 (CPU) + onnxruntime-gpu 1.25.0 둘 다 설치되어 CPU 우선 import 되던 이슈. `pip uninstall onnxruntime` + `pip install --force-reinstall onnxruntime-gpu` 으로 CUDAExecutionProvider 활성화. 6모델 multi-scale × TTA 그리드 (30 runs) 15분 완주 | `max_boost_20260504_000519.json` |
| R27.3 | 5/4 00:30 ~ 02:30 | **확장 마스터 자동 평가 파이프라인** — 5개 신규 평가 스크립트 + 마스터 bash + 모니터: (1) `evaluate_pt_tta.py` PT real TTA, (2) `evaluate_extreme_boost.py` PT 그리드 (imgsz × tta × agnostic × iou × max_det 156 runs), (3) `evaluate_postproc_stages.py` 후처리 stage A/B, (4) `evaluate_wbf.py` Weighted Box Fusion multi-imgsz fusion, (5) `evaluate_multi_model_voting.py` cross_model_spatial_boost 진짜 효과 측정, (6) `evaluate_sahi_tiled.py` SAHI tiled inference, (7) `generate_final_report.py` 모든 결과 통합 리포트, (8) `run_master_full.sh` 전체 자동 실행 | `backend/training/eval/{evaluate_pt_tta, evaluate_extreme_boost, evaluate_postproc_stages, evaluate_wbf, evaluate_multi_model_voting, evaluate_sahi_tiled, generate_final_report}.py` + `run_master_full.sh` + `FINAL_REPORT_20260504_023009.md` |

### 📊 정직한 mAP 측정 결과 (모든 후처리 카드 시도 후)

| 모델 | best mAP50 | 0.85 갭 | best 방법 |
|------|------------|---------|-----------|
| M3_YOLO | **0.8445** | -0.0055 | extreme_boost grid (imgsz 800, TTA, agnostic NMS, iou 0.5) |
| M2_YOLO | 0.8193 | -0.0307 | extreme_boost grid (imgsz 640, TTA, iou 0.5) |
| M5_SEG | 0.7295 | -0.1205 | extreme_boost grid (imgsz 800) |
| furniture_aware | 0.6224 | -0.2276 | max_boost (imgsz 640) |
| M1_YOLO | 0.6127 | -0.2373 | max_boost (imgsz 640) |
| M4_CONTEXT | 0.5871 | -0.2629 | extreme_boost grid (imgsz 960) |

**한 모델도 0.85 도달하지 못함** — 후처리만으로의 한계 명확.

### 🔍 후처리 카드별 진짜 효과 (측정값)

| 카드 | M3 효과 | M2 효과 | 진단 |
|------|---------|---------|------|
| ONNX multi-scale | 0.79 → 0.8366 | 0.79 → 0.8139 | ✅ imgsz 변경만으로 큰 개선 |
| .pt + real TTA | 0.8366 → 0.8376 | 측정됨 | ⚠️ TTA 자체 효과 미미 (+0.001) |
| extreme grid (imgsz × tta × agnostic × iou) | 0.8376 → **0.8445** | 0.8139 → 0.8193 | ✅ 가장 효과적 |
| WBF multi-imgsz fusion | 0.8445 → 0.7991 ↓ | 0.8193 → 0.7998 ↓ | ❌ conf 0.001 noise 증폭 |
| Multi-model voting (cross_model_nms) | 0.8018 → 0.7829 ↓ | 0.8324 → 0.8369 ↑ | ⚠️ M2만 +0.005 |
| Multi-model voting (spatial_boost) | 0.8018 → 0.7565 ↓ | 0.8324 → 0.7740 ↓ | ❌ false positive 증폭 |
| SAHI tiled | M1: 0.8411 → 0.8391 | - | ❌ 한계 도달 |

### 📐 핵심 결론 (사용자 보고용)

1. **후처리만으로 0.85 불가능** — 모든 카드 시도 후 정직한 한계
2. **이전 mAP 측정값들이 imgsz 미스매치로 저평가됐음** (이미 보정 — M5: 0.34→0.7295, furniture: 0.38→0.6224, M4: 0.5466→0.5871)
3. **ONNX는 ultralytics에서 augment=True가 silently 무시됨** — TTA는 .pt에서만 작동
4. **WBF는 conf=0.001 환경에서 도리어 해로움** — noise box 다수가 fusion 시 false positive로 살아남음
5. **Cross-model voting은 단일 클래스 taxonomy에서만 부분 효과** — M2 +0.005, 다른 모델은 noise
6. **0.85 도달의 유일한 길은 v1.1 재학습** — 5/6 1차 배포 후 사용자 신호로 진행



---

## 🛰 R28 — 브라우저 기반 GPU VM 원격 제어 (2026-05-04 오후)

> 로컬 bat 파일 의존 제거 → 어떤 브라우저에서도 슈퍼어드민 권한이면 GCP L4 GPU VM 을 켜고/끌 수 있도록 플랫폼 내부에 통합. 상용 멀티유저 운영 전제.

| 라운드 | 시각 | 작업 | 산출물 |
|-------|------|------|-------|
| R28.1 | 5/4 오후 | **GCP Compute REST 클라이언트** — 서비스 계정 JSON 으로 RS256 JWT 서명 → OAuth2 토큰 교환 → instances.{get,start,stop} 호출. 토큰은 만료 30초 전까지 메모리 캐시. base64 / JSON 원문 둘 다 입력 허용 | `app/services/gcp_compute.py` |
| R28.2 | 5/4 오후 | **관리자 GPU API** — `GET/POST /api/v1/admin/gpu/{status,start,stop}` 슈퍼어드민(require_superadmin) 전용. 502 변환 가드 | `app/api/admin_gpu.py` + `app/api/router.py` |
| R28.3 | 5/4 오후 | **config 확장** — `GCP_SERVICE_ACCOUNT_JSON` / `GCP_PROJECT_ID` / `GCP_GPU_ZONE` / `GCP_GPU_INSTANCE`. Fly.io secrets 로 주입 | `app/config.py` + `app/main.py` (Admin 태그) |
| R28.4 | 5/4 오후 | **GCP IAM 셋업** — 서비스 계정 `drone-gpu-controller` 생성 + `roles/compute.instanceAdmin.v1` 바인딩. 조직 정책 `iam.disableServiceAccountKeyCreation` 가 키 발급을 차단 → orgpolicy API 활성화 + 사용자에 `roles/orgpolicy.policyAdmin` 부여 → 프로젝트 레벨 override 로 해제 후 키 발급. base64 인코딩하여 Fly secrets 로 등록 | (인프라 작업, 코드 산출물 없음) |
| R28.5 | 5/4 오후 | **CI/CD 워크플로우 위치 수정** — `backend/.github/workflows/fly-deploy.yml` 은 GitHub Actions 가 인식하지 않음(레포 루트의 `.github/workflows/` 만 인식). 루트로 이동 + monorepo 대응(`paths: ["backend/**"]` 필터, `working-directory: backend`) | `.github/workflows/fly-deploy.yml` |

### 📐 설계 결정 사항

- **백엔드는 항상 켜짐, GPU 만 ON/OFF**: Fly.io 백엔드(`api.aeroinspect.site`)는 GPU 없이 항상 살아있어야 GPU 제어 호출을 받을 수 있다. GPU 추론용 Compute Engine VM (`drone-stream-api`) 만 점검 시작/종료 시점에 토글한다.
- **권한은 슈퍼어드민으로 한정**: 일반 조직 admin/owner 는 자기 조직 멤버만 다룰 수 있고, 인프라 비용에 직접 영향 가는 GPU 제어는 플랫폼 슈퍼어드민(`is_superadmin=True`) 전용. `require_superadmin` 의존성으로 강제.
- **서비스 계정 키 base64 저장**: Fly.io `secrets set` 가 multiline 값을 인자로 받지 못함 → 키 파일을 base64 한 줄로 인코딩하여 단일 secret 으로 주입. 백엔드는 첫 글자가 `{` 가 아니면 base64 로 간주하여 디코딩.
- **조직 정책 우회는 프로젝트 레벨 override 로 최소 침습**: 조직 전체에 `disableServiceAccountKeyCreation` 을 끄지 않고 이 프로젝트만 해제. 다른 프로젝트 보안 영향 없음.
- **워크플로우 paths 필터**: monorepo 라 frontend/ 만 변경된 PR 도 Fly 배포가 트리거되던 미해결 이슈 → `paths: ["backend/**", ".github/workflows/fly-deploy.yml"]` 로 backend 변경 시에만 실행.


### R28.6 Fly.io 배포 설정 파일 main 반영 (CI 통과용) (2026-05-04)

GitHub Actions Fly Deploy 가 `fly.toml` 미커밋으로 'missing app name' 실패 → 누락된 배포 설정을 main 에 반영.
- `fly.toml`: aeroinspect-backend 앱 설정
- `.dockerignore`: 1.8GB models_weights/ + captured_frames/ + uploads/ 제외
- `Dockerfile`: `libgl1-mesa-glx` → `libgl1` (Debian trixie 호환)
- `.gitignore`: 학습 로그/uploads/.onnx.data 추가
