# AeroInspect Backend

드론 기반 하자 점검 플랫폼 FastAPI 서버. 이 문서는 **3-모델 추론 파이프라인** (YOLOv8 × 2 + ResNet50) 관련 설정·엔드포인트·마이그레이션 절차를 다룹니다. 인증/사이트/보고서 등 다른 모듈은 [app/api/](app/api/) 코드 참고.

## 실행

```bash
cd backend
python -m venv venv && source venv/Scripts/activate  # Windows bash
pip install -r requirements.txt
cp .env.example .env          # DB URL, 가중치 경로 수정
# weights 3개 파일을 models_weights/ 폴더에 배치
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: `http://localhost:8000/docs`

## 3-모델 추론 파이프라인

| 모델 | 파일명 | 클래스 | 용도 | 학습 설정 |
|------|-------|--------|------|-----------|
| YOLOv8s thermal | `yolov8s_crack_moisture_best.pt` | Crack, Moisture | 열화상 균열+습기 | epochs=100, batch=16, imgsz=640, lr0=0.001, patience=20 |
| YOLOv8s delam | `yolov8s_delamination_best.pt` | delamination | 외벽 박리 | 동일 |
| ResNet50 wallpaper | `resnet50_wallpaper_best.pt` | 19 클래스 | 벽지 하자 분류 | Transfer Learning, val_acc≈0.54 |

### ⚠️ 벽지 19 클래스와 `good` 클래스의 정체

ResNet50 체크포인트의 `class_names` (알파벳 순):

```
Baseboard, Crying, Damage, Defective_Joint, Exploded,
Furniture, Gypsum, Kink, Many_niches, Mold,
Molding, Piece, Plane, Pollution, Rust,
Spot, W.F_D.F, Wrong_punch, good
```

**`good`은 "정상" 클래스가 아님.** 데이터셋 폴더명이 실수로 `good`으로 지어졌을 뿐, 실제 내용은 **"터짐(Burst/Tear)" 하자 이미지**입니다. 가중치에 baked-in 되어 있어 내부명은 유지하되, [app/services/defect_taxonomy.py](app/services/defect_taxonomy.py)의 `CLASS_DISPLAY_MAP`이 프론트에는 반드시 `Burst` / `터짐`으로 표시해주고 severity는 `MED`로 격상됩니다.

절대 `good`을 "정상=하자 없음"으로 필터링하지 마세요.

### severity 규칙 ([inference_pipeline.py](app/services/inference_pipeline.py))

- YOLO thermal/delam 탐지 있음 → `HIGH`
- 벽지 `is_confident` + top1 ∈ {Mold, Damage, Exploded, Defective_Joint, good} → `MED`
- 벽지 `is_confident` + 그 외 → `LOW`
- 그 외 (신뢰도 부족) → `null` (판단 보류)

## 엔드포인트

### REST

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 카메라 + 3-모델 + 스트림 워커 상태 |
| `POST` | `/api/v1/detect` | multipart 이미지 1장 → `DetectionResult` (R-v1.1.17: grade 필드 포함) |
| `POST` | `/api/v1/detect/batch` | 최대 10장 → `List[DetectionResult]` |
| `GET` | `/api/v1/defects/recent?limit=50&severity=HIGH` | 최신순 하자 로그 N건 |
| `GET` | `/api/v1/defects` | 하자 로그 목록 (필터+페이지네이션) |
| `POST` | `/api/v1/defects` | 하자 로그 저장 + WS broadcast |
| `PATCH` | `/api/v1/defects/{id}/review` | 검수 승인/거부/플래그 (R-v1.1.08 감사 로그 기록) |
| `GET` | `/api/v1/defects/{id}/audit-trail` | 단일 하자 검수 이력 조회 |
| `GET` | `/api/v1/audit-logs` | 전체 감사 로그 (admin/owner/superadmin 전용) |
| `GET` | `/api/v1/employee/schedule/today` | 오늘 점검 일정 (R-v1.1.05) |
| `GET` | `/api/v1/employee/kpi/monthly` | 월간 KPI |
| `GET` | `/api/v1/employee/activities` | 최근 활동 |
| `GET` | `/api/v1/stream/stats` | 추론 워커 실시간 메트릭 |
| `GET` | `/api/v1/coverage/{site_id}` | 현장별 점검 커버리지 |
| `POST` | `/api/v1/auth/refresh` | refresh token rotation (R-v1.1.17) — 응답에 새 access_token + 새 refresh_token |

### WebSocket

| 경로 | 방향 | 메시지 |
|------|------|--------|
| `/api/v1/ws?channel=defects` | 구독 전용 | `{"type":"defect.new", "data":{...}}` |
| `/api/v1/ws?channel=stream` | 구독 전용 | `{"type":"detection", ...}` (아래와 같음) |
| `/api/v1/ws/stream` | 양방향 | 바이너리 JPEG 송신 → 추론 결과 수신 |

`/ws/stream` 프로토콜:

- **클라이언트 → 서버**:
  - `bytes`: 드론 캡처 JPEG 프레임 (raw bytes)
  - `text`: `{"type":"ping"}` (하트비트)
- **서버 → 클라이언트**:
  - `{"type":"pong"}`
  - `{"type":"detection","timestamp":1713580800.12,"frame_id":42,"result":{DetectionResult}}`
  - `{"type":"error","message":"..."}`

`/ws/stream`의 탐지 결과는 `/ws?channel=defects` 구독자에게도 자동으로 레거시 `defect.new` 포맷으로 동시 전송됩니다.

### `DetectionResult` 스키마 예시

```json
{
  "yolo_thermal": [
    {"class":"Crack","class_display_en":"Crack","class_display_ko":"균열",
     "conf":0.87,"bbox_xyxy":[x1,y1,x2,y2]}
  ],
  "yolo_delam": [],
  "wallpaper_cls": {
    "top1_class":"good","top1_class_display_en":"Burst","top1_class_display_ko":"터짐",
    "top1_conf":0.62,"is_confident":true,
    "top3":[
      {"class":"good","class_display_en":"Burst","class_display_ko":"터짐","conf":0.62},
      {"class":"Mold","class_display_en":"Mold","class_display_ko":"곰팡이","conf":0.18},
      {"class":"Spot","class_display_en":"Spot","class_display_ko":"반점","conf":0.08}
    ]
  },
  "severity": "HIGH",
  "has_defect": true,
  "defect_count": 2,
  "image_shape": {"width": 640, "height": 480}
}
```

**API 응답은 `bbox_xyxy` (픽셀)**, DB 저장은 기존 `bbox_x/y/w/h` (xywhn 정규화). 변환은 [defect_taxonomy.py::xyxy_to_xywhn](app/services/defect_taxonomy.py) 헬퍼 사용.

## WebSocket 성능 최적화

드론 IRC-256CA는 15~30 fps, T4 GPU에서 3-모델 순차 추론은 80~150 ms/frame → 전부 처리 불가. [stream_inference.py](app/core/stream_inference.py)가 다음 패턴으로 처리:

- **드롭 큐**: `asyncio.Queue(maxsize=1)` — 워커 바쁘면 새 프레임을 그냥 드롭
- **프레임 스킵**: `FRAME_SKIP=3` — N프레임 중 1프레임만 추론 (env로 조절)
- **스레드 풀**: YOLO/ResNet 추론은 `asyncio.to_thread()`로 감싸 이벤트 루프 블로킹 방지
- **JPEG 디코딩도 스레드**: `asyncio.to_thread(cv2.imdecode, ...)`

## React 연동 예제

```jsx
// /ws/stream — 드론 프레임 업링크 + 탐지 결과 다운링크
import { useEffect, useRef } from "react";

export function DroneStreamView({ canvasRef }) {
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/api/v1/ws/stream");
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      const msg = JSON.parse(ev.data);
      if (msg.type === "detection") drawDetections(msg.result, canvasRef.current);
    };

    // ping 하트비트
    const pingId = setInterval(() => {
      ws.readyState === 1 && ws.send(JSON.stringify({ type: "ping" }));
    }, 25_000);

    return () => { clearInterval(pingId); ws.close(); };
  }, [canvasRef]);

  // 드론에서 프레임 업로드 예시 (getUserMedia/캡쳐카드에서 온 JPEG bytes)
  const sendFrame = (jpegBytes) => {
    wsRef.current?.readyState === 1 && wsRef.current.send(jpegBytes);
  };
  return null;
}

function drawDetections(result, canvas) {
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  [...result.yolo_thermal, ...result.yolo_delam].forEach((d) => {
    const [x1, y1, x2, y2] = d.bbox_xyxy;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillText(`${d.class_display_ko} ${(d.conf * 100).toFixed(0)}%`, x1, y1 - 4);
  });
}
```

## DB 마이그레이션 절차 (첫 배포)

> ⚠️ **운영 배포 전 정리 필요 (TODO)**
>
> 현재 [`init_db.py`](app/db/init_db.py)가 `Base.metadata.create_all`로 테이블을 자동 생성하고 있어, alembic 마이그레이션 시스템과 **이중으로 굴러가는 상태**입니다. 이대로 두면:
> - 모델에 컬럼 추가해도 기존 테이블엔 반영 안 됨 (`create_all`은 신규 테이블만 만듦)
> - 팀원/서버마다 DB 스키마가 달라질 수 있음
> - 이미 `versions/`에 9개 마이그레이션 파일이 있는데, 적용 이력(`alembic_version`)이 추적 안 됨
>
> **출시 전 작업 항목:**
> 1. `init_db.py` 에서 `create_all` 호출 제거 (시드 데이터 삽입만 남김)
> 2. 운영 DB에 `alembic stamp head` 1회 실행 (현재 스키마를 최신 리비전으로 도장만 찍기)
> 3. 이후 모든 모델 변경은 `alembic revision --autogenerate -m "..."` → `alembic upgrade head` 로만 진행
>
> **현재까지는** 팀이 로컬에서 `init_db` 방식으로 잘 쓰고 있으니 그대로 유지. 출시 직전 운영 DB 백업 후 한 번에 정리할 것.

Alembic 리비전 [0002_defect_class_display.py](alembic/versions/0002_defect_class_display.py)가 `defect_logs`에 4개 컬럼 추가 + 기존 `area/category_code/defect_type`을 `NULLABLE`로 완화합니다. 기존 스키마와 drift 나지 않도록 **baseline은 `stamp` 방식으로 처리**합니다.

### 신규 DB (비어 있음)
```bash
# 1) app 기동 시 init_db가 현재 모델로 테이블 생성
uvicorn app.main:app  # 1회 기동 후 종료
# 2) alembic을 현재 상태로 stamp
alembic stamp 0002_defect_class_display
```

### 기존 DB (풀 전 스키마로 이미 생성돼 있음)
```bash
# 1) 기존 스키마를 "초기 상태"로 stamp (마이그레이션 실행은 하지 않음)
alembic stamp base
# 2) 0002 마이그레이션 적용 (컬럼 4개 추가 + NULLABLE 완화)
alembic upgrade head
```

### 로컬 개발 (매번 테이블 드롭·재생성해도 OK)
```bash
# init_db가 Base.metadata.create_all로 신규 스키마 반영
# alembic 건너뛰고 그냥 서버 기동
```

## 환경변수 (.env)

핵심 3-모델 관련:

```
AEROINSPECT_WEIGHTS_DIR=./models_weights
YOLO_THERMAL_WEIGHTS=yolov8s_crack_moisture_best.pt
YOLO_DELAM_WEIGHTS=yolov8s_delamination_best.pt
WALLPAPER_WEIGHTS=resnet50_wallpaper_best.pt
YOLO_CONF_THRESHOLD=0.25
WALLPAPER_CONF_THRESHOLD=0.35
WALLPAPER_MARGIN_THRESHOLD=0.15
FRAME_SKIP=3
DEVICE=auto
```

전체 목록은 [.env.example](.env.example) 참조.

---

## 운영 모니터링 & 관측성

### 상태 조회 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 카메라 / 3-모델 / 스트림 워커 / LiDAR / telemetry 캐시 전체 상태 |
| `GET` | `/api/v1/stream/stats` | 추론 워커 실시간 메트릭 (submitted/processed/dropped/queue_size) + LiDAR 연결 상태 |
| `GET` | `/api/v1/coverage/{site_id}` | 현장별 점검 커버리지 (텔레메트리 convex hull → covered/supplied/ratio) |

`/api/v1/stream/stats` 응답 예시:
```json
{
  "worker": {"running": true, "submitted": 18420, "processed": 6123, "dropped": 12, "queue_size": 0, "frame_skip": 3},
  "telemetry_cache": {"ready": true, "age_sec": 0.12},
  "lidar": {"connected": true, "distance_m": 2.43}
}
```

### 구조화 로깅 (structlog)

- `LOG_JSON=false` (기본): 개발용 컬러 콘솔
- `LOG_JSON=true`: 운영용 JSON 한 줄 로그 → Grafana Loki / Datadog / CloudWatch 바로 적재
- 모든 로그에 `request_id`, `method`, `path` 자동 바인딩 ([app/core/middleware.py](app/core/middleware.py))
- 표준 이벤트: `http.request` (status, duration_ms) / `http.request.failed` (traceback)
- 클라이언트 → `X-Request-ID` 헤더 전달 시 그대로 재사용, 미전달 시 서버가 16자리 hex 자동 발급

### 벽지 이중 게이트 임계값 튜닝

운영 로그(JSONL) 축적 후 [scripts/sweep_wallpaper_thresholds.py](scripts/sweep_wallpaper_thresholds.py)로 `WALLPAPER_CONF_THRESHOLD` × `WALLPAPER_MARGIN_THRESHOLD` 격자 탐색:

```bash
python scripts/sweep_wallpaper_thresholds.py --input ops_logs.jsonl --out sweep.csv
```

입력 한 줄: `{"top1_conf": 0.62, "top2_conf": 0.41, "label": "defect"}` (label은 사람이 태깅한 GT).

---

## 20종 하자 검출 파이프라인 (신규)

> `USE_20DEFECT_PIPELINE=true` 설정 시 활성화. 기존 3-모델 파이프라인과 병존.

### 아키텍처: 6-Model + Geometric (ONNX Runtime)

기존 PyTorch 직접 추론에서 **전 모델 ONNX Runtime** 추론으로 전환. 프레임워크 종속성 제거, 추론 속도 ~20% 향상.

| 모델 | 아키텍처 | 커버 하자 | 입력 |
|------|---------|----------|------|
| M1 | YOLOv8m → ResNet50 (2-Stage) | A-02 구조균열, A-03 마감균열, B-03 코킹, B-04 방수 | RGB |
| M2 | YOLOv8m → ResNet50 (2-Stage) | C-01~C-05 (도배/도색/스크래치/걸레받이) | RGB |
| M3 | YOLOv8m → ResNet50 (2-Stage) | D-03 바닥오염, D-04 줄눈, E-01 유리, E-02 문틀 | RGB |
| M4 | U-Net (EfficientNet-B3) | B-01 창호단열, B-02 벽체단열, B-05 기밀, D-01 바닥난방 | Thermal+RGB |
| M5+G1 | YOLOv8m-seg + Hough/RANSAC | A-01 수직수평, A-04 직각도 | RGB+IMU |
| M6 | PatchCore (Anomalib) | 전체 앙상블 보완 | RGB |

### 2-Stage 구조

YOLO가 하자 영역을 검출(Stage 1) → ROI 크롭 → ResNet50이 하자 유형을 정밀 분류(Stage 2).

### 계층적 실행 (실시간 스트리밍)

```
Tier 1 (매 3프레임): M1 + M2      → ~50ms  (HIGH severity 즉시 검출)
Tier 2 (매 6프레임): + M3 + M5+G1  → ~55ms  (MED/LOW + 기하학)
Tier 3 (매 9프레임): + M4 + M6     → ~70ms  (열화상 + 앙상블)
```

### 새 환경변수 (.env)

```env
USE_20DEFECT_PIPELINE=true   # false면 기존 3-모델 사용
M1_YOLO_ONNX=m1_yolo_structural.onnx
M1_RESNET_ONNX=m1_resnet_crack_classifier.onnx
M1_CONF_THRESHOLD=0.15
# ... (전체 목록은 app/config.py 참조)
```

### 새 DB 컬럼 (마이그레이션 0003)

```bash
alembic upgrade head  # deviation_degrees, deviation_mm_per_m, delta_temperature, ensemble_boosted 추가
```

### `DetectionResult20` 스키마 예시

```json
{
  "detections": [
    {"class":"crack_structural","code":"A-02","class_display_ko":"균열 (구조 균열)",
     "conf":0.82,"bbox_xyxy":[120,80,340,210],"severity":"HIGH",
     "defect_source":"yolo_structural"}
  ],
  "insulation": [
    {"class":"wall_insulation_gap","code":"B-02","delta_temperature":4.2,
     "max_temperature":28.5,"min_temperature":18.2,"severity":"HIGH"}
  ],
  "alignment": [
    {"class":"frame_squareness_defect","code":"A-04",
     "deviation_degrees":0.35,"deviation_mm_per_m":6.1,"severity":"MED"}
  ],
  "anomaly_score": 0.72,
  "has_defect": true,
  "defect_count": 3,
  "image_shape": {"width":640,"height":480},
  "tier_executed": 3
}
```

### ML 학습 가이드

데이터 준비 → 모델 학습 → ONNX 변환 → 배포 전체 과정:

**[training/README.md](training/README.md)** 참조.

---

## 알려진 제약

- **벽지 분류 val_acc ≈ 54%**: 19-way 분류라 정확도 낮음. 이중 게이트로 필터링:
  - `top1_conf >= WALLPAPER_CONF_THRESHOLD` (기본 0.35, top1 절대 신뢰도)
  - AND `top1_conf - top2_conf >= WALLPAPER_MARGIN_THRESHOLD` (기본 0.15, top2와의 분리도 — 근소차 예측 차단)
  - 두 조건 모두 만족해야 `is_confident=true`. 그렇지 않으면 하자 판정 보류 (severity null).
- **`good` 클래스는 터짐**: 필터링 시 절대 "정상"으로 취급 금지. 위 경고 박스 참조.
- **단일 워커 프로세스 전제**: `stream_inference_worker`는 프로세스 내 싱글톤. gunicorn multi-worker로 띄우면 워커마다 큐가 생겨 FRAME_SKIP 효과가 배수. uvicorn 단일 워커 또는 Redis pub/sub 기반 리팩터링 필요.
- **MAVLink/LiDAR 좌표 연동은 아직**: `drone_coordinates`는 당분간 NULL. 추후 TF 연동 시 기존 `lidar_x/y/z` 컬럼에 채움.

## 운영 에러 모니터링 (Sentry)

운영 환경에서만 활성화. 로컬 개발은 `SENTRY_DSN` 비워두면 자동 no-op.

1. **DSN 발급**: [sentry.io](https://sentry.io) → 새 프로젝트 (Platform: `FastAPI / Python`) → Settings → Client Keys (DSN) 복사
2. **Fly.io secrets 등록** (운영 배포 직전):
   ```bash
   flyctl secrets set \
     SENTRY_DSN="https://xxxxxxxx@oXXXXX.ingest.sentry.io/YYYYY" \
     SENTRY_ENVIRONMENT="production" \
     -a aeroinspect-backend
   ```
3. **자동 적용 항목**: FastAPI/Starlette/SQLAlchemy/Asyncio 미처리 예외, `RequestIDMiddleware`의 `request_id` 자동 태깅, 민감 키(`password/token/secret/...`) `[REDACTED]` 처리, `send_default_pii=False` (이메일/IP 미수집).
4. **검증**: 배포 후 `curl https://aeroinspect-backend.fly.dev/health` 정상 → 임시 테스트 라우트에서 `raise RuntimeError("sentry test")` → Sentry Issues 탭에 이벤트 도착 확인.

## 동작 확인 체크리스트

서버 기동 후 다음을 순서대로 확인:

- [ ] `uvicorn app.main:app --reload --port 8000` 기동 로그에 `[Pipeline] 3-모델 로드 완료` 출력
- [ ] `curl http://localhost:8000/health` → `"status":"ok"`, `models_loaded` 3개 전부 `true`, `wallpaper_classes_count: 19`, `stream_worker_running: true`
- [ ] Swagger `/docs`에서 `POST /api/v1/detect`, `POST /api/v1/detect/batch`, `WebSocket /api/v1/ws/stream`, `GET /api/v1/defects/recent` 노출 확인
- [ ] `POST /api/v1/detect`에 `good` 클래스 이미지 업로드 → 응답 `wallpaper_cls.top1_class_display_ko == "터짐"` 확인
- [ ] `cd backend && pytest tests/test_inference_pipeline.py -v` 통과 (가중치 없어도 taxonomy/xyxy_to_xywhn/health 테스트는 전부 통과)
- [ ] React 대시보드에서 `ws://.../api/v1/ws?channel=defects` 구독 후 `/ws/stream`으로 더미 JPEG 전송 → `defect.new` 이벤트 수신
- [ ] Alembic: 위 "DB 마이그레이션 절차"대로 stamp → upgrade → `\d defect_logs`로 4개 신규 컬럼 확인
