# Backend Task.md

## 프로젝트 개요
- **목적**: AeroInspect AI 백엔드 — 데이터 엔진, 17개 API 라우터, AI 추론 파이프라인(20종 ONNX), 실시간 통신
- **주요 스택**: FastAPI (비동기 Python), PostgreSQL + SQLAlchemy + Alembic, ONNX Runtime + YOLOv8 + ResNet50, WebSocket + Redis Pub/Sub, Prometheus, structlog
- **팀원**: @youminsu0523 (API/ML/서비스), @Hijin554 (추론/텔레메트리/인프라/테스트)

---

## 작업 목록 — @youminsu0523 (branch: MS)

### 백엔드 기초 (260414)
- [x] v2.0_260414 — FastAPI 앱 초기 구조
  - `main.py`: `lifespan` 핸들러 (DB init, RGB/Thermal 카메라 open, YOLOv8 모델 load), CORS, `/api/v1` 라우터, `/health` 헬스체크
  - `requirements.txt`: FastAPI, SQLAlchemy(asyncio), OpenCV, PyTorch, Ultralytics(YOLOv8), anomalib, pymavlink, anthropic, google-generativeai 등

### 인증 시스템 (260416~260420)
- [x] v2.1_260416 — 회원가입 + 사용자 모델 **31개 파일 중 백엔드 부분**
  - `api/auth.py`: POST /auth/signup (개인/사업자 공용, User+BusinessProfile+UserTermAgreement 동시 생성), GET /auth/check-email, GET /auth/check-username
  - `models/user.py`: UUID PK, `account_type`(personal/business), email(unique), username(unique), password_hash(bcrypt), name, phone, oauth_provider, oauth_id
  - `models/business_profile.py`: user_id(FK PK), biz_number(10자리 unique), ceo_name, verified_at
  - `models/term.py`: 약관 마스터 (code/title/is_required/version/effective_from), 초기 시드 3종(service/privacy/marketing)
  - `models/user_term_agreement.py`: user_id FK, term_id FK, version 스냅샷
  - `schemas/user.py`: UserSignupRequest(BusinessInfoInput/TermsAgreementInput 중첩), UserResponse(password_hash 미포함), AvailabilityResponse
  - `core/security.py`: bcrypt 해싱 (hash_password/verify_password/needs_rehash, passlib CryptContext)
- [x] v2.3_260417 — 로그인 + OAuth 3종 **18개 파일 중 백엔드 부분**
  - `api/auth.py` 추가: POST /auth/login (username+password→verify→JWT), GET /auth/me
  - `api/oauth.py` (230줄): Google/Kakao/Naver 3종 소셜 로그인 — authorization code→access_token→userinfo→JWT, `_find_or_create_oauth_user()` 3단계(oauth_id 조회→이메일 매칭→신규 생성)
  - `core/jwt.py`: create_access_token(HS256)/decode_access_token
  - `config.py`: JWT_SECRET, JWT_EXPIRE_MINUTES, Google/Kakao/Naver client_id/secret 6개
  - `dependencies.py`: get_current_user Bearer 토큰 검증
  - `.env.example`: OAuth 환경변수
- [x] v2.4_260420 — OAuth race condition 대응
  - `api/oauth.py`: 이메일 대소문자 무시(`func.lower`), IntegrityError 시 rollback→재조회→409

### 현장 관리 (260418)
- [x] v2.5_260418 — 현장 CRUD API **22개 파일 중 백엔드 부분**
  - `models/site.py`: UUID PK, `seq`(자동 순번), name, `inspection_type` 6종 Enum(사전/입주/정기/하자/특별/기타), address, `building_type` 7종 Enum(아파트/오피스텔/상가 등), total_area, building_count, unit_count, `client_type` Enum(B2B/B2C), client_name/contact, contract_start/end, `status` 4종(active/pending/completed/cancelled), `assigned_members`(JSONB), `recordings`(JSONB), inspection_count, last_inspection_date 등 **20+ 컬럼**
  - `schemas/site.py`: SiteCreate/SiteUpdate/SiteResponse/SiteListResponse + AssignedMember/Recording 중첩
  - `api/sites.py` 5개 엔드포인트: GET(status/building_type/client_type 필터+검색+페이지네이션), GET/:id, POST, PATCH(JSONB 변환), DELETE

### 채팅 + 알림 + 조직 관리 (260420~260424)
- [x] v2.6_260420 — 풀스택 채팅/알림/조직 **48개 파일 중 백엔드 부분** (+4,313줄)
  - **DB 모델 5개 신규**: `Conversation`(dm/group/channel Enum, created_by FK), `ConversationMember`(M:N, last_read_at), `Message`(conversation_id FK, sender_id FK, text, 복합인덱스), `Notification`(10종 카테고리 Enum, JSONB metadata, is_read, 읽음인덱스), `Organization`(biz_number 매칭)+`OrganizationMember`(role owner/admin/member, status active/invited/deactivated, department, position)
  - `api/chat.py` 6개: 대화방 목록/생성/메시지 목록/전송(+WS 브로드캐스트)/읽음/미읽음
  - `api/notifications.py` 5개: 목록(카테고리/읽음 필터+페이지네이션)/미읽음수/단건읽음/전체읽음/삭제
  - `api/organization.py` 6개: 내 조직/멤버목록/생성(biz_number 중복체크)/초대(admin/owner)/수정/삭제(owner 보호)
  - `services/email_service.py`, `services/notification_service.py`
- [x] v2.10_260424 — 채팅/조직 고도화
  - `api/chat.py`: DM 중복 방지(aliased ConversationMember 자기조인), DELETE /conversations/:id/leave (참여자 0→대화방 삭제)
  - `api/organization.py`: POST /members/assign(슈퍼어드민 cross-org), GET /admin/all-orgs(LEFT JOIN+GROUP BY 멤버수), GET /admin/orgs/:id/departments
  - `schemas/organization.py`: invite_code 필드

### AI/ML 파이프라인 (260422)
- [x] v3.0_260422 — 20종 ONNX 추론 **41개 파일** (+6,392줄)
  - **ONNX 추론 엔진** (`onnx_inference.py` 393줄): ONNXYoloDetector(letterbox+NMS+CUDA/CPU EP 자동전환), ONNXResNetClassifier(ImageNet정규화+softmax top3), ONNXUNetSegmenter(온도맵→3ch→멀티클래스 마스크), ONNXPatchCoreDetector(anomalib export 호환), `crop_roi()`(패딩 크롭), `_nms_numpy()`, `_create_session()`
  - **20종 파이프라인** (`inference_pipeline_20.py` 351줄): M1(YOLO 구조/방수→ResNet 균열 2-Stage crack→crack_structural/crack_finishing), M2(YOLO 마감/표면→ResNet 표면 wallpaper_seam/bubble/paint_stain/scratch), M3(YOLO 바닥/창호→ResNet 유형), M4(U-Net 열화상 insulation_detector 위임), M5+G1(YOLO-seg + alignment_detector 위임), M6(PatchCore 앙상블 폴백). Tier 기반 계층 실행(Tier1=M1+M2, Tier2=+M3+M5, Tier3=+M4+M6), `detect_async()`(asyncio.to_thread)
  - **앙상블** (`ensemble.py` 109줄): `cross_model_nms()`(같은 class IoU NMS, 다른 class 겹침 보존=복합 하자), `ensemble_with_patchcore()`(이상 판정 영역 저신뢰 검출→독립사건결합 승격)
  - **alignment_detector.py** (647줄): YOLO-seg 세그멘테이션→서브픽셀 엣지 검출→RANSAC 라인 피팅→LiDAR 기준선 비교→KCS 41 46 01 판정(수직도 ±3mm/m, 직각도 ±2mm/m)
  - **insulation_detector.py** (237줄): U-Net 세그멘테이션 + RGB 컨텍스트 YOLO 건물 요소 검출 → 4종 하자(창호 단열 B-01, 벽체 단열 B-02, 창호 기밀 B-05, 바닥 난방 D-01)
  - **temporal_filter.py** (120줄): window_size 연속 프레임 중 min_detections 이상 보고, 고신뢰(>0.85) 즉시, LiDAR 좌표 공간 중복 억제
  - `defect_taxonomy.py`: DEFECT_20_MAP 20종 class_name→(code, display_ko, severity, area) + `get_20defect_info()`
  - `schemas/detection.py`: DefectDetection, InsulationDetection, AlignmentDetection, DetectionResult20, ModelsLoadedStatus20
  - Alembic: 20종 파이프라인 컬럼 마이그레이션
  - **학습 스크립트 15+개**: auto_train_all.py(410줄 M1~M6 순차+ONNX), auto_train_remaining.py(217줄), train_m1~m6 개별 10개, export_to_onnx.py, eval/benchmark.py+evaluate_all.py, download_and_organize.py, configs/*.yaml 5개, Jupyter 2개

### 비디오 스트리밍 (260417~260424)
- [x] v2.2_260417 — 녹화 + 평면도 OpenCV + 스트리밍 CRUD **50개 파일 중 백엔드 부분**
  - `services/recording.py`: RecordingService — RGB+Thermal 동시 별도 mp4, `_CameraRecorder`(CameraService 구독→cv2.VideoWriter), 시작/중지/상태/목록/다운로드/삭제
  - `services/floorplan_processor.py`: `extract_walls_from_bytes()` — 그레이스케일→이진화→방향성 모폴로지(수평/수직)→Canny→HoughLinesP→정규화 좌표 + `findContours` 건물 외곽
  - `api/stream.py` 확장: GET rgb/thermal/blend MJPEG + POST mode + POST record start/stop + GET status/list + GET/:filename + DELETE/:filename
  - `api/floorplan.py`: POST upload(JPG/PNG/PDF/DXF→DB), POST /:id/process(OpenCV 트리거), POST analyze(Stateless), GET 목록/상세, DELETE
  - `schemas/floorplan.py`: FloorplanUploadResponse/ProcessResponse/AnalyzeResponse/ListResponse
- [x] v2.9_260424 — 테스트 스트림 서비스 **33개 파일 중 백엔드 부분** (+2,787줄)
  - `services/test_stream.py` (1,053줄): TestStreamService — 카테고리별 균등 샘플링, RGB/Thermal 쌍 동기화(프레임 버전 카운터), 재생 제어(start/pause/resume/stop), `rgb_mjpeg_generator()`/`thermal_mjpeg_generator()`, image_crop base64 JPEG 생성, 20종 ONNX 추론 또는 목업 폴백, 7종 디렉토리별 하자 매핑(`_DIR_TO_DEFECT`), 한글 폰트(Malgun Gothic) PIL 텍스트 렌더링
  - `api/stream.py` 테스트 모드 14개: init/start/pause/resume/stop/state/rgb/thermal/source/upload CRUD/detection-mode/defect/:id/:channel
  - `services/floorplan_processor.py` +206줄: `validate_floorplan_quality()` — 해상도/선명도(Laplacian variance)/대비/직선 비율/직각 교차점/기울기/벽체 수 7항목 종합 점수

### 데이터셋 관리 (260423)
- [x] v3.3_260423 — `datasets_sources.md` (9개 데이터셋 63,285장, A-01~E-02 하자코드↔데이터셋↔M1~M6 완전 매핑) + `training/.gitignore`

---

## 작업 목록 — @Hijin554 (branch: Hijin)

### 텔레메트리 + AI 웹훅 + SLAM + 평면도 + 리포팅 (260416)
- [x] v1.0_260416 — **17개 파일** (+1,191줄)
  - `api/ai_webhook.py`: POST /ai/detection (DefectLog DB저장+WS "defects" 브로드캐스트), POST /ai/thermal (DB 미저장, WS "thermal" Push), POST /ai/batch (다건 저장+WS)
  - `api/telemetry.py`: POST /telemetry (저장+WS "telemetry" Push, ROS2/MAVLink 호출용), GET /telemetry (페이지네이션, 최신순), GET /telemetry/latest
  - `api/slam.py`: POST (생성+WS), GET (메타만), GET/:id (이미지포함), PATCH (실시간 갱신+WS 프론트 미니맵), DELETE
  - `api/floorplan.py`: POST upload (aiofiles 비동기), POST /:id/process (OpenCV 트리거 TODO), GET 목록/상세, DELETE (파일+DB)
  - `api/report.py`: POST /report/save (마크다운+하자통계), GET 목록/상세, GET /:id/download (Content-Disposition), DELETE
  - **DB 모델 4개**: telemetry_logs(pos xyz/rpy/vel xyz/battery/flight_mode/is_armed/lidar_distance/sensor_status JSONB), slam_maps(name/resolution/width/height/원점/map_image Base64 PNG/metadata JSONB/status), floorplans(파일정보/처리상태/walls_data JSONB/gazebo_world_path), reports(제목/건물명/점검자/본문 Text/하자통계)
  - Pydantic 스키마 4세트 + router.py 4개 라우터 등록

### 3-모델 추론 + 실시간 스트림 워커 (260420)
- [x] v1.1_260420 — **23개 파일** (+2,139줄)
  - `services/inference_pipeline.py` (460줄): InferencePipeline 싱글톤 — YOLOv8s crack_moisture + delamination + ResNet50 wallpaper 3-모델 순차 추론, `detect()`/`detect_async()`, `_compute_severity()`(YOLO=HIGH, SEVERE_CLASSES=MED), `detect_defects_legacy()` A-E taxonomy 호환
  - `services/wallpaper_classifier.py` (188줄): ResNet50 19클래스 벽지 분류 싱글톤, top1/top3 softmax, torchvision lazy import
  - `services/defect_taxonomy.py` (188줄): WALLPAPER_CLASSES 19개(체크포인트 순서), CLASS_DISPLAY_MAP(**"good"=터짐 Burst 주의**), YOLO_DISPLAY_MAP 3클래스, WALLPAPER_SEVERE_CLASSES(Mold/Damage/Exploded/Defective_Joint/good), LEGACY_MAP, `xyxy_to_xywhn()`
  - `core/stream_inference.py` (239줄): StreamInferenceWorker 싱글톤 — asyncio.Queue(maxsize=1) 드롭큐, 프레임 스킵(N프레임 중 1), 추론 asyncio.to_thread 비블로킹, "stream"+"defects" WS 동시 브로드캐스트, `_to_legacy_events()` 호환
  - `api/detect.py`: POST /detect (단건 multipart→3-모델→DetectionResult), POST /detect/batch (최대 10장)
  - `api/ws_stream.py`: WS /ws/stream 바이너리 JPEG 수신→cv2.imdecode→드롭큐 submit, 텍스트 제어(ping/pong)
  - `schemas/detection.py`: DetectionResult, YoloDetection, WallpaperPrediction, WSStreamMessage, HealthResponse
  - `models/defect.py`: class_display_en/ko 컬럼 추가
  - Alembic: defect_logs display 컬럼 마이그레이션
  - `tests/test_inference_pipeline.py` (193줄): xyxy→xywhn 회귀, "good"=Burst taxonomy, /health, /detect 503/400/404

### 이중 게이트 도입 (260421)
- [x] v1.2_260421 — 벽지 분류 이중 게이트
  - `services/inference_pipeline.py`: `is_confident = (top1_conf >= 0.35) AND (top1_conf - top2_conf >= 0.15)` (기존 단일 0.4→이중 조건, val_acc 54% 대응 오탐 차단)
  - `config.py`: WALLPAPER_CONF_THRESHOLD 0.4→0.35, WALLPAPER_MARGIN_THRESHOLD 0.15

### LiDAR 3D + 이미지 저장소 + 구조화 로깅 (260421)
- [x] v1.3_260421 — **19개 파일** (+819줄)
  - `services/telemetry_cache.py` (104줄): DronePose dataclass 메모리 캐시 싱글톤, `update()` asyncio.Lock O(1), `snapshot()`/`snapshot_fresh()` (5초 stale 판정)
  - `services/image_storage.py` (95줄): DB Base64→파일시스템 전환 `./uploads/defects/{YYYY-MM-DD}/{uuid}.jpg`, `save_base64_jpeg()`/`get_url()`(StaticFiles)/`delete()` 파일 정리
  - `core/logging.py` (74줄): structlog configure_logging() JSON(운영)/컬러(개발), request_id_ctx ContextVar
  - `core/middleware.py` (65줄): RequestIDMiddleware — X-Request-ID 자동 생성/왕복, structlog 바인딩, http.request status/duration_ms
  - `core/stream_inference.py`: `_compute_lidar_xyz()` — telemetry_cache fresh pose + lidar_service 거리→3D 월드 좌표, 추론 결과에 lidar_position:{x,y,z}
  - `main.py`: lidar_service start/stop, telemetry_cache 초기화, RequestIDMiddleware 등록, /health에 lidar/telemetry_cache 상태
  - `api/ai_webhook.py`: image_crop→image_crop_path, image_crop_url 응답
  - `api/defects.py`: image_crop_url 응답 포함
  - `models/defect.py`: image_crop_path 컬럼
  - Alembic: image_crop_path 마이그레이션
  - **테스트 3개**: test_telemetry_cache(갱신/stale/clear), test_image_storage(base64 저장/URL/삭제), test_wallpaper_double_gate(이중 게이트 회귀)

### 커버리지 + 임계값 최적화 (260422)
- [x] v1.4_260422 — **9개 파일** (+421줄)
  - `api/coverage.py` (146줄): GET /coverage/:site_id — site별 텔레메트리 pos_x/pos_y convex hull(Andrew's monotone chain O(n log n)) → Shoelace 면적 → covered/supplied/ratio/uncovered/hull 폴리곤
  - `api/stream.py`: GET /stream/stats — 추론 워커 submitted/processed/dropped, telemetry_cache, LiDAR 상태
  - `scripts/sweep_wallpaper_thresholds.py` (132줄): JSONL 기반 conf/margin 격자 탐색, precision/recall/F1 CSV 출력
  - `tests/test_coverage_geometry.py`: convex hull/면적 기하 단위 테스트

### 모니터링 스키마 + 하자 삭제 클린업 + 테스트 (260422)
- [x] v1.5_260422 — **8개 파일** (+387줄)
  - `schemas/monitoring.py` (69줄): StreamStatsResponse(WorkerStats/TelemetryCacheStats/LidarStats), CoverageResponse(site_id/covered/supplied/ratio/hull)
  - `api/coverage.py`: dict→CoverageResponse Pydantic 전환
  - `api/defects.py`: DELETE 시 image_storage.delete 로직 보완
  - **테스트 3개**: test_coverage_response_shape(UUID/ratio범위/hull형태/부족fallback/supplied optional), test_defect_delete_cleanup(DB→파일 순서 보장, 404), test_wallpaper_double_gate 보완

### 텔레메트리 site_id FK + 테스트 리팩토링 (260422)
- [x] v1.6_260422 — **7개 파일** (+272줄)
  - Alembic: telemetry_logs site_id FK 추가 (ondelete=SET NULL, nullable, indexed)
  - `models/telemetry.py`: site_id 컬럼
  - `schemas/telemetry.py`: Create/Response에 site_id
  - `api/coverage.py`: site별 필터링 + fallback(0건→전역 최근 N건, note 메시지)
  - `tests/test_defects_api.py` 전면 리팩토링: 조직 스코핑 적용, dependency_overrides+AsyncMock, 인증 401 테스트

### Refresh Token + Auth Guards + Prometheus + Push + Redis WS (260422)
- [x] v1.7_260422 — **26개 파일** (+1,532줄)
  - `core/jwt.py`: create_refresh_token(type="refresh"), decode_refresh_token(교차 차단), `_decode()` 공통(레거시 호환)
  - `api/auth.py`: POST /auth/refresh + login/OAuth 응답에 refresh_token
  - `core/metrics.py` (158줄): PrometheusMiddleware(HTTP method/path/status 자동), 커스텀 6종(stream_frames submitted/processed/dropped, defect_detected severity별, lidar_distance, telemetry_cache_age, queue_size), `render_metrics()` /metrics
  - `core/ws_manager_redis.py` (155줄): RedisConnectionManager(broadcast→Redis publish, subscribe→로컬 재분배), `create_ws_manager()` 팩토리(WS_BACKEND=memory|redis, Redis 미기동→로컬 폴백)
  - `services/push_notifications.py` (136줄): PushNotificationService — PUSH_PROVIDER=noop|fcm|apns 디스패처, send_to_user(DeviceToken 전부), _mark_inactive(실패 비활성화)
  - `models/device_token.py` (55줄): device_tokens 테이블 (user_id FK, platform fcm/apns/web, token, is_active, UNIQUE)
  - `api/notifications.py` 추가: POST /tokens(upsert), DELETE /tokens/:id, POST /push/test
  - **인증 가드 보강**: slam 5개 + telemetry GET 2개 + floorplan 전체에 `get_current_user` Depends (POST /telemetry는 ROS2용 의도적 오픈)
  - `api/floorplan.py` 추가: POST /:id/calibrate (scale_px_per_meter, 2점 거리 자동 계산)
  - `models/floorplan.py`: scale_px_per_meter 컬럼
  - `schemas/floorplan.py`: FloorplanCalibrateRequest/Response
  - `schemas/user.py`: RefreshTokenRequest/Response
  - `main.py`: PrometheusMiddleware 등록, /metrics
  - `config.py`: JWT_REFRESH_EXPIRE_DAYS, LOG_JSON/LEVEL, PUSH_PROVIDER, WS_BACKEND, REDIS_URL
  - Alembic 2개: device_tokens, floorplans scale_px_per_meter
  - **테스트 6개**: test_refresh_token(발급/검증/type혼용차단/만료), test_metrics(카운터/gauge), test_push_service(noop/디바이스 없음), test_ws_manager_redis(팩토리/메모리 폴백), test_logging_json(JSON 출력 유효성), test_floorplan_calibration(스케일 보정 계산)

---

## 테스트 현황 (16개 파일)
| 테스트 파일 | 담당 | 검증 대상 |
|------------|------|----------|
| test_inference_pipeline.py | @Hijin554 | xyxy→xywhn, taxonomy "good"=Burst, /health, /detect 503/400/404 |
| test_yolo_inference.py | @Hijin554 | YOLO 추론 단위 |
| test_wallpaper_double_gate.py | @Hijin554 | 이중 게이트 conf/margin 조건 회귀 |
| test_ws_manager.py | @Hijin554 | WebSocket 매니저 연결/브로드캐스트 |
| test_ws_manager_redis.py | @Hijin554 | Redis 팩토리/메모리 폴백 |
| test_refresh_token.py | @Hijin554 | 리프레시 발급/검증/type 혼용/만료 |
| test_push_service.py | @Hijin554 | noop 모드 발송/디바이스 없음 |
| test_telemetry_cache.py | @Hijin554 | 캐시 갱신/stale/clear |
| test_metrics.py | @Hijin554 | Prometheus 카운터/gauge |
| test_logging_json.py | @Hijin554 | JSON 출력 유효성 |
| test_image_storage.py | @Hijin554 | base64 저장/URL/삭제 |
| test_defects_api.py | @Hijin554 | 조직 스코핑 CRUD, 인증 401 |
| test_defect_delete_cleanup.py | @Hijin554 | DB→파일 삭제 순서 보장 |
| test_coverage_geometry.py | @Hijin554 | convex hull/Shoelace 면적 기하 |
| test_coverage_response_shape.py | @Hijin554 | CoverageResponse 스키마 검증 |
| test_floorplan_calibration.py | @Hijin554 | 스케일 보정 계산 |

---

## 요구사항
1. `team_project_rules.md` 준수
2. Python `# //!`, `# //*` Better Comment
3. API 응답 스키마 변경 시 프론트엔드 정합성 체크 필수
4. 모든 엔드포인트 에러 핸들링 (4xx, 5xx)

---

## Revision History

### v6.5_260527 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.09) 운영 신뢰성 가이드 + PostgreSQL 백업 스크립트 + 콜드스타트 옵션** — Track D-3 (`DEPLOYMENT_GUIDE.md` 신규 — 시크릿 등록/마이그레이션/백업·복구 RTO·RPO/콜드스타트 트레이드오프/Sentry 통합/감사 로그 운영/롤백/CI·CD/보안 체크리스트/장애 시나리오 10 섹션, `scripts/backup_pg.ps1` 신규 — pg_dump custom format + R2 업로드 + RETENTION_DAYS 정책, `fly.toml` 의 `min_machines_running` 에 상업 운영 권장값 가이드 주석).

### v6.4_260527 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.08) 하자 검수 메타 + 감사 로그 인프라** — Track B+C backend (상업 도메인 책임 추적 기반).
  - `app/models/defect.py` 컬럼 추가: `review_status` (Enum pending/approved/rejected/flagged_false_positive, server_default=pending), `reviewed_by_user_id` (FK users.id SET NULL), `reviewed_at`, `review_note`, `detection_model_id`, `gps_lat/lon/alt`. 인덱스 2개 추가 (`idx_defect_review_status`, `idx_defect_reviewer`).
  - `app/models/audit_log.py` 신규: AuditLog (user_id/organization_id/action 점구분 doted-name/resource_type/resource_id/before·after JSONB/ip/user_agent/request_id/note/created_at). 인덱스 4종 (org/user/resource/action × created_at DESC).
  - `app/services/audit_logger.py` 신규: `write_audit()` 헬퍼 — 민감 키(password/token/secret/api_key/authorization/cookie/session/private_key/client_secret 등) 재귀 redact. structlog `request_id_ctx` 자동 첨부. 실패 silent (감사 로그가 메인 트랜잭션 막지 않음).
  - `app/schemas/defect.py`: DefectLogResponse 에 6개 신규 필드 + `DefectReviewRequest` 신규 (rejected/flagged_false_positive 는 review_note 필수).
  - `app/schemas/audit_log.py` 신규: AuditLogResponse/AuditLogListResponse/AuditLogFilter.
  - `app/api/defects.py`: `PATCH /defects/{id}/review` 신규 — 조직 격리 + audit_logger 자동 호출 + WS "defect.reviewed" broadcast. `GET /defects/{id}/audit-trail` 신규 — 단일 하자 감사 이력. DELETE 에 `write_audit("defect.delete", before=snapshot)` 추가.
  - `app/api/audit_logs.py` 신규: `GET /audit-logs` (admin/owner/superadmin, 조직 격리 + action prefix/resource/user/시각 필터 + 페이지네이션), `GET /audit-logs/{id}` (단건).
  - `app/api/router.py`: audit_logs 라우터 등록 (`/audit-logs`, tags=Audit).
  - `alembic/versions/n7b8c9d0e1f2_add_defect_review_and_audit_logs.py` 신규: down=`m6a7b8c9d0e1`. defect_logs 8 컬럼 추가 + audit_logs 테이블 + FK + 인덱스 6종.
  - 검증: `python -m py_compile` OK, route 등록 검증 PASS (defects 8 + audit 2).

### v6.3_260527 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.07) ONNX 4-way 매핑 회귀 가드** — Track D-2 — 5/7 거짓 라벨 사고 재발 방지.
  - `tests/test_onnx_class_mapping.py` 신규: 9개 모델 parametrize (M1/M2/M3 YOLO + M4_CONTEXT + M5_SEG + furniture_aware + M1/M2/M3 ResNet). ONNX 출력 dim ↔ data.yaml `names` ↔ `EXPECTED_CLASS_NAMES` ↔ `inference_pipeline_20.py` AST 정적 비교까지 4-way 검증.
  - `tests/conftest.py` 신규: `onnx_weights_dir` / `datasets_dir` fixture. ONNX_WEIGHTS_DIR / DATASETS_DIR 환경변수 override 지원.
  - `app/services/defect_taxonomy.py`: `EXPECTED_CLASS_NAMES` 상수 + `validate_class_mapping(model_name, onnx_path, yaml_path)` 헬퍼 + `_infer_onnx_class_count` / `_read_yaml_class_names` 내부 함수.
  - `tests/README.md` 신규: "신규 ONNX 추가 시 본 테스트 실행 필수" 명시.
  - 검증: `pytest tests/test_onnx_class_mapping.py -v` → **11 passed, 0 failed, 0 skipped**. 매핑 불일치 0건 — 운영 ONNX 정합성 확인.

### v6.2_260527 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.06) Sentry 에러 모니터링 통합** — 운영 중 미처리 예외/에러 알림·집계 경로 신설 (운영 갭 해소).
  - `requirements.txt`: `sentry-sdk[fastapi]>=2.0.0` 추가.
  - `app/config.py`: `SENTRY_DSN` (Optional), `SENTRY_ENVIRONMENT`, `SENTRY_TRACES_SAMPLE_RATE` (0.1), `SENTRY_PROFILES_SAMPLE_RATE` (0.0). DSN 미설정 + APP_ENV=production 시 경고 로그만 (기동 차단 X — 로컬 개발 영향 0).
  - `app/core/sentry.py` (신규): `init_sentry(settings)` — FastAPI/Starlette/SQLAlchemy/Asyncio integration. `before_send` 훅에서 password/token/secret/authorization/api_key/cookie 등 민감 키 재귀 redact. structlog `request_id` contextvar 를 Sentry tag 로 승격. `send_default_pii=False`. release 자동 탐지 (SENTRY_RELEASE / FLY_RELEASE_VERSION / GIT_SHA).
  - `app/core/middleware.py`: `RequestIDMiddleware` 가 `sentry_sdk.set_tag("request_id", ...)` + `set_context("request_meta", ...)` 추가 호출. sentry-sdk 미설치 시 silent skip.
  - `app/main.py`: lifespan 시작 첫 단계에서 `init_sentry(settings)` 호출 (이후 startup 오류도 캡처).
  - `.env.example`: SENTRY_DSN / ENVIRONMENT / TRACES_SAMPLE_RATE / PROFILES_SAMPLE_RATE 4개 항목 + 운영 전용 주석.
  - `README.md`: "운영 에러 모니터링 (Sentry)" 섹션 — DSN 발급 → `flyctl secrets set SENTRY_DSN=...` → 검증 절차.
  - 사용자 직접 작업: Sentry 프로젝트(FastAPI) 생성 → DSN 발급 → Fly secrets 등록.

### v6.1_260515 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.05) 챗봇 자동 제목 흐름 요약 강화** — 임시 prefix("안녕하세요" 같은 첫 메시지가 제목으로 굳음) 제거 + 첫 3턴 동안 매 응답 후 LLM 흐름 요약 제목 재생성. `regenerate_thread_title(thread_id)` 시그니처 단순화(내부에서 최근 10건 DB 조회). 프롬프트 강화(명사형 5~7단어, 하자 코드/현장명 키워드 포함, 단순 인사면 '신규 도메인 문의' 일반화). `_is_first_user_message` → `_count_user_messages` 일반화. 마이그레이션 없음(`title_locked` 컬럼은 v1.2 검토).

### v6.0_260515 (작성자: @youminsu0523 / branch: MS)
- **(backend R-v1.1.01) OpenAI 챗봇(건축물·하자 도메인 어시스턴트) 통합** — 통합 repo 와 동일 구현. 분리 repo head 가 `k4e5f6a7b8c9` 이므로 마이그레이션 down_revision 분기.
  - 신규 ORM 2: `AiChatThread`(user_id+organization_id 격리, summary watermark, soft delete) / `AiChatMessage`(role enum, tokens, JSONB meta).
  - 신규 Pydantic 6: ThreadCreate/Update/Response/ListResponse, MessageCreate/Response/HistoryResponse. role=system 응답 제외.
  - 신규 Alembic `m6a7b8c9d0e1` (down=`k4e5f6a7b8c9`): FK 사이클 회피 위해 threads → messages → ALTER threads ADD FK 순.
  - 신규 서비스 `app/services/openai_chat.py` — SYSTEM_PROMPT(DEFECT_CATALOG 20종 + 안전 가이드), astream(SSE), light-RAG, 자동 요약.
  - 신규 라우터 `app/api/ai_chat.py` (6 엔드포인트, `get_current_org_member` 의존성, user_id+org_id 이중 검증, 사용자별 분당 20 메시지 카운터).
  - router.py include + rate_limit.py 한도, settings 4개(OPENAI_API_KEY/MODEL/MAX_OUTPUT_TOKENS/SUMMARY_MODEL), requirements `openai>=1.40.0`.

### v5.1_260503 (작성자: @youminsu0523 / branch: MS)
- **R26 후속 정정**: tasks 문서(API 명세서 v1.1→v1.2, ERD v1.0→v1.1) 부록을 본문 인라인 위치(4.17 Employee API / 2.1.5 Swagger securityScheme / 8.5 운영 가드 / 4.19 inspection_schedules / 5장 관계 / 6.1 인덱스 / 13장 결론)로 분산 + 파일명 rename + 팀명 `다마코더 → AeroInspect`. 가이드 3종 문서이력 위치 정정.
- **DB 시드 실 적용**: `alembic merge` 로 분기 head 2개(`0003`, `i2c3d4e5f6a7`) → `89b53c16de85` 병합 → `upgrade head` 성공. `defect_logs` 의 alembic_version 과 실제 컬럼 inconsistent (image_crop_path/track_id/accumulated_conf/tier_executed/deviation_*/delta_temperature/ensemble_boosted/defect_class_display_*/) 10건 `ADD COLUMN IF NOT EXISTS` 일괄 보정. `seed_demo_data --reset` 결과: 1 org / 3 depts / 2 users(백승희·오희진) / 8 sites / **315 defects (HIGH 77)** / 12 reports / 3 today schedules (잠실 리센츠 14:00 KST 백승희 시드 검증).

### v5.0_260503 (작성자: @youminsu0523 / branch: MS)
- **R19 (4/28)** Recall 후처리 파이프라인 신설 — TemporalFilter Noisy-OR / ByteTrack ObjectTracker / SAHI TiledInference / ActiveLearning Hard Example Mining / DefectPersistence + 단위 테스트 4종. `app/models/defect.py`에 track_id/accumulated_conf/tier_executed 컬럼 추가.
- **R20 (4/28~4/30)** M1~M5 학습 스크립트 일괄 보강 + 데이터셋 빌더 6종 (compress_m1, build_m4_context, convert_ade20k, build_m5v2, auto_run_m4v2, build_furniture_aware).
- **R21 (5/2)** ONNX inference + TTA 후처리, furniture_gate / geometric_gate 신설 (가구 위 false positive 차단, 수직수평 편차 검증) + 단위 테스트.
- **R22 (5/3 새벽)** 통합 평가 파이프라인 정착 — dry_run_full_pipeline / postprocess_ablation / evaluate_integrated. detection schema 확장.
- **R23 (5/3 오후)** 후처리 강도 정책 정착 — postprocess_config.yaml 단일 소스, ensemble + furniture_gate, evaluate_ultralytics_val + evaluate_max_boost 측정. DEPLOYMENT_GUIDE 작성.
- **R24 (5/3 본 세션)** Swagger Phase 1~3 — main.py에 HTTPBearer(bearerFormat=JWT)/AIWebhookSecret 보안 스키마 명시 등록, 17개 tags_metadata, persistAuthorization. PROTECTED/PUBLIC/WEBHOOK 공통 responses(401/403). schemas/common.py 신규. user/site/defect schema에 example 추가. config.py/init_db.py에 `APP_ENV=production` 가드 (placeholder secret 차단 + create_all 자동 스킵, alembic 책임 분리). `.env.example` APP_ENV·AI_WEBHOOK_SECRET·PUSH_PROVIDER·OAUTH_REDIRECT_BASE 보강.
- **R25 (5/3 본 세션)** InspectionSchedule 모델 + alembic migration `i2c3d4e5f6a7` 신규. `/api/v1/employee` 라우터(schedule/today + kpi/monthly + activities) 신규 — 조직 단위 격리. `scripts/seed_demo_data.py` 신설: 조직(DRONE INSPECT 데모) + 부서 3 + 사용자(백승희/오희진) + 현장 8 + 하자 25~60건/현장 + 보고서 3~5건/완료현장 + 오늘 일정 3건(09:00 헬리오시티/14:00 잠실 리센츠 백승희/16:30 잠실 엘스 오희진) + 알림 8종. idempotent + APP_ENV 가드 + `--reset`/`--force-prod` 옵션. router.py에 employee 등록.

## v1.1 사이클 (2026-05-07 ~ 2026-05-31) — 3차 프로젝트 최종 제출 사이클

> 5/6 1차 배포 후 시작. memory: 자유 진행 X, 약한 모델 보완 + 검증 통과 후 최종 제출.

- **R-v1.1.01~05 (5/7~5/26)** v1.1 초기 작업 — 영상 수신기 미도착 임시 정책 (testMode 위장), 모델 v1.1 ckpt 학습 라운드들 (m1_v4/v5, m2_v4/v4s, m3_v4/v4s, m4_v2, m5_v2, m6 PatchCore 재구축). 약점 모델 Roboflow 데이터 보강 + 자체 학습 (3자 .pt 다운 금지, [[project_roboflow_finetune_program]]).
- **R-v1.1.06 (5/26)** Sentry 에러 모니터링 통합 — DSN 환경변수 + APP_ENV 기반 활성화 + sensitive data scrubbing.
- **R-v1.1.07 (5/27)** ONNX 4-way 매핑 회귀 가드 — 신규/갱신 모델 통합 시 ONNX dim ↔ data.yaml/CLASS_NAMES ↔ inference 매핑 ↔ taxonomy 4-way cross-check. 5/7 검출 거짓 라벨 5건 동시 사고 재발 방지 ([[feedback_onnx_class_mapping_audit]]).
- **R-v1.1.08 (5/27)** 하자 검수 메타 + 감사 로그 인프라 — defect_log review/audit-trail 라우터 8 routes. flagged_false_positive 컬럼 추가 (Active Learning hook). audit_logs 테이블 + admin/owner/superadmin 권한.
- **R-v1.1.09 (5/27 오후)** 운영 신뢰성 가이드 + PostgreSQL 백업 — DEPLOYMENT_GUIDE.md 작성, scripts/backup_pg.ps1 (pg_dump custom format + R2 업로드), fly.toml min_machines_running 가이드 주석.
- **R-v1.1.10 (5/28 오전)** 신뢰도 3단계 등급 시스템 + Thermal/M4 재설계 학습 스크립트 — `confidence_grader.py` 신규 (CONFIRMED/REVIEW/REFERENCE/DROP). PatchCore/anomaly 단독은 CONFIRMED 불가. schema/Pipeline20에 grade 필드. 20종 클래스 통일 (단열 특례 폐지). 학습 스크립트 신규: train_m4_context_seg.py (bbox→seg 전환), prepare_thermal_anomaly.py + train_thermal_anomaly.py (Moisture/delam → PatchCore unsupervised), cleanup_furniture_coco.py.
- **R-v1.1.11 (5/28 오전)** v1.2 학습 chain 가동 + 자동저장 안전장치 — train_chain_v1_2.py (STAGES=[M4_Seg, ThermalAnomaly, Furniture] precondition_ok 자동 검증). monitor_report.py META 확장 (seg 모델 경로 분기). backup_checkpoints.py 신규 (10분 best.pt/last.pt 복제).
- **R-v1.1.12 (5/28 오후)** chain 사고 복구 + thermal_anomaly 사전 통합 + verify_test_mode — M4_Seg 38초 실패 진단(bbox 라벨 80%) → validate_m4_seg_labels.py + convert_m4_bbox_to_polygon.py (95,875개 변환 + 원본 백업). config.py THERMAL_ANOMALY_ONNX 키. defect_taxonomy thermal_anomaly_area 클래스. inference_pipeline_20 `_anomaly_mask_to_bboxes` 헬퍼 + thermal_frame_bgr 시그니처. verify_test_mode.py (등급별 시각화 + Recall proxy).
- **R-v1.1.13 (5/28 오후)** Thermal Anomaly 일시 보류 + stream thermal_frame_bgr 사전 전달 — 사용자 명시 ("thermal은 일단 보류"). THERMAL_ANOMALY_ENABLED=False 토글. M4 U-Net 단열은 유지. stream_inference QueuedFrame/submit/_process_20 thermal_frame_bgr 전달.
- **R-v1.1.14 (5/29)** chain 사후 처리 + 노트북 OFF 복구 절차 — export_furniture_onnx.py (Furniture cuDNN 사고 best.pt 0.349 → ONNX 98.9MB). train_m4_context_seg.py cuDNN 안전화 (amp=False/workers=2/cache=False) + verify_test_mode 자동 호출. resume_m4_seg.py 신규 (ultralytics resume=True 복구). 노트북 OFF 무손실 검증.
- **R-v1.1.15 (5/29 오후)** M4 seg epoch 30 중간 ONNX + verify 경로 버그 수정 — best mAP50-95 0.483 baseline +0.128 (M5 seg 사례 +0.111 초과). verify_test_mode 경로 버그 2건 수정 (cwd backend + roboflow test/images 재귀). 257장 검출률 100% (놓침 0), CONFIRMED 1018 / REVIEW 369 / REFERENCE 673.
- **R-v1.1.16 (5/30)** M4 epoch 60 best 0.503 + GT Precision 검증 + grade 임계 조정 — M4 seg epoch 30→60 완주, best mAP50-95 **0.503** (baseline +0.148 **+41.7%**). verify_gt_precision.py 신규 (roboflow GT IoU 매칭 + FP source 분석). GT 3차 시도: 1차 P 0.535/R 0.748, 2차 (임계 0.85→0.90) 거의 동일, 3차 (M2/M3 voting 필수) Recall 폭락 롤백. 결론: 도메인 mismatch (test_external 외부 도메인). grade CONFIRMED_STRONG 0.90 + WITH_VOTING 0.75 적용. Furniture 재학습 취소 (FP 0건 기여).

### v4.0_260427 (작성자: @youminsu0523 / branch: MS)
- 전면 재작성: git log 기반 @youminsu0523 + @Hijin554 10일간 백엔드 작업 상세 기록
- 각 버전별 변경 파일 수, 함수명/클래스명/API 엔드포인트/DB 컬럼/스키마/테스트 상세

### v1.0_260413 (작성자: @Antigravity / branch: main)
- 백엔드 Task.md 초기 생성
