# Backend Implementation Plan

## 아키텍처 개요
- **프레임워크**: FastAPI (비동기 Python, uvicorn)
- **데이터베이스**: PostgreSQL + SQLAlchemy ORM (async) + Alembic 마이그레이션
- **캐시/브로커**: Redis (Pub/Sub WebSocket 브로커, 텔레메트리 캐시)
- **AI 추론**: ONNX Runtime (20종 결함 분류), YOLOv8 (객체 탐지), ResNet50 (분류), U-Net (세그멘테이션), PatchCore (이상 탐지)
- **모니터링**: Prometheus (HTTP 메트릭 + 커스텀 6종 + /metrics 엔드포인트)
- **로깅**: structlog JSON/컬러 + RequestID 미들웨어
- **인증**: JWT HS256 (Access + Refresh) + OAuth 2.0 (Google/Kakao/Naver)
- **실시간 통신**: WebSocket (인메모리 or Redis Pub/Sub) + MJPEG 스트리밍
- **푸시 알림**: FCM/APNs 스켈레톤 (noop/fcm/apns 전환)

### 앱 초기화 흐름 (main.py lifespan)
```
1. DB 초기화 (init_db) + 슈퍼어드민 시드 (admin/admin)
2. DRONE_CONNECTED=True 일 때만:
   ├── RGB 카메라 open (rgb_camera_service)
   ├── Thermal 카메라 open (thermal_camera_service)
   ├── YOLOv8 3-모델 로드 (yolo_service.load_model)
   ├── TF-Luna LiDAR 시리얼 연결
   └── StreamInferenceWorker 시작
3. DRONE_CONNECTED=False: API 전용 모드 (카메라/LiDAR/추론 건너뜀)
4. 미들웨어 등록: CORS → RequestID → Prometheus (LIFO)
5. /uploads StaticFiles 마운트
```

### 데이터 흐름
```
[드론/클라이언트]
  ├── REST API → FastAPI Router → Service → SQLAlchemy → PostgreSQL
  ├── WS 텔레메트리 → telemetry_cache O(1) 메모리 → WS "telemetry" 브로드캐스트
  ├── WS /ws/stream → 바이너리 JPEG → StreamInferenceWorker(드롭큐+to_thread)
  │     └── 3-Model or 20종 ONNX → WS "stream"+"defects" 동시 브로드캐스트
  ├── POST /ai/detection → DefectLog DB + WS "defects"
  ├── POST /ai/thermal → WS "thermal" (DB 미저장, 대시보드 Recharts용)
  └── MJPEG GET /stream/rgb|thermal|blend → StreamingResponse
```

### API 라우터 전체 맵 (router.py — 17개 라우터, 60+ 엔드포인트)
| Prefix | 모듈 | Tag | 담당 | 엔드포인트 |
|--------|------|-----|------|-----------|
| `/auth` | auth | Auth | @youminsu0523, @Hijin554 | POST signup/login/refresh, GET me/check-email/check-username |
| `/oauth` | oauth | OAuth | @youminsu0523 | POST google/kakao/naver (code→token→userinfo→JWT) |
| `/sites` | sites | Sites | @youminsu0523 | GET(필터+페이지네이션)/GET/:id/POST/PATCH/DELETE |
| `/defects` | defects | Defects | @youminsu0523, @Hijin554 | GET(summary/필터)/GET/:id/DELETE(+image_storage 파일 정리) |
| `/detect` | detect | Detect | @Hijin554 | POST 단건/batch(10장) multipart 3-모델 추론 |
| `/stream` | stream | Stream | @youminsu0523 | GET rgb/thermal/blend MJPEG, POST mode/record start-stop, GET stats, 테스트 모드 14개 |
| `/report` | report | Report | @youminsu0523, @Hijin554 | POST generate(LLM)/save, GET 목록/상세/download, DELETE |
| `/chat` | chat | Chat | @youminsu0523 | GET conversations/messages/unread-counts, POST conversations/messages, PATCH read, DELETE leave |
| `/organizations` | organization | Organizations | @youminsu0523 | GET my/members/admin/all-orgs/departments, POST create/invite/assign, PATCH members, DELETE members |
| `/notifications` | notifications | Notifications | @Hijin554 | GET(필터+페이지네이션)/unread, PATCH read/read-all, DELETE, POST tokens/push/test |
| `/telemetry` | telemetry | Telemetry | @Hijin554 | POST(ROS2/MAVLink→DB+WS, 인증 오픈), GET 목록/latest(인증 필요) |
| `/ai` | ai_webhook | AI Webhook | @Hijin554 | POST detection/thermal/batch → DB+WS 브로드캐스트 |
| `/coverage` | coverage | Coverage | @Hijin554 | GET /:site_id (convex hull 면적 산출) |
| `/floorplan` | floorplan | Floorplan | @Hijin554 | POST upload/process/analyze/calibrate, GET 목록/상세, DELETE |
| `/slam` | slam | SLAM | @Hijin554 | POST/GET 목록/GET:id/PATCH(+WS 미니맵)/DELETE — 전부 인증 가드 |
| (없음) | websocket | WebSocket | @Hijin554 | 실시간 이벤트 채널 (telemetry/defects/stream/thermal/camera) |
| (없음) | ws_stream | WS Stream | @Hijin554 | WS /ws/stream 바이너리 JPEG 프레임 수신 |

### 모듈 의존 관계
```
api/auth.py ──→ core/jwt.py ──→ core/security.py (bcrypt)
api/oauth.py ──→ core/jwt.py (Google/Kakao/Naver→JWT)
api/defects.py ──→ services/inference_pipeline.py ──→ services/wallpaper_classifier.py
                                                   ──→ services/defect_taxonomy.py
api/detect.py ──→ services/inference_pipeline.py
api/stream.py ──→ services/test_stream.py ──→ core/streaming.py
                                           ──→ services/inference_pipeline_20.py
services/inference_pipeline_20.py ──→ services/onnx_inference.py (YOLO/ResNet/UNet/PatchCore)
                                  ──→ services/alignment_detector.py (RANSAC+KCS)
                                  ──→ services/insulation_detector.py (U-Net+RGB)
                                  ──→ services/ensemble.py (cross_model_nms+PatchCore)
                                  ──→ services/temporal_filter.py (시간 일관성)
api/telemetry.py ──→ services/telemetry_cache.py ──→ Redis
api/websocket.py ──→ core/ws_manager.py ──→ core/ws_manager_redis.py ──→ Redis
api/coverage.py ──→ models/telemetry.py (convex hull+Shoelace)
api/report.py ──→ services/llm_report.py (Anthropic/Google AI)
api/ai_webhook.py ──→ services/defect_processor.py ──→ services/image_storage.py
core/stream_inference.py ──→ services/inference_pipeline.py
                         ──→ services/telemetry_cache.py (LiDAR 3D 좌표)
                         ──→ core/ws_manager.py (이중 WS 브로드캐스트)
```

### DB 스키마 (18 ORM 모델)
| 모델 | 테이블 | 주요 컬럼 | FK 관계 |
|------|--------|----------|---------|
| User | users | UUID PK, account_type(personal/business), email/username unique, password_hash(bcrypt), oauth_provider/oauth_id | → business_profiles, term_agreements, conversations, messages |
| BusinessProfile | business_profiles | biz_number(10자리 unique), ceo_name, verified_at | FK user_id (PK) |
| Term | terms | code, title, is_required, version, effective_from | |
| UserTermAgreement | user_term_agreements | version 스냅샷 | FK user_id, term_id |
| Organization | organizations | name, biz_number | → organization_members |
| OrganizationMember | organization_members | role(owner/admin/member), status(active/invited/deactivated), department, position | FK org_id, user_id |
| Site | sites | 20+ 컬럼: seq, name, inspection_type 6종, building_type 7종, total_area, assigned_members JSONB, recordings JSONB, status 4종 | |
| Defect | defect_logs | class_name, confidence, bbox, severity, image_crop_path, class_display_en/ko, lidar_x/y/z | FK site_id |
| Report | reports | title, building_name, inspector, body Text, defect/high/med/low count | |
| Conversation | conversations | type(dm/group/channel), name | FK created_by → users |
| ConversationMember | conversation_members | last_read_at | FK conversation_id, user_id |
| Message | messages | text | FK conversation_id, sender_id |
| Notification | notifications | category 10종 Enum, title, message, metadata JSONB, is_read | FK user_id |
| DeviceToken | device_tokens | platform(fcm/apns/web), token, is_active, UNIQUE(user_id,token) | FK user_id |
| Floorplan | floorplans | file_path, status, walls_data JSONB, gazebo_world_path, scale_px_per_meter | |
| SlamMap | slam_maps | name, resolution, width, height, origin coords, map_image Base64, metadata JSONB, status | |
| Telemetry | telemetry_logs | pos xyz, rpy, vel xyz, battery, flight_mode, is_armed, lidar_distance, sensor_status JSONB, site_id FK | FK site_id → sites |
| Department | departments | name, org_id | FK org_id |

### AI 학습 모델 현황
| ID | 학습 스크립트 | 아키텍처 | 용도 | 데이터셋 |
|----|-------------|---------|------|---------|
| M1 | train_m1_resnet_crack, train_m1_yolo_structural | YOLO+ResNet 2-Stage | 균열 탐지 (crack_structural/crack_finishing) | 63,285장 중 균열 관련 |
| M2 | train_m2_resnet_surface, train_m2_yolo_surface | YOLO+ResNet 2-Stage | 표면 결함 (wallpaper_seam/bubble/paint_stain/scratch) | 벽지 19클래스 |
| M3 | train_m3_resnet_floor_window, train_m3_yolo_floor_window | YOLO+ResNet 2-Stage | 바닥/창문 결함 | 바닥/창호 데이터 |
| M4 | train_m4_thermal_unet | U-Net | 열화상 단열 세그멘테이션 (B-01/B-02/B-05/D-01) | 열화상 이미지 |
| M5 | train_m5_frame_seg | YOLO-seg | 프레임 세그멘테이션 → 기하학 정밀 분석 (RANSAC+KCS 41 46 01) | 프레임 이미지 |
| M6 | train_m6_patchcore | PatchCore | 이상 탐지 앙상블 폴백 | 정상/이상 이미지 |

---

## 구현 계획 (단계별)

### Phase 1. FastAPI 초기 구조 (260414) ✅
- **담당**: @youminsu0523
- **상세**: main.py(lifespan/CORS/health), requirements.txt(40+ 패키지)

### Phase 2. 텔레메트리 + AI 웹훅 + SLAM + 평면도 + 리포팅 (260416) ✅
- **담당**: @Hijin554
- **상세**: 17파일 +1,191줄. 4 API 라우터(ai_webhook/telemetry/slam/floorplan) + report 확장 + DB 모델 4개 + 스키마 4세트

### Phase 3. 인증 시스템 (260416~260420) ✅
- **담당**: @youminsu0523 (회원가입/JWT/OAuth), @Hijin554 (Refresh Token/Auth Guards)
- **상세**: User 4관련 모델, bcrypt, JWT HS256(Access+Refresh type 분리), OAuth 3종(Google/Kakao/Naver), race condition 대응, 인증 가드 11+ 엔드포인트

### Phase 4. 녹화 + 평면도 OpenCV + 스트리밍 (260417) ✅
- **담당**: @youminsu0523
- **상세**: RecordingService(RGB+Thermal mp4), floorplan_processor(OpenCV 벽체추출), stream.py(MJPEG+녹화 CRUD)

### Phase 5. 현장 관리 CRUD (260418) ✅
- **담당**: @youminsu0523
- **상세**: Site 모델(20+ 컬럼, JSONB, Enum), API 5개(필터+페이지네이션)

### Phase 6. 3-모델 추론 + 스트림 워커 (260420) ✅
- **담당**: @Hijin554
- **상세**: 23파일 +2,139줄. InferencePipeline(460줄, 3-모델), WallpaperClassifier(188줄, 19클래스), defect_taxonomy(188줄), StreamInferenceWorker(239줄, 드롭큐), detect REST/WS, 테스트

### Phase 7. 채팅 + 알림 + 조직 (260420~260424) ✅
- **담당**: @youminsu0523
- **상세**: DB 모델 5개, API 4라우터(chat/notifications/organization/auth), DM 중복방지, 대화 나가기, 슈퍼어드민 cross-org, 서비스 2개(email/notification)

### Phase 8. LiDAR 3D + 이미지 저장소 + 로깅 (260421) ✅
- **담당**: @Hijin554
- **상세**: 19파일 +819줄. telemetry_cache(DronePose O(1)), image_storage(파일시스템), structlog+RequestID, LiDAR 3D 좌표 주입, 이중 게이트

### Phase 9. 커버리지 + 모니터링 + 테스트 (260422) ✅
- **담당**: @Hijin554
- **상세**: convex hull 면적, 모니터링 스키마, 하자 삭제 파일 정리, site_id FK, 임계값 격자 탐색, 테스트 6+개

### Phase 10. Prometheus + Redis WS + Push + Auth Guards (260422) ✅
- **담당**: @Hijin554
- **상세**: 26파일 +1,532줄. PrometheusMiddleware(HTTP+커스텀 6종), RedisConnectionManager(pub/sub 팩토리), push_notifications(FCM/APNs), device_tokens, calibrate, 테스트 6개

### Phase 11. 20종 ONNX 파이프라인 (260422) ✅
- **담당**: @youminsu0523
- **상세**: 41파일 +6,392줄. ONNX 4클래스, 20종 파이프라인(M1~M6 Tier), ensemble, alignment(647줄 KCS), insulation(237줄), temporal(120줄), 학습 15+개

### Phase 12. 테스트 스트림 서비스 (260424) ✅
- **담당**: @youminsu0523
- **상세**: TestStreamService(1,053줄), 14엔드포인트, floorplan_quality(7항목)

---

## 향후 계획 (미완료)

### Phase 13. DB 실가동
- [ ] PostgreSQL 실DB 연결 (AWS 프리티어 → 최종 단계)
- [ ] Alembic 마이그레이션 실행 (현재 5+개 버전)
- [ ] 시드 데이터 (약관 3종, 슈퍼어드민)

### Phase 14. Redis 서버 구성
- [ ] Redis 서버 세팅
- [ ] Pub/Sub 채널 설계 확정 (telemetry/defects/stream/thermal/camera)
- [ ] 텔레메트리 캐시 TTL 정책

### Phase 15. 파일 스토리지
- [ ] Cloudflare R2 Presigned URL 연동
- [ ] image_storage.py R2 어댑터

### Phase 16. 점검 커버리지 고도화
- [ ] 드론 비행 후 가용면적 자동 계산 + 미점검 구역 리포트

### Phase 17. 성능 최적화 & 배포
- [ ] ONNX 배치 처리 최적화
- [ ] Docker 컨테이너화
- [ ] CI/CD 파이프라인

### Phase 18. ML 후처리 파이프라인 (2026-04-28 ~ 2026-05-03)
- [x] TemporalFilter Noisy-OR (시간 일관성 검증)
- [x] ByteTrack ObjectTracker (track_id 부여, IoU 매칭)
- [x] SAHI TiledInference (작은 객체 탐지)
- [x] ActiveLearning Hard Example Mining (저신뢰 자동 수집)
- [x] DefectPersistence (track 기반 중복 제거)
- [x] TTA (Test-Time Augmentation) + furniture_gate / geometric_gate
- [x] Ensemble (PatchCore boost) + postprocess_config.yaml 단일 소스
- [x] 통합 평가: dry_run_full_pipeline / postprocess_ablation / evaluate_integrated / evaluate_ultralytics_val / evaluate_max_boost
- [x] 단위 테스트 7종 (test_{temporal_filter, object_tracker, tiled_inference, ensemble, furniture_gate, geometric_gate, tta}.py)

### Phase 19. Swagger / 운영 보안 가드 (2026-05-03)
- [x] OpenAPI custom function — HTTPBearer(bearerFormat=JWT) + AIWebhookSecret(apiKey header) 보안 스키마 명시 등록
- [x] 17개 tags_metadata + servers + contact + persistAuthorization
- [x] 공통 에러 응답 (PROTECTED_RESPONSES = 401/403, WEBHOOK_RESPONSES = 401)
- [x] schemas/common.py (ErrorResponse, *_RESPONSES)
- [x] 핵심 schema 4종 example (LoginRequest / TokenResponse / SiteCreate / DefectLogCreate)
- [x] config.py — `APP_ENV=production` 시 placeholder secret 차단 (RuntimeError)
- [x] init_db.py — `APP_ENV=production` 시 `Base.metadata.create_all` 자동 스킵 (alembic 책임 분리)
- [x] `.env.example` 보강 (APP_ENV, AI_WEBHOOK_SECRET, JWT_REFRESH_EXPIRE_DAYS, PUSH_PROVIDER, WS_BACKEND, OAUTH_REDIRECT_BASE, SMTP_*)
- [x] 보안 점검: .env git 추적 0건 / 소스 하드코딩 시크릿 0건 검증

### Phase 20. Mockup → DB 전환 + 시연 시드 (2026-05-03)
- [x] InspectionSchedule 모델 신규 (id/site_id/operator_user_id/organization_id/scheduled_at/status/note + 2 인덱스)
- [x] alembic migration `i2c3d4e5f6a7_add_inspection_schedules.py` 신규 — 총 12 리비전
- [x] `/api/v1/employee` 라우터 신규 — schedule/today + kpi/monthly + activities (조직 단위 격리)
- [x] router.py + main.py tags_metadata에 Employee 등록
- [x] `scripts/seed_demo_data.py` 신설 — 조직(DRONE INSPECT 데모) + 부서 3 + 사용자(백승희/오희진) + 현장 8 + 하자 25~60건/현장 + 보고서 3~5건/완료현장 + 오늘 일정 3건 + 알림 8종/사용자
- [x] idempotent (사전 SELECT) + `--reset`/`--force-prod` 옵션 + APP_ENV 가드
- [x] **alembic 분기 head 병합** — `0003` + `i2c3d4e5f6a7` → `alembic merge` → `89b53c16de85` mergepoint, `upgrade head` 성공
- [x] **DB DDL 보정** — `defect_logs` 의 `image_crop_path`, `track_id`, `accumulated_conf`, `tier_executed`, `deviation_*`, `delta_temperature`, `ensemble_boosted`, `defect_class_display_*` 10개 컬럼 `ADD COLUMN IF NOT EXISTS` (alembic_version 적용 완료지만 실제 DDL 미반영 상태였음)
- [x] **시드 실 적용** — `python -m scripts.seed_demo_data --reset` 실행: org=1, depts=3, users=2(백승희/오희진), sites=8, defects=**315 (HIGH 77)**, reports=12, today_schedules=3 (잠실 리센츠 14:00 KST 백승희 시드 검증 완료)

### Phase 21. tasks 문서 양식 정정 + 변경 목록 (2026-05-03 R26)
- [x] API 명세서 `v1.1.md` → `v1.2.md` 파일 rename + 부록을 4.17 Employee API · 2.1.5 Swagger securityScheme · 8.5 운영 보안 가드 인라인 위치로 분산
- [x] ERD `v1.0.md` → `v1.1.md` 파일 rename + 4.19 inspection_schedules · 5장 관계 · 6.1 인덱스 · 8.3/12.1 Enum · 13장 결론 카운트(19/12/11/32) 인라인 갱신, 문서 이력 위치 마지막 → 목차 이전 이동
- [x] 가이드 3종(AI 추론 파이프라인 / Frontend Guide / Backend Guide) 문서 이력 위치 정정
- [x] tasks 8개 문서 팀명 `다마코더 → AeroInspect` 일괄 교체
- [x] `CHANGES_2026-05-03.md` 신설 — 내일 Claude 웹 문서 변환용 산출물 목록 + 변환 프롬프트 템플릿 + DB 시드 결과 요약

### Phase 22. Sentry 에러 모니터링 통합 (R-v1.1.06, 2026-05-27)
- [x] requirements: `sentry-sdk[fastapi]>=2.0.0`
- [x] `app/config.py`: SENTRY_DSN/ENVIRONMENT/TRACES_SAMPLE_RATE/PROFILES_SAMPLE_RATE
- [x] `app/core/sentry.py` 신규: init_sentry() — FastAPI/Starlette/SQLAlchemy/Asyncio integration, before_send 민감 키 redact, structlog request_id 자동 첨부, send_default_pii=False
- [x] `app/core/middleware.py`: RequestIDMiddleware 에 sentry_sdk.set_tag 전파 (silent skip)
- [x] `app/main.py`: lifespan 시작 첫 단계 init_sentry 호출
- [x] `.env.example`: 4 항목 + 운영 전용 주석
- [x] README 운영 섹션: flyctl secrets set 가이드

### Phase 23. ONNX 4-way 매핑 회귀 가드 (R-v1.1.07, 2026-05-27)
- [x] `app/services/defect_taxonomy.py`: EXPECTED_CLASS_NAMES 상수 + validate_class_mapping() 헬퍼 + _infer_onnx_class_count + _read_yaml_class_names
- [x] `tests/conftest.py` 신규: onnx_weights_dir/datasets_dir fixture, env override (ONNX_WEIGHTS_DIR/DATASETS_DIR), CI graceful skip
- [x] `tests/test_onnx_class_mapping.py` 신규: 9 모델 parametrize (ONNX dim ↔ data.yaml ↔ 상수 ↔ inference 인자 AST 비교)
- [x] `tests/README.md` 신규: 신규 ONNX 추가 시 본 테스트 필수 명시
- [x] **검증 결과**: 11 passed / 0 failed / 0 skipped — 운영 ONNX 매핑 불일치 0건

### Phase 24. 하자 검수 메타 + 감사 로그 인프라 (R-v1.1.08, 2026-05-27)
- [x] `app/models/defect.py`: 컬럼 8개 추가 (review_status Enum, reviewed_by_user_id FK, reviewed_at, review_note, detection_model_id, gps_lat/lon/alt) + 인덱스 2개
- [x] `app/models/audit_log.py` 신규: AuditLog ORM (user_id/org_id FK SET NULL, action doted-name, resource_type/id, before/after JSONB, ip/UA/request_id) + 인덱스 4종
- [x] `app/services/audit_logger.py` 신규: write_audit() — 민감 키 재귀 redact, structlog request_id_ctx 자동 첨부, silent failure
- [x] `app/schemas/defect.py`: DefectLogResponse 6 필드 + DefectReviewRequest 신규
- [x] `app/schemas/audit_log.py` 신규: AuditLogResponse/ListResponse/Filter
- [x] `app/api/defects.py`: PATCH /defects/{id}/review + GET /defects/{id}/audit-trail + DELETE 에 audit.delete 첨부
- [x] `app/api/audit_logs.py` 신규: GET /audit-logs (admin/owner/superadmin, 조직 격리 + 5 필터 + 페이지네이션), GET /audit-logs/{id}
- [x] `app/api/router.py`: /audit-logs 등록 (tags=Audit)
- [x] alembic `n7b8c9d0e1f2_add_defect_review_and_audit_logs.py` (down=`m6a7b8c9d0e1`): defect_logs 8 컬럼 + audit_logs CREATE + FK 3 + 인덱스 6
- [x] **검증**: py_compile PASS, 라우터 등록 검증 (defects 8 + audit 2 routes)

### Phase 25. 운영 신뢰성 가이드 + PostgreSQL 백업 (R-v1.1.09, 2026-05-27)
- [x] `DEPLOYMENT_GUIDE.md` 신규 (분리 repo 최초): 10 섹션 — Fly secrets / alembic / 백업·복구 RTO·RPO / 콜드스타트 / Sentry / 감사 로그 운영 / 롤백 / CI·CD / 보안 체크리스트 / 장애 시나리오
- [x] `scripts/backup_pg.ps1` 신규: pg_dump -Fc + R2 업로드 + RETENTION_DAYS 자동 정리
- [x] `fly.toml`: min_machines_running 가이드 주석 (0 vs 1 트레이드오프)
- [ ] **운영자 후속 결정**: min_machines_running 0→1 변경 (비용 vs 가용성), 백업 cron 등록, Sentry DSN 발급 + flyctl secrets set

---

## Revision History

### v6.5_260527 (작성자: @youminsu0523 / branch: MS)
- **Phase 22~25 신설** — Sentry 통합(R-v1.1.06) / ONNX 4-way 매핑 회귀 가드(R-v1.1.07) / 하자 검수 메타 + 감사 로그 인프라(R-v1.1.08) / 운영 신뢰성 가이드 + PostgreSQL 백업(R-v1.1.09).
- 19 모델 → **21 모델** (AuditLog + DefectLog 컬럼 확장) , 12 alembic 리비전 → **13 리비전** (n7b8c9d0e1f2), 63+ 엔드포인트 → **66+** (audit-logs 2 + defects review/audit-trail 2). 운영 관점 "프로토타입 → 초기 운영" 전환 — 감사 추적 + 에러 모니터링 + 회귀 가드 + 백업·복구 인프라 동시 정착.

### v6.1_260515 (작성자: @youminsu0523 / branch: MS)
- **Phase 24 보강 — 챗봇 자동 제목 흐름 요약 (R-v1.1.05)** — `openai_chat.py` 의 임시 prefix 제목 부여 코드 제거(astream 의 첫 user 메시지 분기), BackgroundTask 호출 조건을 `user_count_before < 3` 으로 확장하여 1·2·3번째 응답마다 LLM 흐름 요약 제목 재생성. `regenerate_thread_title(thread_id)` 단일 인자로 단순화 — 내부에서 최근 10건 DB 조회 후 LLM 입력 구성. 프롬프트는 한국어 명사형 5~7단어, 하자 코드/현장명 키워드 포함, 단순 인사 시 일반 시작 제목 부여. `_is_first_user_message` → `_count_user_messages` 일반화. 마이그레이션 없음.

### v6.0_260515 (작성자: @youminsu0523 / branch: MS)
- **Phase 24 신설 — OpenAI 챗봇 백엔드 통합 (R-v1.1.01)** — 통합 repo 와 동일. 분리 repo head 가 `k4e5f6a7b8c9` 이므로 마이그레이션 `m6a7b8c9d0e1` down_revision 만 `k4e5f6a7b8c9` 로 분기.
  - DB 모델 2: AiChatThread / AiChatMessage. 멀티테넌트 user_id + organization_id 이중 격리. summary watermark 로 컨텍스트 압축.
  - 서비스 `OpenAIChatService` — SYSTEM_PROMPT(DEFECT_CATALOG 20종 표 + B영역 엄격 + 안전 직결 + 추측 금지 + 인젝션 거절). SSE 스트리밍. light-RAG(정규식 카테고리 코드 + 사이트 키워드, organization_id 필터). 30턴 초과 시 BackgroundTasks 자동 요약.
  - API `/api/v1/ai-chat` 6엔드포인트. `get_current_org_member` 의존성 + thread.user_id·org_id 이중 검증 + 사용자별 분당 20 메시지 라우터 내부 카운터.
  - settings 4(OPENAI_API_KEY/MODEL/MAX_OUTPUT_TOKENS/SUMMARY_MODEL), requirements `openai>=1.40.0`, rate_limit `/api/v1/ai-chat`:120/min.

### v5.1_260503 (작성자: @youminsu0523 / branch: MS)
- Phase 20 추가 완료(alembic 분기 head 병합 `89b53c16de85` + 누락 컬럼 10건 ALTER 보정 + seed_demo_data 실 적용 sites=8/defects=315/reports=12/schedules=3) + Phase 21 신설(tasks 문서 양식 정정 — 부록 → 인라인, 파일 rename, 팀명 일괄, 가이드 3종 문서이력 위치, CHANGES md 신설)

### v5.0_260503 (작성자: @youminsu0523 / branch: MS)
- Phase 18~20 신설: ML 후처리 파이프라인 (R19~R23) / Swagger·운영 가드 (R24) / Mockup→DB 시드 (R25)
- 18 모델 → 19 모델, 11 alembic 리비전 → 12 리비전, 60+ 엔드포인트 → 63+ (employee 3 추가)

### v4.0_260427 (작성자: @youminsu0523 / branch: MS)
- 전면 재작성: git log 기반 12 Phase 상세 기록
- 아키텍처 상세 (앱 초기화 흐름, 데이터 흐름, API 60+엔드포인트 전체 맵, 모듈 의존 관계, DB 스키마 18모델, AI 6모델)

### v1.0_260413 (작성자: @Antigravity / branch: main)
- 백엔드 구현 계획서 초기화
