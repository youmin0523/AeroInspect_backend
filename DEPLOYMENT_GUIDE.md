# AeroInspect 운영 배포 가이드

상업 운영용 신뢰성 보강 가이드입니다. 입주자/사용자 피해 직결 안전 도메인이라 모든 단계가 책임 추적 가능해야 합니다.

---

## 1. 환경별 시크릿 등록 (Fly secrets)

```powershell
# 운영 진입 시 1회만. 키 회전 시 동일 명령으로 덮어쓰기.
flyctl secrets set `
  JWT_SECRET=<32+chars random> `
  JWT_REFRESH_EXPIRE_DAYS=14 `
  AI_WEBHOOK_SECRET=<32+chars random> `
  OPENAI_API_KEY=<sk-...> `
  OPENAI_MODEL=gpt-4o-mini-search-preview `
  SENTRY_DSN=<https://...sentry.io/...> `
  SENTRY_ENVIRONMENT=production `
  SENTRY_TRACES_SAMPLE_RATE=0.1 `
  APP_ENV=production `
  LOG_JSON=true `
  LOG_LEVEL=info `
  PUSH_PROVIDER=noop `
  WS_BACKEND=memory `
  --app aeroinspect-backend
```

### 검증
```powershell
flyctl secrets list --app aeroinspect-backend  # 키 이름만 노출, 값은 노출 X
flyctl ssh console --app aeroinspect-backend -C "python -c 'from app.config import settings; print(settings.APP_ENV)'"
```

APP_ENV=production 일 때 placeholder 시크릿(`change-me-in-production` 등)이 남아있으면 기동 자체가 차단됩니다 — [app/config.py](app/config.py) `enforce_no_placeholder_secrets_in_prod()`.

---

## 2. 마이그레이션 (alembic)

매 배포마다 자동 실행되지는 않습니다 — 스키마 변경이 있는 배포만 SSH 로 1회 실행.

```powershell
flyctl ssh console --app aeroinspect-backend
# 머신 내부에서:
alembic current      # 현재 적용된 head 확인
alembic heads        # 코드에 정의된 head 확인 (둘이 같아야 정상)
alembic upgrade head # 차이가 있으면 적용
```

**현재 head**: `n7b8c9d0e1f2` (defect review 메타 + audit_logs)
**이전 head**: `m6a7b8c9d0e1` (ai_chat 테이블)

### 신규 컬럼/테이블 추가 시 절차
1. 로컬에서 모델 수정 → `alembic revision --autogenerate -m "..."` (수동 검토 필수, autogenerate 가 빠뜨릴 수 있음)
2. PR 머지 → `flyctl deploy`
3. SSH 로 `alembic upgrade head`
4. `/health` 200 확인

### 롤백 (스키마)
```powershell
alembic downgrade -1   # 직전 리비전으로
```
다만 데이터 손실 위험 있는 downgrade 는 신중. `audit_logs` 테이블은 DROP 시 모든 감사 기록이 사라지므로 prod 에서 downgrade 금지.

---

## 3. PostgreSQL 백업 정책

### Fly Postgres 자동 스냅샷 (기본)
Fly.io 가 볼륨 단위로 자동 일일 스냅샷을 보관합니다 (기본 7일).
```powershell
flyctl postgres list
flyctl postgres backup list --app <pg-app-name>     # 보관 중 백업 목록
flyctl postgres backup restore <backup-id> --app <pg-app-name>
```

### 추가 외부 백업 (분쟁 대응용 장기 보관)
PR 분쟁 / 입주자 클레임은 수개월 후 발생할 수 있어 7일 보관으로 부족합니다.
[scripts/backup_pg.ps1](scripts/backup_pg.ps1) 또는 `scripts/backup_pg.sh` 로 주기적 pg_dump → Cloudflare R2 또는 로컬 보관.

#### 일일 스케줄 (Task Scheduler / cron)
- **Windows**: `schtasks /create /SC DAILY /ST 03:00 /TN "aeroinspect-pg-backup" /TR "powershell -File C:\path\scripts\backup_pg.ps1"`
- **GitHub Actions cron** (권장 — 머신 의존성 없음): `.github/workflows/backup-pg.yml` 일일 03:00 UTC 실행, dump → R2 업로드

#### 복구 절차
1. R2 에서 `aeroinspect-pg-YYYYMMDD.dump` 다운로드
2. `pg_restore -h <host> -U <user> -d <new_db_name> -v aeroinspect-pg-YYYYMMDD.dump`
3. 또는 신규 Fly Postgres 에 복원 후 `DATABASE_URL` swap

### RTO/RPO 목표
| 지표 | 목표 | 현재 |
|------|------|------|
| RPO (데이터 손실 허용) | 24h | 24h (일일 백업) |
| RTO (복구 시간) | 4h | 1~2h (Fly snapshot) / 4h (R2 → 신규 Postgres) |

상업 도입 후 클레임 빈도 따라 RPO 1h (시간별 백업) 로 단축 검토.

---

## 4. 콜드스타트 (가용성)

`fly.toml` 의 `min_machines_running` 설정:
- **0** (현재): 무료 tier 친화, 그러나 첫 요청 수초 지연 → **드론 라이브 스트림 첫 연결 시 끊김 위험**
- **1** (상업 운영 권장): 한 머신 24/7 가동, 콜드스타트 0. 비용 증가 발생 (Fly 머신 시간당 요금).

변경 절차:
```powershell
# fly.toml 의 min_machines_running 1 로 수정 후
flyctl deploy --app aeroinspect-backend
```

### 단일 region (nrt) 운영
현재 도쿄 단일 region. failover 미설정. multi-region 은 비용·복잡도 증가라 v1.2 이후 검토.
국내 사용자 응답 시간 80~120ms 수준 (실측 필요).

---

## 5. 에러 모니터링 (Sentry)

### DSN 발급
1. Sentry 가입 → Project 생성 (Python FastAPI + React 별도 프로젝트 권장)
2. 각 프로젝트의 DSN 복사
3. Fly secrets / Vercel envs 등록:
   ```powershell
   flyctl secrets set SENTRY_DSN=https://...@sentry.io/... --app aeroinspect-backend
   ```
   Vercel: Dashboard → Settings → Environment Variables → `VITE_SENTRY_DSN`

### 알림 룰
Sentry 콘솔에서:
- High severity 즉시 슬랙/이메일
- 5분간 동일 에러 10건 → 묶음 알림
- p95 응답 시간 5초 초과 → 알림

### 통합 검증
배포 직후 `/health` 호출 → Sentry 에 transaction 1건 기록되는지 확인.
구조화 로그(structlog)의 `request_id` 가 Sentry tag 로 자동 첨부됨 — 로그 ↔ 에러 양방향 추적 가능.

---

## 6. 감사 로그 (Audit) 운영

`audit_logs` 테이블에 다음 사건이 자동 기록됩니다:
- `defect.review.approve / reject / flag_false_positive / reset`
- `defect.delete`
- (향후 추가) `report.publish / update / delete`, `site.update`, `org.member.role_change`

### 보관 정책
- 기본: 전체 보관 (DELETE 금지)
- 1년 경과분은 별도 archive 테이블로 이전 검토 (v1.2)

### 조회 권한
- `GET /api/v1/audit-logs` — admin/owner/superadmin 만
- 일반 사용자: 본인이 검수한 하자의 `GET /defects/{id}/audit-trail` 접근 가능

### 분쟁 대응 시 추출
```powershell
flyctl ssh console -C "python -c 'from app.db.session import async_session_factory; ...'"
# 또는 직접 SQL:
psql $DATABASE_URL -c "COPY (SELECT * FROM audit_logs WHERE created_at >= '2026-04-01' AND resource_type='defect') TO STDOUT WITH CSV HEADER" > audit_dispute.csv
```

---

## 7. 롤백 (배포)

### 코드 롤백
```powershell
flyctl releases --app aeroinspect-backend           # 최근 릴리스 목록
flyctl releases rollback <version> --app aeroinspect-backend
```

### 스키마 + 코드 동시 롤백
1. 코드 롤백 (위)
2. SSH → `alembic downgrade <prev-revision>`
3. `/health` 확인

### Blue-Green 또는 canary
현재 `rolling` 전략. 트래픽 분할 canary 미설정 (v1.2 검토).

---

## 8. CI/CD 자동화 현황

`.github/workflows/fly-deploy.yml` 존재 — main 브랜치 푸시 시 자동 deploy.

### 추가 권장 워크플로우
- `test.yml` — PR 시 pytest 자동 실행 (현재 51개 테스트)
- `frontend-test.yml` — frontend 빌드 + (향후) Vitest
- `backup-pg.yml` — 일일 03:00 UTC pg_dump → R2

---

## 9. 보안 체크리스트 (배포 전)

- [ ] `flyctl secrets list` 에 모든 필수 키 등록 (JWT_SECRET, AI_WEBHOOK_SECRET, OPENAI_API_KEY, SENTRY_DSN, DATABASE_URL)
- [ ] `.env` 파일이 git 추적되지 않음 (`git ls-files | grep .env` 비어있어야)
- [ ] APP_ENV=production 에서 기동 성공 (placeholder secret 차단 검증)
- [ ] CORS allowlist 에 vercel 도메인만 (개발 localhost 제거)
- [ ] rate_limit.py 가 모든 인증 엔드포인트에 적용됨
- [ ] HTTPS 강제 (`force_https = true`)
- [ ] DB 백업 직전 1회 실행 + 복원 dry-run

---

## 10. 장애 시나리오 대응

| 장애 | 즉시 조치 | 영구 조치 |
|------|---------|---------|
| API 5xx 폭증 | Sentry 알림 확인 → `flyctl logs` → 이전 릴리스 롤백 | 회귀 테스트 추가 |
| DB 응답 없음 | `flyctl postgres status` → 최신 스냅샷 복원 | 백업 주기 단축 |
| AI 추론 서버 다운 | `/health` 의 `inference_pipeline` 상태 확인 → GCP GPU VM 재시작 | 추론 서버 redundancy |
| 인증 토큰 유출 의심 | JWT_SECRET 회전 → 모든 사용자 재로그인 | 의심 IP 차단, audit_logs 조사 |

---

**문서 버전**: v1.1 (2026-06-01)

**v1.1 변경 (R-v1.1.10~17)**:
- grade 시스템 (CONFIRMED/REVIEW/REFERENCE) 도입 — 보고서 등재 기준 강화
- M4 Context bbox→seg 전환, mAP50-95 0.355→0.503 (+41.7%)
- Thermal Anomaly PatchCore (Moisture/delam 대체), `THERMAL_ANOMALY_ENABLED=False` 보류 토글
- refresh token rotation (auth/refresh 응답에 새 refresh_token 발급)
- CORS_ORIGINS에 Vercel 도메인 3개 추가
- .env.example 신규: `THERMAL_ANOMALY_ENABLED`, `R2_*` 6개
- API 응답 정보 누출 수정 (detect.py `detail=str(e)` → 일반 메시지)

**다음 갱신 트리거**: 신규 환경변수 추가, 신규 마이그레이션, RTO/RPO 변경, 신규 ML 모델 통합
