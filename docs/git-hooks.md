# Git Hooks — AeroInspect

본 repo (및 분리 repo `AeroInspect_backend` / `AeroInspect_frontend`) 는 두 개의 git hook 으로 commit 품질을 강제합니다.

## 동작

### 1. `pre-commit` — Vibe_Coding_Log.md 갱신 강제
- 코드 변경이 stage 되어 있으면 **Vibe_Coding_Log.md 변경**도 함께 stage 되어야 한다
- 통합 repo: `backend/Vibe_Coding_Log.md` 또는 `frontend/Vibe_Coding_Log.md`
- 분리 repo: 루트의 `Vibe_Coding_Log.md`
- 누락 시 commit 차단 + 가이드 메시지

### 2. `commit-msg` — Conventional Commits 강제
- 형식: `<type>(<scope>): <description>`
- 허용 type: `feat fix docs style refactor test chore perf ci build release hotfix`
- `[type]:` 대괄호 형태도 허용
- 메시지 길이 ≥ 10자
- Merge / Revert / fixup / squash 통과
- 누락 시 commit 차단 + 한국어 가이드

## 활성화

### 통합 repo (현재 워크스페이스)
```bash
bash tools/setup-githooks.sh
```
또는 PowerShell:
```powershell
.\tools\setup-githooks.ps1
```

### 분리 repo (`AeroInspect_backend`, `AeroInspect_frontend`)

분리 repo 가 로컬에 클론되어 있으면 통합 repo 의 `.githooks/` 를 한 번에 배포할 수 있습니다.

```bash
# bash (Linux / macOS / WSL / Git Bash)
bash tools/setup-githooks.sh /path/to/AeroInspect_backend
bash tools/setup-githooks.sh /path/to/AeroInspect_frontend
```
```powershell
# PowerShell (Windows)
.\tools\setup-githooks.ps1 -Target C:\path\to\AeroInspect_backend
.\tools\setup-githooks.ps1 -Target C:\path\to\AeroInspect_frontend
```

스크립트가 자동으로 수행:
1. `<target>/.githooks/{pre-commit,commit-msg}` 복사
2. 실행 권한 부여 (Linux 계열)
3. `git -C <target> config core.hooksPath .githooks`

## 검증

활성 여부:
```bash
git config --get core.hooksPath        # → .githooks 가 출력돼야 함
ls -la .githooks                        # pre-commit, commit-msg 두 개
```

## 우회 (긴급용 — 드물게만)

```bash
SKIP_VIBE_LOG_CHECK=1   git commit -m "..."   # Vibe 로그 검사 우회
SKIP_COMMIT_MSG_CHECK=1 git commit -m "..."   # commit-msg 검사 우회
```

PowerShell:
```powershell
$env:SKIP_VIBE_LOG_CHECK='1';   git commit -m "..."
$env:SKIP_COMMIT_MSG_CHECK='1'; git commit -m "..."
```

> 우회 후엔 사후에 Vibe 로그 보강 commit 또는 amend 권장.

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `core.hooksPath` 가 비어 있음 | setup 스크립트 미실행 | `bash tools/setup-githooks.sh` |
| Linux/macOS 에서 `permission denied` | 실행 권한 없음 | `chmod +x .githooks/*` |
| Windows Git Bash 에서 hook 안 돌아감 | 줄바꿈 CRLF | `.gitattributes` 에 `*.sh text eol=lf` 또는 setup 스크립트 재실행 |
| 분리 repo 에서 commit 차단 | hook 미적용 | `bash tools/setup-githooks.sh /path/to/repo` |
