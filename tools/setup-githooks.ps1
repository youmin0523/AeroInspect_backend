# =============================================
# tools/setup-githooks.ps1
# 역할: Windows PowerShell 환경에서 git hook 활성화.
# 사용:
#   .\tools\setup-githooks.ps1                          # 현재 repo
#   .\tools\setup-githooks.ps1 -Target C:\path\to\AeroInspect_backend
# =============================================
param(
  [string]$Target = $null
)

$ErrorActionPreference = "Stop"

$Here = (Resolve-Path "$PSScriptRoot\..").Path
$SrcHooks = Join-Path $Here ".githooks"
if (-not $Target) { $Target = $Here }

if (-not (Test-Path $SrcHooks)) {
  Write-Error "source hooks 디렉터리를 찾을 수 없습니다: $SrcHooks"
  exit 1
}
if (-not (Test-Path (Join-Path $Target ".git"))) {
  Write-Error "git repo 가 아닙니다: $Target"
  exit 1
}

# 1) 분리 repo 라면 hook 복사
if ($Target -ne $Here) {
  $DstHooks = Join-Path $Target ".githooks"
  if (-not (Test-Path $DstHooks)) { New-Item -ItemType Directory -Path $DstHooks | Out-Null }
  Copy-Item -Force (Join-Path $SrcHooks "pre-commit") $DstHooks
  Copy-Item -Force (Join-Path $SrcHooks "commit-msg") $DstHooks
  Write-Host "[setup] hooks 복사 → $DstHooks"
}

# 2) core.hooksPath
Push-Location $Target
try {
  git config core.hooksPath .githooks
  Write-Host "[setup] core.hooksPath = .githooks  ($Target)"
  Write-Host "[setup] 활성 hook 확인:"
  git config --get core.hooksPath
  Get-ChildItem (Join-Path $Target ".githooks") | Select-Object Name, Length
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "✅ git hook 활성화 완료."
Write-Host "   pre-commit  : Vibe_Coding_Log.md 갱신 강제"
Write-Host "   commit-msg  : Conventional Commits 형식 강제"
Write-Host ""
Write-Host "💡 우회 (드물게만)"
Write-Host "   `$env:SKIP_VIBE_LOG_CHECK='1';  git commit ..."
Write-Host "   `$env:SKIP_COMMIT_MSG_CHECK='1'; git commit ..."
