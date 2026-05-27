# =============================================
# scripts/backup_pg.ps1
# 역할: PostgreSQL 백업 → 로컬 dump + (선택) Cloudflare R2 업로드
# 사용: pwsh ./scripts/backup_pg.ps1
#       또는 Task Scheduler 일일 03:00 등록
# 의존: pg_dump (PostgreSQL client tools), aws-cli (R2 업로드 시)
# 환경변수:
#   DATABASE_URL          - postgres://user:pass@host:5432/db (필수)
#   BACKUP_DIR            - 로컬 저장 경로 (기본: ./backups)
#   R2_BUCKET             - Cloudflare R2 버킷 이름 (선택, 없으면 로컬만)
#   R2_ENDPOINT_URL       - https://<account>.r2.cloudflarestorage.com
#   AWS_ACCESS_KEY_ID     - R2 API token 의 access key
#   AWS_SECRET_ACCESS_KEY - R2 API token 의 secret
#   RETENTION_DAYS        - 로컬 보관 일수 (기본: 30)
# =============================================

$ErrorActionPreference = "Stop"

$DatabaseUrl = $env:DATABASE_URL
if (-not $DatabaseUrl) {
    Write-Error "DATABASE_URL 환경변수가 필요합니다."
    exit 1
}

$BackupDir = if ($env:BACKUP_DIR) { $env:BACKUP_DIR } else { "./backups" }
$RetentionDays = if ($env:RETENTION_DAYS) { [int]$env:RETENTION_DAYS } else { 30 }

if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$DumpFile = Join-Path $BackupDir "aeroinspect-pg-$Timestamp.dump"

Write-Host "[backup_pg] pg_dump 시작 -> $DumpFile"
# -Fc: custom format (압축 + pg_restore 호환), -Z 9: 최고 압축
& pg_dump --dbname=$DatabaseUrl --format=custom --compress=9 --file=$DumpFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "pg_dump 실패 (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

$DumpSize = (Get-Item $DumpFile).Length
Write-Host "[backup_pg] dump 완료: $([math]::Round($DumpSize/1MB, 2)) MB"

# R2 업로드 (옵션)
if ($env:R2_BUCKET -and $env:R2_ENDPOINT_URL) {
    $R2Key = "pg-backups/aeroinspect-pg-$Timestamp.dump"
    Write-Host "[backup_pg] R2 업로드 -> s3://$($env:R2_BUCKET)/$R2Key"
    & aws s3 cp $DumpFile "s3://$($env:R2_BUCKET)/$R2Key" --endpoint-url $env:R2_ENDPOINT_URL
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "R2 업로드 실패 — 로컬 백업은 유효합니다."
    } else {
        Write-Host "[backup_pg] R2 업로드 성공"
    }
} else {
    Write-Host "[backup_pg] R2 환경변수 미설정 — 로컬 보관만 진행"
}

# 로컬 보관 정책 (RETENTION_DAYS 초과분 삭제)
$Cutoff = (Get-Date).AddDays(-$RetentionDays)
$OldFiles = Get-ChildItem -Path $BackupDir -Filter "aeroinspect-pg-*.dump" |
    Where-Object { $_.LastWriteTime -lt $Cutoff }

if ($OldFiles) {
    Write-Host "[backup_pg] 보관 정책 ($RetentionDays 일) 초과 파일 정리"
    foreach ($f in $OldFiles) {
        Write-Host "  삭제: $($f.Name) ($([math]::Round($f.Length/1MB, 2)) MB)"
        Remove-Item $f.FullName
    }
}

Write-Host "[backup_pg] 완료. 백업 파일: $DumpFile"
