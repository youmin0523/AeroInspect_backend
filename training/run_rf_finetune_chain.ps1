# Roboflow fine-tune 순차 체인 (다운로드 rfenv → 학습 venv → 다음 모델)
# 사용자 지시(2026-06-01): 전 모델 Roboflow 데이터 fine-tune, 로컬 순차, 다운->학습->삭제.
# THERMAL은 이미 별도 실행 중일 수 있음 -> 이 체인은 M2,M3,M4 순차.
# 실행: powershell -ExecutionPolicy Bypass -File run_rf_finetune_chain.ps1

$ErrorActionPreference = "Continue"
$root = "C:\Users\Codelab\Desktop\PROJECT\TEAM_PROJECT_2_Drone_project"
$bt = "$root\backend\training"
$rfpy = "$root\backend\rfenv\Scripts\python.exe"
$vpy = "$root\backend\venv\Scripts\python.exe"

# 모델 = workspace, project, version, out_dir 이름
$models = @(
  @{ m="M2"; ws="builddef2"; proj="building-defect-on-walls"; ver=4; dir="m2_builddef_v4" },
  @{ m="M3"; ws="roboflow-100"; proj="glass-defect-detection-fvbcu"; ver=2; dir="m3_glass_v2" },
  @{ m="M4"; ws="wall-detection"; proj="wall-ceiling-floor-m6bao"; ver=1; dir="m4_wcf_v1" }
)

foreach ($mo in $models) {
  $tag = $mo.m
  $outdir = "$bt\rf_downloads\$($mo.dir)"
  Write-Output "===== [$tag] download ====="
  & $rfpy "$bt\download_rf_dataset.py" $tag $mo.ws $mo.proj $mo.ver $outdir 2>&1 | Select-Object -Last 3
  if (-not (Test-Path "$outdir\data.yaml")) {
    Write-Output "[$tag] DOWNLOAD FAIL - skip"
    continue
  }
  Write-Output "===== [$tag] fine-tune ====="
  & $vpy "$bt\finetune_rf_cycle.py" --model $tag 2>&1 | Select-Object -Last 8
  Write-Output "===== [$tag] done ====="
}
Write-Output "ALL_RF_FINETUNE_CHAIN_DONE"
