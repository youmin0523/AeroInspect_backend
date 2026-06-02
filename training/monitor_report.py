# =============================================
# monitor_report.py
# v1.1 체이닝 학습 진행 평가 리포트 (Monitor가 5분마다 호출)
# epoch 진행률 / 잔여 epoch / 예상 잔여시간 / 현재·best mAP /
# baseline 대비 개선 / 목표(0.9) 대비 / 추세 / GPU·CPU·RAM
# =============================================

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TRAIN = Path(__file__).resolve().parent
RUNS = TRAIN / "runs"                       # 체이닝 상태 파일 위치
STATUS = RUNS / "chain_status.txt"
# ultralytics 글로벌 runs_dir = <repo>/runs/detect → project 출력은 runs/detect/runs/ 아래
RUNS_DETECT = TRAIN.parent.parent / "runs" / "detect" / "runs"
TARGET = 0.90  # 상업 grade mAP50-95 목표

# 모델명 → (baseline mAP50-95, 총 epoch, results.csv 상대경로)
# Thermal baseline 0.299는 Crack 포함 3cls 옛값 — 현재 2cls(Moisture/delam)는 더 높게 나올 것
META = {
    "Thermal": (0.299, 120, "thermal_v11/results.csv"),
    "M5": (0.466, 150, "m5_frame_seg/seg_v2/results.csv"),
    "M4": (0.355, 50, "m4_context/train/results.csv"),
    "M4_Seg": (0.355, 60, "m4_context_seg/train/results.csv"),
    "furniture": (None, 80, "furniture_aware_v2/results.csv"),
    "Furniture": (None, 80, "furniture_aware_v2/results.csv"),
    "ThermalAnomaly": (None, 1, "thermal_anomaly/results.csv"),
}


def current_stage() -> str:
    try:
        return STATUS.read_text(encoding="utf-8").strip()
    except Exception:
        return "(상태 파일 없음)"


def stage_key(stage: str) -> str:
    # 긴 키 우선 매칭 (M4_Seg가 M4보다 먼저, Furniture가 furniture 동등)
    for k in sorted(META.keys(), key=len, reverse=True):
        if k.lower() in stage.lower():
            return k
    return ""


def results_csv_for(key: str) -> str | None:
    if key not in META:
        return None
    rel = META[key][2]
    # seg 모델(M5/M4_Seg)은 runs/segment/, detection은 runs/detect/ 에 저장
    if key in ("M5", "M4_Seg"):
        path = TRAIN.parent.parent / "runs" / "segment" / "runs" / rel
    else:
        path = RUNS_DETECT / rel
    return str(path) if path.exists() else None


def col(row: dict, *names):
    # 공백 포함 컬럼명 대응
    keys = {k.strip(): k for k in row.keys()}
    for n in names:
        if n in keys:
            return row[keys[n]]
    return None


def parse_results(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    cur_epoch = int(float(col(last, "epoch") or 0))
    cur_map = float(col(last, "metrics/mAP50-95(B)", "metrics/mAP50-95(M)") or 0)
    cur_map50 = float(col(last, "metrics/mAP50(B)", "metrics/mAP50(M)") or 0)
    maps = [float(col(r, "metrics/mAP50-95(B)", "metrics/mAP50-95(M)") or 0) for r in rows]
    best_map = max(maps) if maps else 0.0
    total_time = float(col(last, "time") or 0)
    per_epoch = total_time / max(cur_epoch, 1)
    # 최근 3 epoch 추세
    trend = "—"
    if len(maps) >= 4:
        recent = maps[-3:]
        if recent[-1] > recent[0] + 0.005:
            trend = "상승"
        elif recent[-1] < recent[0] - 0.005:
            trend = "하락"
        else:
            trend = "정체"
    return cur_epoch, cur_map, cur_map50, best_map, per_epoch, trend


def gpu_cpu_ram():
    gpu = "?"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10
        ).splitlines()[0].strip()
    except Exception:
        pass
    cpu = ram = "?"
    try:
        cpu = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
            text=True, timeout=15).strip()
        ram = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem; [math]::Round(($o.TotalVisibleMemorySize-$o.FreePhysicalMemory)/1MB,1)"],
            text=True, timeout=15).strip()
    except Exception:
        pass
    return gpu, cpu, ram


def main():
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    forced = sys.argv[1] if len(sys.argv) > 1 else None
    if forced:
        stage = f"[단일] {forced} 재학습"
        key = forced
    else:
        stage = current_stage()
        key = stage_key(stage)
    gpu, cpu, ram = gpu_cpu_ram()

    lines = [f"[{ts}] 단계: {stage}"]

    if "전체 완료" in stage:
        lines.append("  ✅ 체이닝 전체 완료")
        print("\n".join(lines), flush=True)
        return

    csv_path = results_csv_for(key)
    parsed = parse_results(csv_path) if csv_path else None

    if parsed and key:
        cur_e, cur_m, cur_m50, best_m, per_e, trend = parsed
        baseline, total_e, _ = META[key]
        remain_e = max(total_e - cur_e, 0)
        eta = timedelta(seconds=int(per_e * remain_e))
        finish = (datetime.now() + eta).strftime("%m-%d %H:%M")
        lines.append(f"  진행: epoch {cur_e}/{total_e} (잔여 {remain_e}, ~{eta} → 완료예상 {finish} / {per_e:.0f}s per epoch)")
        lines.append(f"  현재 mAP50-95 {cur_m:.3f} (mAP50 {cur_m50:.3f}) / best {best_m:.3f} | 추세 {trend}")
        if baseline is not None:
            diff = best_m - baseline
            sign = "+" if diff >= 0 else ""
            verdict = "개선중" if diff > 0.005 else ("baseline 회복 전" if diff < -0.005 else "baseline 수준")
            lines.append(f"  baseline {baseline:.3f} → {sign}{diff:.3f} ({verdict}) | 목표 {TARGET} 까지 {TARGET-best_m:.3f}")
        else:
            lines.append(f"  baseline 미측정(신규) | 목표 {TARGET} 까지 {TARGET-best_m:.3f}")
    else:
        lines.append("  진행: 첫 epoch 대기 중 (데이터 로딩/검증) — results.csv 아직 없음")

    lines.append(f"  자원: CPU {cpu}% / RAM {ram}GB used / GPU(used/free/util/temp) {gpu}")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
