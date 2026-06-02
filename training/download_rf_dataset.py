"""
Roboflow 데이터셋 다운로드 (rfenv py3.12 전용 — roboflow 패키지).
순차 래퍼가 학습 직전 호출. 다운로드 위치는 finetune_rf_cycle.CONFIGS[model]["rf_dir"].

실행: backend/rfenv/Scripts/python.exe download_rf_dataset.py <model> <workspace> <project> <version> <out_dir>
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from roboflow import Roboflow

KEY = "nuC9Lxr51Ds7c1IwN4Gy"

def main():
    if len(sys.argv) < 6:
        print("usage: download_rf_dataset.py <model> <ws> <proj> <ver> <out_dir>")
        return 1
    model, ws, proj, ver, out_dir = sys.argv[1:6]
    rf = Roboflow(api_key=KEY)
    p = rf.workspace(ws).project(proj)
    v = p.version(int(ver))
    ds = v.download("yolov8", location=out_dir)
    print(f"[download] {model} → {ds.location}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
