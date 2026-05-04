# =============================================
# scripts/sweep_wallpaper_thresholds.py
# 역할: 벽지 분류기 이중 게이트 임계값 스윕
#       - (WALLPAPER_CONF_THRESHOLD, WALLPAPER_MARGIN_THRESHOLD) 격자 탐색
#       - JSONL 운영 로그 또는 사람이 태깅한 평가셋 기반
#       - precision / recall / F1 / FP per HIGH confidence frame 출력
#
# 입력 포맷 (JSONL 한 줄 = 한 프레임/레코드):
#   {"top1_conf": 0.62, "top2_conf": 0.41, "label": "defect" | "normal"}
#   (label 은 사람이 오탐/미탐 태깅한 ground truth)
#
# 사용:
#   python scripts/sweep_wallpaper_thresholds.py --input ops_logs.jsonl
#   python scripts/sweep_wallpaper_thresholds.py --input eval.jsonl --out sweep.csv
#
# 결과 해석:
#   - "감"으로 고정한 0.35 / 0.15 주변부터 F1 등고선 확인
#   - 건물 검사 특성상 recall 우선 → F1 대신 recall@precision≥0.7 기준도 참고
# =============================================

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_records(path: Path) -> Iterable[dict]:
    """JSONL 스트리밍 파서. 깨진 라인은 skip + stderr 경고."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] L{i} JSON 파싱 실패: {e}", file=sys.stderr)


def classify(top1: float, top2: float, conf_t: float, margin_t: float) -> bool:
    """이중 게이트 — 탐지로 판정하면 True."""
    return top1 >= conf_t and (top1 - top2) >= margin_t


def confusion(records: List[dict], conf_t: float, margin_t: float) -> Tuple[int, int, int, int]:
    """(TP, FP, FN, TN). label ∈ {defect, normal}."""
    tp = fp = fn = tn = 0
    for r in records:
        pred = classify(float(r["top1_conf"]), float(r["top2_conf"]), conf_t, margin_t)
        actual = r["label"] == "defect"
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="JSONL 평가셋 경로")
    ap.add_argument("--out", type=Path, default=None, help="결과 CSV 저장 경로 (옵션)")
    ap.add_argument("--conf-grid", default="0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--margin-grid", default="0.05,0.10,0.15,0.20,0.25")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[error] 입력 파일 없음: {args.input}", file=sys.stderr)
        sys.exit(2)

    records = [
        r for r in iter_records(args.input)
        if "top1_conf" in r and "top2_conf" in r and r.get("label") in ("defect", "normal")
    ]
    if not records:
        print("[error] 유효 레코드 0건", file=sys.stderr)
        sys.exit(2)
    print(f"[info] 유효 레코드 {len(records)}건 로드")

    conf_grid = [float(x) for x in args.conf_grid.split(",")]
    margin_grid = [float(x) for x in args.margin_grid.split(",")]

    rows: List[dict] = []
    for c in conf_grid:
        for m in margin_grid:
            tp, fp, fn, tn = confusion(records, c, m)
            row = {"conf": c, "margin": m, **metrics(tp, fp, fn, tn)}
            rows.append(row)

    # 정렬: F1 내림차순, 동점이면 recall 우선
    rows.sort(key=lambda r: (-r["f1"], -r["recall"]))

    header = ["conf", "margin", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    # 상위 15개 콘솔 출력
    col_w = {k: max(len(k), max(len(f"{r[k]}") for r in rows)) for k in header}
    print(" | ".join(k.ljust(col_w[k]) for k in header))
    print("-+-".join("-" * col_w[k] for k in header))
    for r in rows[:15]:
        print(" | ".join(f"{r[k]}".ljust(col_w[k]) for k in header))

    if args.out:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[info] CSV 저장: {args.out}")


if __name__ == "__main__":
    main()
