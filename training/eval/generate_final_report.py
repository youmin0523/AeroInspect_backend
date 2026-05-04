"""
최종 종합 리포트 생성
- 모든 평가 결과 (max_boost, pt_tta, extreme_boost, wbf_*) 수집
- 모델별 best mAP 추출
- 0.85 도달 여부 판정
- COMPREHENSIVE_REPORT.md 갱신
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path(__file__).parent / "results"


def load_latest(pattern: str) -> dict | list | None:
    files = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files: return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    print("=== 최종 종합 리포트 생성 ===")

    max_boost = load_latest("max_boost_*.json") or []
    pt_tta = load_latest("pt_tta_*.json") or []
    extreme = load_latest("extreme_boost_*.json") or []
    wbf_m3 = load_latest("wbf_M3_*.json")
    wbf_m2 = load_latest("wbf_M2_*.json")
    wbf_m5 = load_latest("wbf_M5_*.json")

    # 모델별 모든 평가 결과 collect
    by_model = {}
    for r in max_boost:
        by_model.setdefault(r["key"], []).append({
            "method": "max_boost (ONNX multi-scale)",
            "mAP50": r["mAP50"], "P": r.get("P"), "R": r.get("R"),
            "imgsz": r.get("imgsz"), "tta": r.get("tta"),
        })
    for r in pt_tta:
        by_model.setdefault(r["key"], []).append({
            "method": "pt_tta (PT real TTA)",
            "mAP50": r["mAP50"], "P": r.get("P"), "R": r.get("R"),
            "imgsz": r.get("imgsz"), "tta": r.get("tta"),
        })
    for r in extreme:
        by_model.setdefault(r["key"], []).append({
            "method": "extreme_boost (PT grid)",
            "mAP50": r["mAP50"], "P": r.get("P"), "R": r.get("R"),
            "imgsz": r.get("imgsz"), "tta": r.get("tta"),
            "agnostic": r.get("agnostic"), "iou": r.get("iou"),
        })

    for wbf, mod_key in [(wbf_m3, "M3_YOLO"), (wbf_m2, "M2_YOLO"), (wbf_m5, "M5_SEG")]:
        if wbf:
            by_model.setdefault(mod_key, []).append({
                "method": f"WBF (multi-imgsz × TTA, {wbf['n_images']} images)",
                "mAP50": wbf["wbf"]["mAP50"], "P": wbf["wbf"]["P"], "R": wbf["wbf"]["R"],
                "imgsz_list": wbf["imgsz_list"],
            })
            by_model[mod_key].append({
                "method": f"WBF best single ({wbf['best_single']['config']})",
                "mAP50": wbf["best_single"]["mAP50"],
                "P": wbf["best_single"]["P"], "R": wbf["best_single"]["R"],
            })

    # Best per model
    best = {}
    for k, results in by_model.items():
        best_r = max(results, key=lambda x: x.get("mAP50", -1))
        best[k] = best_r

    # Markdown report
    lines = [
        f"# 모델별 0.85 도달 종합 리포트",
        f"\n**생성**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 모델별 BEST mAP50 (모든 평가 방법 통합)\n",
        "| 모델 | best mAP50 | P | R | 방법 | 0.85 갭 | 상태 |",
        "|------|------------|---|---|------|---------|------|",
    ]
    targets = ["M1_YOLO", "M2_YOLO", "M3_YOLO", "M4_CONTEXT", "M5_SEG", "furniture_aware"]
    for k in targets:
        if k not in best:
            lines.append(f"| {k} | - | - | - | (no data) | - | ❓ |")
            continue
        b = best[k]
        m = b.get("mAP50", 0)
        p = b.get("P") or 0
        r = b.get("R") or 0
        gap = m - 0.85
        gap_s = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        status = "🎯 도달" if m >= 0.85 else "❌ 미달"
        lines.append(f"| {k} | **{m:.4f}** | {p:.4f} | {r:.4f} | {b['method']} | {gap_s} | {status} |")

    lines.append("\n## 모델별 전체 평가 이력\n")
    for k in targets:
        lines.append(f"\n### {k}\n")
        if k not in by_model:
            lines.append("- (no data)\n")
            continue
        sorted_r = sorted(by_model[k], key=lambda x: -x.get("mAP50", -1))
        lines.append("| 방법 | mAP50 | P | R | 추가 정보 |")
        lines.append("|------|-------|---|---|----------|")
        for r in sorted_r:
            extra = []
            if r.get("imgsz"): extra.append(f"imgsz={r['imgsz']}")
            if r.get("tta") is not None: extra.append(f"tta={'O' if r['tta'] else 'X'}")
            if r.get("agnostic") is not None: extra.append(f"ag={'O' if r['agnostic'] else 'X'}")
            if r.get("iou"): extra.append(f"iou={r['iou']}")
            extra_s = ", ".join(extra)
            p_v = r.get("P") or 0
            r_v = r.get("R") or 0
            lines.append(f"| {r['method']} | {r['mAP50']:.4f} | {p_v:.4f} | {r_v:.4f} | {extra_s} |")

    lines.append("\n## 0.85 미달 모델 — v1.1 재학습 필요 항목\n")
    for k in targets:
        if k in best and best[k].get("mAP50", 0) < 0.85:
            gap = 0.85 - best[k]["mAP50"]
            lines.append(f"- **{k}**: 현재 {best[k]['mAP50']:.4f}, 0.85까지 +{gap:.4f} 필요. 재학습 권장.")

    out = RESULTS / f"FINAL_REPORT_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트 저장: {out}")
    print(f"\n=== 모델별 BEST mAP50 ===")
    for k in targets:
        if k in best:
            m = best[k]['mAP50']
            mark = "🎯" if m >= 0.85 else "❌"
            print(f"  {mark} {k}: {m:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
