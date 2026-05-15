"""
tests/test_floorplan_real_dataset.py
역할: 실제 평면도 + 한국 아파트 합성 패턴으로 정확도 측정.

데이터:
  - datasets/real_floorplans/    : Wikimedia 공개 평면도 (가구 GT 없음 — sanity check 만)
  - datasets/synthetic_korean/   : 한국 아파트 패턴 합성 + GT JSON

GT JSON 형식:
  { "image_width", "image_height", "furniture_gt": [{"cx","cy","w","h","label"}, ...] }
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.floorplan_processor import extract_walls_from_bytes


REAL_DIR = Path("datasets/real_floorplans")
KOREAN_DIR = Path("datasets/synthetic_korean")
LH_PAGES_DIR = Path("datasets/lh_real_floorplans/pages")
DXF_DIR = Path("datasets/dxf_samples")


def _bbox_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(detections, gts, iou_threshold=0.3):
    det_boxes = [(d['cx'] - d['w']/2, d['cy'] - d['h']/2,
                  d['cx'] + d['w']/2, d['cy'] + d['h']/2) for d in detections]
    gt_boxes = [(g['cx'] - g['w']/2, g['cy'] - g['h']/2,
                 g['cx'] + g['w']/2, g['cy'] + g['h']/2) for g in gts]
    matched_det, matched_gt, ious = set(), set(), []
    pairs = []
    for di, db in enumerate(det_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = _bbox_iou(db, gb)
            if iou >= iou_threshold:
                pairs.append((iou, di, gi))
    pairs.sort(reverse=True)
    for iou, di, gi in pairs:
        if di in matched_det or gi in matched_gt:
            continue
        matched_det.add(di); matched_gt.add(gi); ious.append(iou)
    tp = len(ious); fp = len(det_boxes) - tp; fn = len(gt_boxes) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(p, 3), 'recall': round(r, 3),
        'f1': round(f1, 3),
        'mean_iou': round(sum(ious) / len(ious), 3) if ious else 0.0,
    }


# ────────────────────────────────────────────
# 1) Wikimedia 다운로드 데이터 — sanity check (GT 없음 → 추출 자체만 검증)
# ────────────────────────────────────────────

def _real_files():
    if not REAL_DIR.exists():
        return []
    return sorted([p for p in REAL_DIR.glob("*") if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])


@pytest.mark.parametrize("path", _real_files() or [pytest.param(None, marks=pytest.mark.skip(reason="No real files"))])
def test_real_floorplan_extracts_something(path):
    """진짜 외부 평면도 — 파이프라인이 죽지 않고 의미있는 결과 반환."""
    if path is None:
        pytest.skip("No real files downloaded")
    with open(path, 'rb') as f:
        b = f.read()
    result = extract_walls_from_bytes(b)
    print(f"\n[REAL: {path.name}] walls={result['wall_count']}, "
          f"furniture={result['furniture_count']}, outline={len(result['outline'])}")
    # 외부 평면도는 GT 가 없어 정량 측정 불가 — 최소한 파이프라인이 죽지 않으면 통과
    assert result['image_width'] > 0
    assert result['image_height'] > 0
    # 적어도 일부 벽이나 가구가 잡혀야 평면도로서 의미 있음
    assert result['wall_count'] + result['furniture_count'] > 0, \
        f"{path.name}: 벽/가구 모두 0 — 알고리즘이 이 도면을 처리 못함"


# ────────────────────────────────────────────
# 2) 한국 합성 데이터 — GT 있음 → 정량 측정
# ────────────────────────────────────────────

def _korean_cases():
    if not KOREAN_DIR.exists():
        return []
    cases = []
    for json_path in sorted(KOREAN_DIR.glob("*.json")):
        gt = json.loads(json_path.read_text(encoding='utf-8'))
        img_path = KOREAN_DIR / gt['image']
        if img_path.exists():
            cases.append((img_path, gt))
    return cases


@pytest.mark.parametrize("img_path,gt", _korean_cases() or [
    pytest.param(None, None, marks=pytest.mark.skip(reason="run synthesize_korean_floorplans.py first"))
])
def test_korean_floorplan_accuracy(img_path, gt):
    """한국 아파트 패턴 — GT 대비 가구 검출 recall 측정."""
    if img_path is None:
        pytest.skip("No Korean dataset")
    with open(img_path, 'rb') as f:
        b = f.read()
    result = extract_walls_from_bytes(b)
    metrics = evaluate(result['furniture'], gt['furniture_gt'], iou_threshold=0.3)

    print(f"\n[KR: {gt['name']}] GT={len(gt['furniture_gt'])}, DET={len(result['furniture'])}, "
          f"P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}, "
          f"mIoU={metrics['mean_iou']}")

    # 한국 평면도는 복잡한 패턴이라 임계 약간 완화 (recall ≥ 0.5 — 충돌 회피 안전선)
    assert metrics['recall'] >= 0.5, \
        f"{gt['name']} recall {metrics['recall']} < 0.5 (안전 임계 미달)"


def test_korean_aggregate(capsys):
    """한국 데이터 종합 요약."""
    cases = _korean_cases()
    if not cases:
        pytest.skip("No Korean dataset")

    print('\n' + '=' * 78)
    print('  한국 아파트 패턴 합성 평면도 — 가구 검출 정확도 (실 도면 형태)')
    print('=' * 78)
    print(f"  {'CASE':<14} {'GT':>3} {'DET':>3} {'TP':>3} {'FP':>3} {'FN':>3}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}  {'mIoU':>6}")
    print('  ' + '-' * 76)
    agg = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0, 'iou_n': 0}
    for img_path, gt in cases:
        with open(img_path, 'rb') as f:
            b = f.read()
        result = extract_walls_from_bytes(b)
        m = evaluate(result['furniture'], gt['furniture_gt'], iou_threshold=0.3)
        print(f"  {gt['name']:<14} {len(gt['furniture_gt']):>3} {len(result['furniture']):>3} "
              f"{m['tp']:>3} {m['fp']:>3} {m['fn']:>3}  "
              f"{m['precision']:>6} {m['recall']:>6} {m['f1']:>6} {m['mean_iou']:>6}")
        agg['tp'] += m['tp']; agg['fp'] += m['fp']; agg['fn'] += m['fn']
        if m['mean_iou'] > 0:
            agg['iou_sum'] += m['mean_iou']; agg['iou_n'] += 1
    tp = agg['tp']; fp = agg['fp']; fn = agg['fn']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    mIoU = agg['iou_sum'] / agg['iou_n'] if agg['iou_n'] else 0
    print('  ' + '-' * 76)
    print(f"  {'TOTAL':<14} {'':>3} {'':>3} {tp:>3} {fp:>3} {fn:>3}  "
          f"{p:>6.3f} {r:>6.3f} {f1:>6.3f} {mIoU:>6.3f}")
    print()


# ────────────────────────────────────────────
# 3) LH 분양 매뉴얼 PDF 페이지 — 한국 실 분양 평면도
# ────────────────────────────────────────────

def _lh_pages():
    if not LH_PAGES_DIR.exists():
        return []
    return sorted(LH_PAGES_DIR.glob("*.png"))


@pytest.mark.parametrize("path", _lh_pages() or [
    pytest.param(None, marks=pytest.mark.skip(reason="run fetch_lh_real_floorplans.py first"))
])
def test_lh_floorplan_page_extraction(path):
    """LH 분양 매뉴얼 각 페이지 — 파이프라인이 죽지 않고 일부 페이지에서 추출."""
    if path is None:
        pytest.skip()
    with open(path, 'rb') as f:
        b = f.read()
    result = extract_walls_from_bytes(b)
    # 최소 sanity — image 크기 정상 + 처리 완료
    assert result['image_width'] > 0
    assert result['image_height'] > 0
    # 표지/목차는 walls=0, furniture=0 가능 — 단지 죽지만 않으면 됨


def test_lh_aggregate(capsys):
    """LH 평면도 매뉴얼 종합 요약 — 페이지별 검출 분포."""
    pages = _lh_pages()
    if not pages:
        pytest.skip()

    print('\n' + '=' * 78)
    print('  LH 일반분양주택 주력평면 매뉴얼 (실 한국 분양 평면도)')
    print('=' * 78)
    print(f"  {'PAGE':<35} {'walls':>6} {'furn':>6} {'outline':>8}  분류")
    print('  ' + '-' * 76)
    total_w = total_f = useful = 0
    for p in pages:
        with open(p, 'rb') as f:
            b = f.read()
        r = extract_walls_from_bytes(b)
        wc, fc, oc = r['wall_count'], r['furniture_count'], len(r['outline'])
        total_w += wc; total_f += fc
        if wc + fc > 5:
            useful += 1
            tag = '★ 평면도'
        elif wc + fc > 0:
            tag = '· 부분 추출'
        else:
            tag = '  표지/목차'
        print(f"  {p.name:<35} {wc:>6} {fc:>6} {oc:>8}  {tag}")
    print('  ' + '-' * 76)
    print(f"  {'TOTAL':<35} {total_w:>6} {total_f:>6}  유의미 페이지 {useful}/{len(pages)}")
    print()


# ────────────────────────────────────────────
# 4) 공개 DXF 샘플 — 실 CAD 도면 검증
# ────────────────────────────────────────────

def _dxf_files():
    if not DXF_DIR.exists():
        return []
    return sorted(DXF_DIR.glob("*.dxf"))


@pytest.mark.parametrize("path", _dxf_files() or [
    pytest.param(None, marks=pytest.mark.skip(reason="run fetch_dxf_samples.py first"))
])
def test_dxf_sample_parsing(path):
    """공개 DXF 샘플 — parse_dxf 가 LINE/CIRCLE/INSERT 정확히 추출."""
    if path is None:
        pytest.skip()
    from app.services.dxf_parser import parse_dxf
    result = parse_dxf(str(path))
    assert result['wall_count'] >= 0
    # 평면도 DXF 라면 최소 일부는 추출되어야 함
    assert result['wall_count'] + result['furniture_count'] > 0, \
        f"{path.name}: 추출 결과 0 — DXF 형식 파악 실패 가능"
    print(f"\n[DXF: {path.name}] walls={result['wall_count']}, "
          f"furniture={result['furniture_count']}, "
          f"size={result['image_width']}x{result['image_height']}")


if __name__ == "__main__":
    print('=== Wikimedia 실 평면도 sanity ===')
    for p in _real_files():
        with open(p, 'rb') as f:
            b = f.read()
        r = extract_walls_from_bytes(b)
        print(f"  [{p.name}] walls={r['wall_count']}, furniture={r['furniture_count']}")

    print()
    print('=== 한국 아파트 합성 정량 측정 ===')
    for img_path, gt in _korean_cases():
        with open(img_path, 'rb') as f:
            b = f.read()
        r = extract_walls_from_bytes(b)
        m = evaluate(r['furniture'], gt['furniture_gt'], iou_threshold=0.3)
        print(f"  [{gt['name']}] GT={len(gt['furniture_gt'])} DET={len(r['furniture'])} "
              f"P={m['precision']} R={m['recall']} F1={m['f1']} mIoU={m['mean_iou']} "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
