"""
Roboflow 보조모델 추론 어댑터 (rfenv py3.12 전용, CPU).
채택된 Roboflow 학습모델을 test 이미지에 추론 → 우리 detection 포맷(JSON)으로 저장.
이 JSON을 backend/venv 쪽 ensemble_eval.py가 읽어 우리 모델 검출과 WBF fuse.

Cross-env 이유: 우리 ONNX는 venv(py3.13), Roboflow inference는 rfenv(py3.12) — 동일 프로세스 불가.
→ rfenv에서 보조검출만 뽑아 JSON으로 전달.

Recall 최우선(feedback_recall_priority_paid_service): conf 낮게(0.05) → 놓침 최소.
실행: backend/rfenv/Scripts/python.exe roboflow_adapter.py <test_glob> <out_json> [model_id1 model_id2 ...]
"""
import sys, glob, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from inference import get_model

KEY = "nuC9Lxr51Ds7c1IwN4Gy"
# Recall 우선: 약한 검출도 살림. 약한 OOD 모델(예 crack-bphdr mAP22)은 conf<0.05만 내므로
# 기본 0.01로 낮춤. 등급단계(합의=CONFIRMED)에서 Precision 방어. env RF_CONF로 조절.
CONF = float(os.environ.get("RF_CONF", "0.01"))

# 채택 모델 → 우리 taxonomy 클래스 매핑 (feedback_onnx_class_mapping_audit)
# Roboflow 클래스명(소문자) → 우리 통합 클래스명. 매핑 없으면 drop.
CLASS_MAP = {
    # M1 균열
    "crack": "crack", "crack_detection": "crack", "crack detection": "crack",
    # M3 유리
    "defect": "glass_defect", "glass": "glass_defect",
    # M4 context / furniture (게이팅 보조)
    "wall": "wall", "ceiling": "ceiling", "floor": "floor", "window": "window", "door": "door",
    # thermal 단열·습기
    "moisture": "moisture", "insulation": "insulation", "delamination": "delamination",
    "hollow": "delamination", "air infiltration": "insulation", "air leakage": "insulation",
}


def infer_one(model, img_path):
    """단일 이미지 추론 → [{bbox_xyxy, conf, class, src}] (우리 포맷)."""
    import cv2
    im = cv2.imread(img_path)
    if im is None:
        return []
    h, w = im.shape[:2]
    out = []
    try:
        res = model.infer(img_path, confidence=CONF)
        preds = res[0].predictions
    except Exception as e:
        print(f"  infer fail {os.path.basename(img_path)}: {str(e)[:60]}", flush=True)
        return []
    for p in preds:
        cname = (getattr(p, "class_name", None) or getattr(p, "class", "") or "").lower()
        mapped = CLASS_MAP.get(cname)
        if mapped is None:
            continue  # 매핑 불가 클래스 drop (오탐 방지)
        # roboflow는 center x,y,width,height (픽셀)
        cx, cy, bw, bh = float(p.x), float(p.y), float(p.width), float(p.height)
        x1, y1, x2, y2 = cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2
        out.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "conf": float(p.confidence),
            "class": mapped,
            "src": "roboflow",
        })
    return out


def main():
    if len(sys.argv) < 4:
        print("usage: roboflow_adapter.py <test_glob> <out_json> <model_id...>")
        return
    pattern = sys.argv[1]
    out_json = sys.argv[2]
    model_ids = sys.argv[3:]

    # 정렬 + 전체(상한 500) — eval(sorted) 과 basename 정합 위해
    imgs = sorted(glob.glob(pattern))[: int(os.environ.get("RF_MAX", "500"))]
    print(f"[adapter] {len(imgs)} imgs, {len(model_ids)} models", flush=True)

    models = {}
    for mid in model_ids:
        try:
            models[mid] = get_model(model_id=mid, api_key=KEY)
            print(f"[adapter] loaded {mid}", flush=True)
        except Exception as e:
            print(f"[adapter] LOAD FAIL {mid}: {str(e)[:80]}", flush=True)

    result = {}  # img_path -> list of detections (all roboflow models merged)
    for ip in imgs:
        dets = []
        for mid, m in models.items():
            dets.extend(infer_one(m, ip))
        result[os.path.basename(ip)] = dets

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f)
    total = sum(len(v) for v in result.values())
    hit = sum(1 for v in result.values() if v)
    print(f"[adapter] done → {out_json} | {hit}/{len(imgs)}장 검출, 총 {total}건", flush=True)


if __name__ == "__main__":
    main()
