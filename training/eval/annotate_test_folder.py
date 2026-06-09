# =============================================
# training/eval/annotate_test_folder.py
# 역할: 임의 폴더(기본: 바탕화면 test)의 이미지·영상에 하이브리드 검출을 돌려
#       bbox + 하자종류 + 등급을 그린 주석 이미지를 생성 (눈으로 교차검증용).
#
# 사용:
#   # GOOGLE_API_KEY 설정 + VLM_DETECTION_ENABLED=true 권장 (없으면 ONNX 후보만)
#   python training/eval/annotate_test_folder.py --folder "C:/Users/Codelab/Desktop/test"
#   python training/eval/annotate_test_folder.py --folder ... --provider gemini --video-interval 2
#
# 출력: <folder>/_annotated/<name>.jpg  +  _annotated/summary.json
# =============================================

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).resolve().parent
BACKEND = EVAL_DIR.parent.parent

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def _summarize(result) -> list:
    return [
        {
            "code": d.code, "type": d.class_display_ko or d.class_,
            "grade": d.grade, "status": d.status, "conf": d.conf,
            "bbox": [round(x, 1) for x in d.bbox_xyxy] if d.localization == "bbox" else None,
            "loc": d.localization, "onnx_conf": d.onnx_conf, "vlm_conf": d.vlm_conf,
        }
        for d in result.detections
    ]


async def main(folder: Path, provider: str | None, limit: int, vid_interval: float) -> None:
    os.chdir(str(BACKEND))
    sys.path.insert(0, str(BACKEND))

    import cv2
    import numpy as np
    from app.services.hybrid_detector import detect_hybrid_async
    from app.services.detection_overlay import annotate_hybrid, encode_jpeg

    if not folder.exists():
        print(f"[annotate] 폴더 없음: {folder}")
        return

    out_dir = folder / "_annotated"
    out_dir.mkdir(exist_ok=True)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VID_EXT)
    print(f"[annotate] 이미지 {len(images)} · 영상 {len(videos)} → {out_dir}")

    summary: dict = {"folder": str(folder), "provider": provider, "items": {}}

    # ── 이미지 ──
    for img_path in images:
        raw = img_path.read_bytes()
        try:
            result = await detect_hybrid_async(raw, provider=provider)
        except Exception as e:
            print(f"  [SKIP] {img_path.name}: {e}")
            continue
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        annotated = annotate_hybrid(frame, result)
        (out_dir / img_path.name).write_bytes(encode_jpeg(annotated))
        dets = _summarize(result)
        summary["items"][img_path.name] = {
            "engine": result.onnx_engine, "vlm_calls": result.vlm_calls,
            "confirmed": result.confirmed_count, "review": result.review_count,
            "detections": dets,
        }
        print(f"  [{img_path.name}] {result.onnx_engine} | "
              f"CONFIRMED {result.confirmed_count} REVIEW {result.review_count} | "
              + ", ".join(f"{d['code']}:{d['type']}({d['grade']})" for d in dets[:6]))

    # ── 영상 (interval 초마다 1프레임 샘플) ──
    for vid_path in videos:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [SKIP] {vid_path.name}: 열기 실패")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(fps * vid_interval))
        sampled = 0
        idx = 0
        while sampled < limit:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                try:
                    result = await detect_hybrid_async(buf.tobytes(), provider=provider)
                    annotated = annotate_hybrid(frame, result)
                    sec = round(idx / fps, 1)
                    name = f"{vid_path.stem}_t{sec}s.jpg"
                    (out_dir / name).write_bytes(encode_jpeg(annotated))
                    dets = _summarize(result)
                    summary["items"][name] = {"detections": dets}
                    print(f"  [{name}] " + ", ".join(
                        f"{d['code']}:{d['type']}({d['grade']})" for d in dets[:6]))
                except Exception as e:
                    print(f"  [SKIP] {vid_path.name}@{idx}: {e}")
            sampled += 1
            idx += step
            if total and idx >= total:
                break
        cap.release()

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[annotate] 완료 → {out_dir}\\summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="폴더 이미지·영상 하이브리드 검출 주석")
    ap.add_argument("--folder", default=r"C:/Users/Codelab/Desktop/test")
    ap.add_argument("--provider", default=None, help="gemini|claude|openai (기본 settings)")
    ap.add_argument("--limit", type=int, default=6, help="영상당 최대 샘플 프레임 수")
    ap.add_argument("--video-interval", type=float, default=2.0, help="영상 샘플 간격(초)")
    args = ap.parse_args()
    asyncio.run(main(Path(args.folder), args.provider, args.limit, args.video_interval))
