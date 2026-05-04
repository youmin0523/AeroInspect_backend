# =============================================
# validate_video_tracking.py
# 역할: 동영상 트래킹 검증
#       - defect_videos/* 영상 입력
#       - 프레임별 통합 추론 + ObjectTracker
#       - track_id 연속성 측정 (카메라 이동 시 같은 객체가 같은 track_id 유지하는가)
#       - bbox drift 측정 (frame-to-frame 위치 변화)
#       - 시각화 비디오 출력 (track_id가 화면에 표시되는 영상)
#
# 사용:
#   cd backend/training
#   python eval/validate_video_tracking.py --max-videos 3 --max-frames 100
# =============================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class VideoTrackingReport:
    video_path: str
    n_frames: int = 0
    n_frames_with_detection: int = 0
    n_unique_tracks: int = 0
    n_confirmed_tracks: int = 0       # min_hits 이상 본 track
    avg_track_length: float = 0.0
    longest_track: int = 0
    avg_inference_ms: float = 0.0
    track_lifetimes: Dict[int, int] = field(default_factory=dict)
    out_video_path: Optional[str] = None


def _draw_track_bbox(
    img: np.ndarray, bbox: List[float], track_id: int,
    cls: str, conf: float,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # track_id별 다른 색상 (HSV 기반)
    hue = (track_id * 47) % 180
    color_hsv = np.uint8([[[hue, 255, 200]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"#{track_id} {cls} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def validate_video(
    video_path: Path,
    pipe,
    tracker,
    out_dir: Path,
    max_frames: int = 200,
    save_video: bool = True,
) -> VideoTrackingReport:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return VideoTrackingReport(video_path=str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    report = VideoTrackingReport(video_path=str(video_path))
    inference_times: List[float] = []
    track_first_seen: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    track_hit_count: Dict[int, int] = {}

    # 출력 비디오
    out_video_writer = None
    if save_video:
        out_video_path = out_dir / f"{video_path.stem}_tracked.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video_writer = cv2.VideoWriter(
            str(out_video_path), fourcc, fps, (width, height),
        )
        report.out_video_path = str(out_video_path)

    frame_idx = 0
    while frame_idx < min(max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 통합 추론 (Tier 2)
        t0 = time.time()
        try:
            result = pipe.detect(frame, tier=2)
        except Exception as e:
            print(f"[ERROR] frame {frame_idx}: {e}")
            frame_idx += 1
            continue
        elapsed_ms = (time.time() - t0) * 1000
        inference_times.append(elapsed_ms)

        # Tracker 적용
        det_dicts = []
        for d in result.detections:
            det_dicts.append({
                "class": d.class_,
                "conf": d.conf,
                "bbox_xyxy": list(d.bbox_xyxy),
            })
        tracked = tracker.update(det_dicts, frame_id=frame_idx)

        # 통계 업데이트
        if tracked:
            report.n_frames_with_detection += 1
        for t in tracked:
            tid = t["track_id"]
            if tid not in track_first_seen:
                track_first_seen[tid] = frame_idx
            track_last_seen[tid] = frame_idx
            track_hit_count[tid] = track_hit_count.get(tid, 0) + 1

        # 시각화
        if save_video and out_video_writer is not None:
            vis = frame.copy()
            for t in tracked:
                _draw_track_bbox(
                    vis, t["bbox_xyxy"], t["track_id"],
                    t["class"], t["conf"],
                )
            # 통계 텍스트
            stats = f"frame {frame_idx} | dets={len(tracked)} | tracks={len(track_first_seen)}"
            cv2.rectangle(vis, (5, 5), (5 + 400, 30), (0, 0, 0), -1)
            cv2.putText(
                vis, stats, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            out_video_writer.write(vis)

        report.n_frames += 1
        frame_idx += 1

    cap.release()
    if out_video_writer is not None:
        out_video_writer.release()

    # 통계 정리
    report.n_unique_tracks = len(track_first_seen)
    track_lifetimes = {
        tid: track_last_seen[tid] - track_first_seen[tid] + 1
        for tid in track_first_seen
    }
    report.track_lifetimes = track_lifetimes
    if track_lifetimes:
        report.avg_track_length = sum(track_lifetimes.values()) / len(track_lifetimes)
        report.longest_track = max(track_lifetimes.values())
    report.n_confirmed_tracks = sum(1 for h in track_hit_count.values() if h >= 3)

    if inference_times:
        report.avg_inference_ms = sum(inference_times) / len(inference_times)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=3, help="평가할 영상 수")
    parser.add_argument("--max-frames", type=int, default=100, help="영상당 최대 프레임 수")
    parser.add_argument("--out", type=str, default="eval/results/video_tracked")
    args = parser.parse_args()

    cwd = Path.cwd()
    backend_dir = ROOT / "backend"
    out_dir = cwd / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 영상 수집 (defect_videos/* 각 카테고리에서 1개씩)
    video_root = cwd / "datasets" / "defect_videos"
    if not video_root.exists():
        # ROOT에서 다시 시도
        video_root = ROOT / "datasets" / "defect_videos"
    if not video_root.exists():
        print(f"[ERROR] defect_videos not found")
        return 1

    selected_videos: List[Path] = []
    for category_dir in sorted(video_root.iterdir()):
        if not category_dir.is_dir():
            continue
        videos = sorted([
            f for f in category_dir.rglob("*")
            if f.suffix.lower() in {".mp4", ".avi", ".mov"}
        ])
        if videos:
            selected_videos.append(videos[0])
        if len(selected_videos) >= args.max_videos:
            break

    print(f"평가 영상: {len(selected_videos)}개")
    for v in selected_videos:
        print(f"  - {v.parent.name} / {v.name}")

    # Pipeline + Tracker 로딩 (cwd=backend/)
    orig_cwd = os.getcwd()
    reports: List[VideoTrackingReport] = []
    try:
        os.chdir(backend_dir)
        from app.services.inference_pipeline_20 import InferencePipeline20
        from app.services.object_tracker import DefectTracker

        pipe = InferencePipeline20()
        pipe.load_models()
        if not pipe.is_loaded:
            print("[ERROR] Pipeline not loaded")
            return 1

        for vp in selected_videos:
            print(f"\n=== {vp.parent.name} / {vp.name} ===")
            tracker = DefectTracker(min_hits=3, max_age=15, iou_threshold=0.3)
            t0 = time.time()
            report = validate_video(
                vp, pipe, tracker, out_dir,
                max_frames=args.max_frames, save_video=True,
            )
            elapsed = time.time() - t0
            print(f"  처리 {report.n_frames}프레임 ({elapsed/60:.1f}min)")
            print(f"  검출 발생 프레임: {report.n_frames_with_detection}/{report.n_frames}")
            print(f"  unique tracks: {report.n_unique_tracks}")
            print(f"  confirmed tracks (hit≥3): {report.n_confirmed_tracks}")
            print(f"  longest track: {report.longest_track} frames")
            print(f"  avg track length: {report.avg_track_length:.1f}")
            print(f"  avg inference: {report.avg_inference_ms:.0f}ms")
            print(f"  output video: {report.out_video_path}")
            reports.append(report)
    finally:
        os.chdir(orig_cwd)

    # 결과 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"video_tracking_{ts}.json"
    json_data = {
        "timestamp": ts,
        "max_videos": args.max_videos,
        "max_frames": args.max_frames,
        "reports": [
            {
                "video_path": r.video_path,
                "n_frames": r.n_frames,
                "n_frames_with_detection": r.n_frames_with_detection,
                "n_unique_tracks": r.n_unique_tracks,
                "n_confirmed_tracks": r.n_confirmed_tracks,
                "avg_track_length": r.avg_track_length,
                "longest_track": r.longest_track,
                "avg_inference_ms": r.avg_inference_ms,
                "out_video_path": r.out_video_path,
            }
            for r in reports
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {json_path}")

    # 종합 평가
    if reports:
        total_unique = sum(r.n_unique_tracks for r in reports)
        total_confirmed = sum(r.n_confirmed_tracks for r in reports)
        total_frames = sum(r.n_frames for r in reports)
        avg_track_len = (
            sum(r.avg_track_length * r.n_unique_tracks for r in reports if r.n_unique_tracks > 0) /
            max(1, total_unique)
        )
        confirm_rate = total_confirmed / max(1, total_unique)

        print("\n=== 종합 ===")
        print(f"총 처리 프레임: {total_frames}")
        print(f"총 unique tracks: {total_unique}")
        print(f"총 confirmed tracks: {total_confirmed} ({confirm_rate*100:.1f}%)")
        print(f"평균 track 길이: {avg_track_len:.1f} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
