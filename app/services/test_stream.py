# =============================================
# app/services/test_stream.py
# 역할: 테스트 모드 스트리밍 서비스
#       - 카테고리별 균등 샘플링: 각 하자 유형이 골고루 노출
#       - RGB/Thermal 쌍 동기화: 프레임 버전 카운터로 정합성 보장
#       - 쌍이 없는 데이터는 Thermal에 No Signal 표시
#       - 시작/일시중지/정지 재생 제어
#       - image_crop 생성: DefectCard 썸네일 표시용 base64 JPEG
#       - 20종 ONNX 추론 또는 목업 하자 생성 폴백
# =============================================

from __future__ import annotations

import asyncio
import base64
import os
import random
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import settings

# ── 한글 폰트 로드 (Windows: Malgun Gothic, 폴백: PIL 기본) ──
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}

def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """캐시된 한글 TrueType 폰트를 반환."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font_paths = [
        "C:/Windows/Fonts/malgunbd.ttf",   # Malgun Gothic Bold
        "C:/Windows/Fonts/malgun.ttf",      # Malgun Gothic
        "C:/Windows/Fonts/gulim.ttc",       # Gulim
    ]
    for fp in font_paths:
        if os.path.isfile(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font

# 지원 파일 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# 테스트 이미지 디렉토리명 → 하자 카테고리 매핑 (목업 생성용)
_DIR_TO_DEFECT = {
    "ext_crack":          {"area": "A", "category_code": "A-02", "defect_type": "균열 (구조 균열)",        "severity": "HIGH", "defect_class": "crack_structural"},
    "ext_building_crack": {"area": "A", "category_code": "A-02", "defect_type": "균열 (구조 균열)",        "severity": "HIGH", "defect_class": "crack_structural"},
    "ext_wall_crack":     {"area": "A", "category_code": "A-03", "defect_type": "균열 (마감 균열)",        "severity": "MED",  "defect_class": "crack_finishing"},
    "ext_floor_crack":    {"area": "D", "category_code": "D-03", "defect_type": "바닥 오염·스크래치",      "severity": "LOW",  "defect_class": "floor_stain"},
    "ext_glass":          {"area": "E", "category_code": "E-01", "defect_type": "창호 유리 스크래치·파손", "severity": "MED",  "defect_class": "glass_scratch"},
    "ext_surface":        {"area": "C", "category_code": "C-04", "defect_type": "찍힘·스크래치 (벽·천장)", "severity": "LOW",  "defect_class": "scratch_wall"},
    "ext_concrete":       {"area": "B", "category_code": "B-04", "defect_type": "방수층 들뜸 / 누수 흔적", "severity": "HIGH", "defect_class": "waterproof_defect"},
}

# Crack900 쌍 데이터에 대한 하자 정보
_PAIRED_DEFECT = {"area": "A", "category_code": "A-02", "defect_type": "균열 (구조 균열)", "severity": "HIGH", "defect_class": "crack_structural"}


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


@dataclass
class TestFrame:
    """단일 테스트 프레임 (RGB + Thermal 쌍)."""
    rgb_path: str
    thermal_path: Optional[str] = None  # None이면 No Signal
    category: str = "unknown"           # 하자 카테고리 태그


class TestStreamService:
    """테스트 모드 스트리밍 서비스."""

    def __init__(self):
        # ── 카테고리별 프레임 저장소 (균등 샘플링용) ──
        self._category_frames: Dict[str, List[TestFrame]] = {}
        self._category_indices: Dict[str, int] = {}

        # ── 업로드 파일 ──
        self._uploaded_files: List[str] = []
        self._upload_index: int = 0

        # ── 현재 표시 중인 프레임 (RGB ↔ Thermal 동기화) ──
        self._current_rgb_jpeg: Optional[bytes] = None
        self._current_thermal_jpeg: Optional[bytes] = None
        self._frame_version: int = 0   # RGB 제너레이터가 증가, Thermal이 추적

        self._source: str = "project"
        self._models_loaded: bool = False
        self._models_loading: bool = False  # load_models 진행 중 플래그 (사전로드/중복로드 가드 + 프론트 로딩 표시)
        self._frame_counter: int = 0
        self._scanned: bool = False

        # 영상 추론 background task가 적재하는 가장 최신 detection.
        # (이미지 카테고리 경로용 — 영상은 별도 _video_inference_task로 처리.)
        self._pending_video_detection: Optional[dict] = None

        # ── 영상 직접재생 모드(test mode 업로드 mp4 전용) ──
        # 프론트가 <video src=/test/upload/file/{name}> 으로 네이티브 디코드.
        # 백엔드는 MJPEG yield를 멈추고 background inference만 돌리고 WS로 결과 push.
        # _active_video_filename 이 set 되면 rgb_mjpeg_generator 는 "WATCHING" 플레이스홀더만 흘림.
        self._active_video_filename: Optional[str] = None
        self._active_video_fps: float = 0.0
        self._active_video_duration: float = 0.0
        self._active_video_frame_w: int = 0
        self._active_video_frame_h: int = 0
        self._video_inference_task: Optional[asyncio.Task] = None

        # ── 하자별 프레임 저장소 (클릭 시 해당 시점 프레임 조회용) ──
        self._defect_frames: OrderedDict[str, Tuple[bytes, Optional[bytes]]] = OrderedDict()
        self._MAX_DEFECT_FRAMES = 200

        # ── 감지 시각화 모드 ──
        self._detection_mode: str = "bbox"  # 'bbox' | 'detection'

        # ── 재생 상태 ──
        self._playing: bool = False
        self._paused: bool = False

        # ── 이미지 detection 백그라운드 태스크 동시성 제어 ──
        # 이미지 프레임 detection(ONNX+VLM 하이브리드)을 MJPEG yield 경로에서 분리해
        # 백그라운드로 돌린다(영상 경로와 동일 패턴). 1 vCPU 보호: 직전 detection 이 아직
        # 돌고 있으면 새 프레임은 표시만 하고 detection 은 스킵 — 태스크 누적 방지.
        self._image_detect_inflight: int = 0
        self._MAX_INFLIGHT_DETECT: int = 1
        # 프레임당 동시 표시할 최대 검출 수 — 여러 하자를 한 번에 표시하되,
        # 화면 박스 혼잡/카드 폭주를 막기 위한 상한 (신뢰도 내림차순 상위 N).
        self._MAX_DEFECTS_PER_FRAME: int = 8

    # ── 재생 제어 ────────────────────────────
    @property
    def play_state(self) -> str:
        if self._playing and not self._paused:
            return "playing"
        if self._playing and self._paused:
            return "paused"
        return "stopped"

    def start_playback(self) -> None:
        self._playing = True
        self._paused = False
        print("[TestStream] PLAY (start)")

    def pause_playback(self) -> None:
        self._paused = True
        print("[TestStream] PAUSE")

    def resume_playback(self) -> None:
        self._paused = False
        print("[TestStream] PLAY (resume)")

    def stop_playback(self) -> None:
        self._playing = False
        self._paused = False
        self._frame_version = 0
        # 카테고리별 인덱스 리셋
        for cat in self._category_indices:
            self._category_indices[cat] = 0
        self._upload_index = 0
        self._current_rgb_jpeg = None
        self._current_thermal_jpeg = None
        # 진행 중이던 영상 inference task 취소 + active video 메타 클리어
        self._cancel_video_inference()
        self._clear_active_video()
        print("[TestStream] STOP")

    # ── 영상 직접재생 모드 메타 ────────────────────
    @property
    def active_media(self) -> dict:
        """프론트가 현재 재생 대상이 영상인지 이미지인지 결정할 때 폴링하는 메타.
        영상이면 <video src=/test/upload/file/{filename}> 으로 직접 재생,
        아니면 기존 MJPEG <img src=/test/rgb>."""
        if self._active_video_filename:
            return {
                "kind": "video",
                "filename": self._active_video_filename,
                "fps": self._active_video_fps,
                "duration_sec": self._active_video_duration,
                "frame_w": self._active_video_frame_w,
                "frame_h": self._active_video_frame_h,
            }
        return {"kind": "image", "filename": None}

    def _clear_active_video(self) -> None:
        self._active_video_filename = None
        self._active_video_fps = 0.0
        self._active_video_duration = 0.0
        self._active_video_frame_w = 0
        self._active_video_frame_h = 0

    def _cancel_video_inference(self) -> None:
        task = self._video_inference_task
        if task is not None and not task.done():
            task.cancel()
        self._video_inference_task = None

    # ── 이미지 스캔 (카테고리별 그룹핑) ────────────
    def scan_images(self) -> dict:
        """프로젝트 로컬 이미지를 스캔하고 카테고리별로 그룹핑한다."""
        self._category_frames.clear()
        self._category_indices.clear()

        # 1) Crack900 paired dataset 스캔
        thermal_dir = settings.TEST_THERMAL_DIR
        rgb_paired_dir = thermal_dir.replace("2_IR", "1_RGB")

        rgb_map: Dict[str, str] = {}
        ir_map: Dict[str, str] = {}

        if os.path.isdir(rgb_paired_dir):
            for root, _, files in os.walk(rgb_paired_dir):
                for f in files:
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                        rgb_map[f] = os.path.join(root, f)

        if os.path.isdir(thermal_dir):
            for root, _, files in os.walk(thermal_dir):
                for f in files:
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                        ir_map[f] = os.path.join(root, f)

        paired_frames: List[TestFrame] = []
        for fname in rgb_map:
            if fname in ir_map:
                paired_frames.append(TestFrame(
                    rgb_path=rgb_map[fname],
                    thermal_path=ir_map[fname],
                    category="paired_crack",
                ))

        if paired_frames:
            random.shuffle(paired_frames)
            self._category_frames["paired_crack"] = paired_frames
            self._category_indices["paired_crack"] = 0

        # 2) test_external 하위 디렉토리 → 1단계 폴더명 = 하나의 카테고리
        #    예: ext_crack/train/images/*.jpg → 카테고리 "ext_crack"
        rgb_dir = settings.TEST_IMAGES_DIR
        if os.path.isdir(rgb_dir):
            for root, _, files in os.walk(rgb_dir):
                if not files:
                    continue
                # 최상위 디렉토리 자체는 건너뜀
                if os.path.normpath(root) == os.path.normpath(rgb_dir):
                    continue

                # test_external 바로 아래 1단계 디렉토리명을 카테고리로 사용
                rel_path = os.path.relpath(root, rgb_dir)
                cat_key = Path(rel_path).parts[0]  # "ext_crack", "ext_glass", …
                if cat_key not in self._category_frames:
                    self._category_frames[cat_key] = []

                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                        self._category_frames[cat_key].append(TestFrame(
                            rgb_path=os.path.join(root, f),
                            thermal_path=None,
                            category=cat_key,
                        ))

        # 각 카테고리 셔플 + 인덱스 초기화
        for cat, frames in list(self._category_frames.items()):
            if not frames:
                del self._category_frames[cat]
                continue
            random.shuffle(frames)
            if cat not in self._category_indices:
                self._category_indices[cat] = 0

        # 업로드 디렉토리 스캔
        self._scan_uploaded_files()

        self._scanned = True
        total = sum(len(v) for v in self._category_frames.values())
        cat_summary = {k: len(v) for k, v in self._category_frames.items()}
        print(
            f"[TestStream] 스캔 완료: 총={total}, "
            f"카테고리={len(self._category_frames)}개, "
            f"분포={cat_summary}, "
            f"Upload={len(self._uploaded_files)}"
        )

        return {
            "total_frames": total,
            "categories": cat_summary,
            "uploaded_count": len(self._uploaded_files),
        }

    def _scan_uploaded_files(self) -> None:
        self._uploaded_files.clear()
        upload_dir = settings.TEST_UPLOAD_DIR
        if os.path.isdir(upload_dir):
            for f in os.listdir(upload_dir):
                if Path(f).suffix.lower() in ALL_EXTENSIONS:
                    self._uploaded_files.append(os.path.join(upload_dir, f))

    # ── 모델 로드 ────────────────────────────
    @property
    def models_status(self) -> dict:
        """프론트가 '로딩 중 / 준비됨 / 오류'를 구분해 표시하도록 노출하는 모델 상태."""
        return {"models_loaded": self._models_loaded, "models_loading": self._models_loading}

    async def load_models(self) -> dict:
        if self._models_loaded:
            return {"status": "already_loaded"}
        # 이미 다른 호출(사전로드 warmup ↔ start)이 로딩 중이면 중복 to_thread 로드 방지.
        if self._models_loading:
            return {"status": "loading"}
        self._models_loading = True
        try:
            from app.services.inference_pipeline_20 import pipeline20
            await asyncio.to_thread(pipeline20.load_models)
            self._models_loaded = pipeline20.is_loaded
            print(f"[TestStream] 20종 파이프라인 로드: loaded={self._models_loaded}")
            return {"status": "loaded" if self._models_loaded else "partial"}
        except Exception as e:
            print(f"[TestStream] 모델 로드 실패 (목업 폴백 사용): {e}")
            return {"status": "fallback_mock", "error": str(e)}
        finally:
            self._models_loading = False

    # ── 소스 전환 ────────────────────────────
    def set_source(self, source: str) -> None:
        if source not in ("project", "upload"):
            raise ValueError(f"Invalid source: {source}")
        self._source = source
        self._upload_index = 0
        if source == "upload":
            self._scan_uploaded_files()
        print(f"[TestStream] 소스 전환: {source}")

    @property
    def source(self) -> str:
        return self._source

    # ── 감지 모드 전환 ────────────────────────
    def set_detection_mode(self, mode: str) -> None:
        if mode in ("bbox", "detection"):
            self._detection_mode = mode
            print(f"[TestStream] 감지 모드: {mode}")

    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    # ── 파일 업로드 ────────────────────────────
    async def add_uploaded_files(self, files) -> dict:
        """대용량 파일을 chunk 스트리밍으로 디스크 저장.
        `await upload_file.read()` 는 전체 바이트를 RAM에 올려 1GB Fly 머신에서
        모델+multipart 동시 로드 시 OOM/스왑으로 36MB 영상이 수십 초 걸리는 사고 발생.
        1MiB chunk 단위로 흘려 보내 RAM 점유는 chunk 크기로 cap.
        """
        upload_dir = settings.TEST_UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        saved = 0
        total_size = 0
        chunk_size = 1024 * 1024  # 1 MiB
        import shutil
        for upload_file in files:
            ext = Path(upload_file.filename or "").suffix.lower()
            if ext not in ALL_EXTENSIONS:
                continue
            safe_name = f"{uuid.uuid4().hex[:8]}_{upload_file.filename}"
            dest = os.path.join(upload_dir, safe_name)
            written = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = await upload_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
            self._uploaded_files.append(dest)
            saved += 1
            total_size += written
        return {
            "saved": saved,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_uploaded": len(self._uploaded_files),
        }

    def clear_uploaded_files(self) -> dict:
        upload_dir = settings.TEST_UPLOAD_DIR
        removed = 0
        if os.path.isdir(upload_dir):
            for f in os.listdir(upload_dir):
                fp = os.path.join(upload_dir, f)
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    pass
        self._uploaded_files.clear()
        return {"removed": removed}

    def list_uploaded_files(self) -> List[dict]:
        self._scan_uploaded_files()
        result = []
        for fp in self._uploaded_files:
            try:
                stat = os.stat(fp)
                result.append({
                    "name": os.path.basename(fp),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "type": "video" if _is_video(fp) else "image",
                })
            except OSError:
                pass
        return result

    # ── 하자별 프레임 저장/조회 ────────────────────
    def store_defect_frame(
        self, defect_id: str,
        bbox: Optional[dict] = None,
        label: str = "",
        severity: str = "HIGH",
        rgb_jpeg: Optional[bytes] = None,
        thermal_jpeg: Optional[bytes] = None,
    ) -> None:
        """하자 발생 시점의 RGB/Thermal JPEG + 메타데이터를 저장.
        rgb_jpeg/thermal_jpeg가 주어지면 그것(=detection이 발생한 그 프레임의 raw 스냅샷)을
        쓰고, 없으면 _current_*_jpeg로 폴백. 폴백은 프레임 드리프트 위험이 있으므로
        호출자는 가급적 명시 전달할 것 (broadcast가 0.4s 지연되는 사이 _current_*_jpeg가
        다음 프레임으로 바뀌어 bbox/이미지 짝이 어긋나는 사고를 방지)."""
        self._defect_frames[defect_id] = {
            "rgb": rgb_jpeg if rgb_jpeg is not None else self._current_rgb_jpeg,
            "thermal": thermal_jpeg if thermal_jpeg is not None else self._current_thermal_jpeg,
            "bbox": bbox,
            "label": label,
            "severity": severity,
        }
        while len(self._defect_frames) > self._MAX_DEFECT_FRAMES:
            self._defect_frames.popitem(last=False)

    def get_defect_frame(
        self, defect_id: str, channel: str, mode: str = "bbox"
    ) -> Optional[bytes]:
        """저장된 하자 시점의 프레임을 mode에 따라 시각화하여 반환.
        channel: 'rgb' | 'thermal', mode: 'bbox' | 'detection' | 'raw'.
        - 'raw': 오버레이 없는 원본 JPEG. 프론트가 SVG로 자체 오버레이를 그릴 때 사용
                 (스캔 sweep + SVG bbox/마스크 + 라벨 페이드인 같은 모션 UX).
        - 'bbox': 단순 네모박스 + 한글 라벨이 burned-in 된 JPEG.
        - 'detection': 반투명 마스크 + 윤곽 + 코너마커 + 심각도 색상이 burned-in 된 JPEG."""
        data = self._defect_frames.get(defect_id)
        if data is None:
            return None
        jpeg = data["rgb"] if channel == "rgb" else data["thermal"]
        bbox = data["bbox"]
        label = data["label"]
        severity = data["severity"]
        if mode == "raw":
            return jpeg
        if mode == "detection":
            return self._draw_detection_on_jpeg(jpeg, bbox, label, severity)
        return self._draw_bbox_on_jpeg(jpeg, bbox, label)

    @staticmethod
    def _draw_bbox_on_jpeg(
        jpeg_bytes: Optional[bytes], bbox: Optional[dict], label: str = ""
    ) -> Optional[bytes]:
        """JPEG 바이트에 bbox 네모박스 + 한글 라벨을 오버레이하여 다시 JPEG로 반환.
        PIL을 사용하여 한글 폰트를 정상 렌더링."""
        if jpeg_bytes is None:
            return None
        # JPEG → numpy
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg_bytes

        if bbox and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(w, int(bbox["x2"]))
            y2 = min(h, int(bbox["y2"]))

            # cv2로 네모박스 그리기 (빨간색, 두께 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # PIL로 한글 라벨 렌더링
            if label:
                # BGR → RGB → PIL
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font = _get_font(16)

                # 텍스트 크기 측정
                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]

                # 라벨 위치 (bbox 위쪽, 화면 밖이면 아래로)
                label_x = x1
                label_y = y1 - th - 8
                if label_y < 0:
                    label_y = y2 + 4

                # 라벨 배경 (빨간색) + 텍스트 (흰색)
                draw.rectangle(
                    [label_x, label_y, label_x + tw + 8, label_y + th + 6],
                    fill=(255, 0, 0),
                )
                draw.text(
                    (label_x + 4, label_y + 2), label,
                    fill=(255, 255, 255), font=font,
                )

                # PIL → BGR numpy
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.MJPEG_JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _draw_detection_on_jpeg(
        jpeg_bytes: Optional[bytes], bbox: Optional[dict],
        label: str = "", severity: str = "HIGH",
    ) -> Optional[bytes]:
        """객체감지 스타일: 반투명 마스크 + 윤곽 강조 + 코너 마커 + 심각도 색상."""
        if jpeg_bytes is None:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg_bytes

        if bbox and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(w, int(bbox["x2"]))
            y2 = min(h, int(bbox["y2"]))

            # 심각도별 색상 (BGR)
            severity_colors = {
                "HIGH": (0, 0, 255),    # 빨강
                "MED":  (0, 140, 255),  # 주황
                "LOW":  (0, 200, 255),  # 노랑
            }
            color = severity_colors.get(severity, (0, 0, 255))
            color_rgb = (color[2], color[1], color[0])  # PIL용 RGB

            # 1) 반투명 마스크 오버레이
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

            # 2) bbox 내부 윤곽 강조 (에지 검출)
            if y2 > y1 and x2 > x1:
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_roi, 50, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    cnt_shifted = cnt.copy()
                    cnt_shifted[:, :, 0] += x1
                    cnt_shifted[:, :, 1] += y1
                    cv2.drawContours(frame, [cnt_shifted], -1, color, 1)

            # 3) 코너 마커 (L자형 브래킷)
            bracket_len = max(12, min(x2 - x1, y2 - y1) // 5)
            t = 3  # 두께
            # 좌상
            cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, t)
            cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, t)
            # 우상
            cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, t)
            cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, t)
            # 좌하
            cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, t)
            cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, t)
            # 우하
            cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, t)
            cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, t)

            # 4) PIL로 한글 라벨 + 심각도 뱃지
            if label:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font = _get_font(15)
                font_sm = _get_font(12)

                # 심각도 뱃지 텍스트
                sev_text = {"HIGH": "HIGH", "MED": "MED", "LOW": "LOW"}.get(severity, severity)
                sev_bbox = draw.textbbox((0, 0), sev_text, font=font_sm)
                sev_w = sev_bbox[2] - sev_bbox[0]
                sev_h = sev_bbox[3] - sev_bbox[1]

                # 라벨 텍스트
                label_bbox = draw.textbbox((0, 0), label, font=font)
                lw = label_bbox[2] - label_bbox[0]
                lh = label_bbox[3] - label_bbox[1]

                total_w = sev_w + 10 + lw + 16
                bar_h = max(sev_h, lh) + 10

                # 위치 (bbox 위쪽, 넘치면 아래)
                bx = x1
                by = y1 - bar_h - 4
                if by < 0:
                    by = y2 + 4

                # 배경 (어두운 반투명)
                draw.rectangle(
                    [bx, by, bx + total_w, by + bar_h],
                    fill=(30, 30, 30, 220),
                )
                # 심각도 뱃지 (색상 배경)
                draw.rectangle(
                    [bx + 2, by + 2, bx + sev_w + 10, by + bar_h - 2],
                    fill=color_rgb,
                )
                draw.text(
                    (bx + 5, by + (bar_h - sev_h) // 2 - 1),
                    sev_text, fill=(255, 255, 255), font=font_sm,
                )
                # 라벨 텍스트
                draw.text(
                    (bx + sev_w + 14, by + (bar_h - lh) // 2 - 1),
                    label, fill=(255, 255, 255), font=font,
                )

                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.MJPEG_JPEG_QUALITY])
        return buf.tobytes()

    # ── 다음 프레임 선택 (카테고리 균등 샘플링) ────────────
    def _advance_frame(self) -> Optional[TestFrame]:
        """카테고리를 균등 확률로 선택한 뒤 해당 카테고리에서 순차 추출.
        → 소수 카테고리(유리, 콘크리트 등)도 균열과 비슷한 빈도로 노출."""
        if self._source == "upload":
            if not self._uploaded_files:
                return None
            idx = self._upload_index % len(self._uploaded_files)
            self._upload_index += 1
            if self._upload_index >= len(self._uploaded_files):
                random.shuffle(self._uploaded_files)
                self._upload_index = 0
            return TestFrame(rgb_path=self._uploaded_files[idx], thermal_path=None, category="upload")

        if not self._category_frames:
            return None

        # 1) 랜덤 카테고리 선택 (균등 확률)
        categories = list(self._category_frames.keys())
        cat = random.choice(categories)
        frames = self._category_frames[cat]

        # 2) 해당 카테고리에서 순차 추출 (한 바퀴 돌면 리셔플)
        idx = self._category_indices.get(cat, 0)
        frame = frames[idx % len(frames)]
        idx += 1
        if idx >= len(frames):
            random.shuffle(frames)
            idx = 0
        self._category_indices[cat] = idx

        return frame

    # ── MJPEG 제너레이터 ────────────────────────
    async def rgb_mjpeg_generator(self):
        """RGB MJPEG 스트림.

        설계 원칙: 프레임 표시는 detection 과 절대 결합하지 않는다.
        과거에는 프레임마다 _detect(ONNX+VLM 하이브리드)를 await 한 뒤에야 yield 해서,
        VLM 응답(수 초~수십 초, 쿼터/네트워크 장애 시 사실상 무한)까지 이미지가 화면에
        안 뜨고 스트림이 얼어붙는 사고가 있었다. 이제 raw 프레임을 즉시 yield 하고
        detection 은 _detect_and_broadcast_image 백그라운드 태스크로 분리한다.
        라이브 박스는 프론트가 WS defect.new → DetectionOverlay(SVG) 로 직접 그린다.
        """
        while True:
            if not self._playing:
                frame = self._stopped_frame("RGB", "Press START to begin")
                yield self._mjpeg_boundary(self._encode_jpeg(frame))
                await asyncio.sleep(1.0)
                continue

            if self._paused:
                if self._current_rgb_jpeg:
                    yield self._mjpeg_boundary(self._current_rgb_jpeg)
                else:
                    frame = self._stopped_frame("RGB", "PAUSED")
                    yield self._mjpeg_boundary(self._encode_jpeg(frame))
                await asyncio.sleep(0.5)
                continue

            test_frame = self._advance_frame()
            if test_frame is None:
                frame = self._no_images_frame("RGB")
                yield self._mjpeg_boundary(self._encode_jpeg(frame))
                await asyncio.sleep(settings.TEST_IMAGE_INTERVAL)
                continue

            filepath = test_frame.rgb_path

            if _is_video(filepath):
                # 영상 직접재생 모드 — backend는 더 이상 frame을 MJPEG로 재인코딩하지 않는다.
                # 프론트가 /test/active 로 메타 받아 <video src=/test/upload/file/{name}> 직접 재생,
                # backend는 background inference 만 돌리고 detection을 WS push.
                # MJPEG 채널엔 사용자가 우연히 <img>로 접근해도 인식 가능한 placeholder만 흘림.
                activated = self.activate_video_mode(filepath)
                if not activated:
                    # 메타 peek 실패 — 손상 파일. 다음 프레임으로 넘김.
                    await asyncio.sleep(0.5)
                    continue
                # active video가 끝날 때까지 placeholder만 흘림.
                # (영상이 끝나면 _video_inference_task가 done. 그래도 사용자가 STOP하기 전까지는
                #  같은 영상이 active로 유지 — 프론트가 seek/replay 할 수 있어야 하므로.)
                while self._playing and self._active_video_filename:
                    if self._paused:
                        pl = self._stopped_frame("RGB", "PAUSED")
                    else:
                        pl = self._watching_video_frame(self._active_video_filename)
                    yield self._mjpeg_boundary(self._encode_jpeg(pl))
                    await asyncio.sleep(1.0)
                # active video 해제됨 → 다음 frame으로
                continue

            frame = await asyncio.to_thread(cv2.imread, filepath)
            if frame is None:
                continue

            self._frame_counter += 1

            # 1) raw 이미지를 즉시 표시 — detection(특히 VLM)을 기다리느라 화면이 비는 것 방지.
            rgb_jpeg = self._encode_jpeg(frame)
            self._current_rgb_jpeg = rgb_jpeg
            raw_thermal_jpeg = await self._prepare_thermal_frame(test_frame, None)
            self._frame_version += 1
            yield self._mjpeg_boundary(rgb_jpeg)

            # 2) 이 이미지의 *모든* 하자를 검출(timeout). 검출이 끝날 때까지 다음 이미지로
            #    넘어가지 않으므로, 박스가 항상 "지금 보이는 그 이미지"에 정확히 동기화된다.
            detections = []
            if self._models_loaded:
                try:
                    detections = await asyncio.wait_for(
                        self._detect_all(frame, filepath),
                        timeout=settings.TEST_DETECT_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    print(f"[TestStream] 이미지 detection 타임아웃 — 박스 없이 진행: "
                          f"{os.path.basename(filepath)}")
                except Exception as e:
                    print(f"[TestStream] 이미지 detection 오류: {e}")

            # 3) 검출이 있으면 같은 이미지에 *모든 박스*를 구워서 갱신 표시(이미지 모드 라이브
            #    다중 박스) + 각 검출 카드 broadcast. (라이브 burned-in + 프론트 카드/클릭뷰 병행)
            if detections:
                annotated = frame
                for det in detections:
                    det["_rgb_snapshot"] = rgb_jpeg
                    if raw_thermal_jpeg is not None and test_frame.thermal_path is not None:
                        det["_thermal_snapshot"] = raw_thermal_jpeg
                    annotated = self._apply_live_overlay(annotated, det)
                annotated_jpeg = self._encode_jpeg(annotated)
                self._current_rgb_jpeg = annotated_jpeg
                self._frame_version += 1
                yield self._mjpeg_boundary(annotated_jpeg)
                for det in detections:
                    await self._broadcast_detection(det)

            # 4) 박스가 충분히 보이도록 대기 후 다음 이미지
            await asyncio.sleep(settings.TEST_IMAGE_INTERVAL)

    async def thermal_mjpeg_generator(self):
        """Thermal MJPEG 스트림. RGB 제너레이터와 프레임 버전으로 동기화."""
        last_version = 0
        while True:
            if not self._playing:
                frame = self._stopped_frame("THERMAL", "Press START to begin")
                yield self._mjpeg_boundary(self._encode_jpeg(frame))
                last_version = self._frame_version
                await asyncio.sleep(1.0)
                continue

            if self._paused:
                if self._current_thermal_jpeg:
                    yield self._mjpeg_boundary(self._current_thermal_jpeg)
                else:
                    frame = self._stopped_frame("THERMAL", "PAUSED")
                    yield self._mjpeg_boundary(self._encode_jpeg(frame))
                await asyncio.sleep(0.5)
                continue

            if self._frame_version == last_version:
                await asyncio.sleep(0.1)
                continue

            last_version = self._frame_version

            if self._current_thermal_jpeg:
                yield self._mjpeg_boundary(self._current_thermal_jpeg)
            else:
                frame = self._no_signal_frame("THERMAL")
                yield self._mjpeg_boundary(self._encode_jpeg(frame))

    async def _prepare_thermal_frame(
        self, test_frame: TestFrame, detection: Optional[dict] = None
    ) -> Optional[bytes]:
        """RGB와 동기화된 Thermal 프레임을 준비. detection이 있으면 오버레이 적용.

        반환: raw thermal JPEG (오버레이 없음). 호출자는 detection["_thermal_snapshot"]에
        붙여서 store_defect_frame으로 전달 — 프레임 드리프트(다음 프레임 JPEG에 이전
        bbox가 페어링되는 버그) 방지용. 라이브 스트림용 오버레이는 self._current_thermal_jpeg.
        """
        if test_frame.thermal_path is None:
            frame = self._no_signal_frame("THERMAL")
            jpeg = self._encode_jpeg(frame)
            self._current_thermal_jpeg = jpeg
            return jpeg

        ir_frame = await asyncio.to_thread(cv2.imread, test_frame.thermal_path)
        if ir_frame is None:
            frame = self._no_signal_frame("THERMAL")
            jpeg = self._encode_jpeg(frame)
            self._current_thermal_jpeg = jpeg
            return jpeg

        if len(ir_frame.shape) == 3:
            gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = ir_frame
        thermal_raw = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        raw_jpeg = self._encode_jpeg(thermal_raw)

        if detection:
            thermal_with_overlay = self._apply_live_overlay(thermal_raw, detection)
            self._current_thermal_jpeg = self._encode_jpeg(thermal_with_overlay)
        else:
            self._current_thermal_jpeg = raw_jpeg

        return raw_jpeg

    # ── 영상 직접재생 모드: 메타 peek + background inference ────────────────────
    def _peek_video_meta(self, filepath: str) -> Optional[Tuple[float, float, int, int]]:
        """cv2.VideoCapture로 fps/duration/width/height만 읽는다. 디코드는 하지 않음.
        실패 시 None — 호출자가 안전 폴백."""
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total / fps if fps > 0 else 0.0
            return (fps, duration, w, h)
        finally:
            cap.release()

    def activate_video_mode(self, filepath: str) -> bool:
        """업로드 영상 1개를 직접재생 모드로 전환. 메타 set + background inference task 발사.
        프론트가 /test/active 폴링/WS 으로 받아 <video> 로 직접 재생. 백엔드는 더 이상
        영상 프레임을 MJPEG로 재인코딩하지 않음 — Fly 1 vCPU 의 결정적 병목 제거."""
        meta = self._peek_video_meta(filepath)
        if meta is None:
            print(f"[TestStream] 영상 메타 peek 실패: {filepath}")
            return False
        fps, duration, w, h = meta
        self._active_video_filename = os.path.basename(filepath)
        self._active_video_fps = fps
        self._active_video_duration = duration
        self._active_video_frame_w = w
        self._active_video_frame_h = h
        # 이전 task 있으면 취소 후 새 task
        self._cancel_video_inference()
        self._video_inference_task = asyncio.create_task(self._video_inference_loop(filepath))
        print(f"[TestStream] 영상 직접재생 모드 진입: {self._active_video_filename} "
              f"({fps:.1f}fps, {duration:.1f}s, {w}x{h})")
        return True

    async def _video_inference_loop(self, filepath: str) -> None:
        """영상을 1회 end-to-end로 읽으며 sample마다 inference + WS broadcast.
        각 detection 에는 video_timestamp_sec(=frame_idx/fps) 가 포함되어 프론트가
        <video>.currentTime 과 동기화하여 SVG 오버레이를 띄울 수 있게 한다.
        play_state(_playing/_paused) 존중. 영상 끝나면 자연 종료."""
        # 콜드 스타트 시 모델 로드(10~20초)가 끝나기 전에 영상이 activate 될 수 있다.
        # 과거: 여기서 즉시 return → 영상은 재생되는데 검출이 영영 안 떠서 "로딩인지
        # 오류인지" 분간 안 되는 사고. 이제 로드 완료를 기다렸다가 추론을 시작한다.
        waited = 0.0
        while not self._models_loaded and self._playing and waited < 30.0:
            await asyncio.sleep(0.5)
            waited += 0.5
        if not self._models_loaded:
            print(f"[TestStream] 모델 미로드(대기 {waited:.0f}s 초과) — 영상 inference 스킵")
            return

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            cap.release()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 하이브리드(VLM) 비용 통제: VLM_KEYFRAME_INTERVAL_SEC 주기로만 샘플 추론.
        # (Live _vlm_keyframe_loop 와 동일 정책 — 영상 프레임 폭주로 VLM 일일상한/동시성 소진 방지.
        #  과거 fps/3(초당 3회)면 VLM 경로에서 즉시 쿼터 고갈.)
        kf = max(1.0, float(settings.VLM_KEYFRAME_INTERVAL_SEC))
        sample_interval = max(1, int(round(fps * kf)))
        frame_idx = 0

        try:
            while cap.isOpened():
                if not self._playing:
                    break
                if self._paused:
                    await asyncio.sleep(0.3)
                    continue

                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    video_t = frame_idx / fps if fps > 0 else 0.0
                    # 영상 경로는 tier=2 (M4 thermal U-Net + M6 PatchCore 제외).
                    # RGB 영상에 thermal 추론은 무의미하고 PatchCore는 무거워 60fps 동반 inference 시
                    # 1 vCPU 결정적 병목. 핵심 RGB 결함(M1/M2/M3/M5)만 충분히 잡힌다.
                    dets = await self._detect_all(frame, filepath, tier=2)
                    if dets:
                        # 같은 키프레임의 모든 하자를 동일 timestamp/스냅샷으로 broadcast
                        # → 프론트 DetectionOverlay 가 그 시점에 여러 박스를 동시 표시.
                        rgb_snap = self._encode_jpeg(frame)
                        fw, fh = frame.shape[1], frame.shape[0]
                        for det in dets:
                            if not det.get("bbox"):
                                continue
                            det["_rgb_snapshot"] = rgb_snap
                            det["_video_timestamp_sec"] = video_t
                            det["_frame_w"] = fw
                            det["_frame_h"] = fh
                            asyncio.create_task(self._broadcast_detection(det))

                frame_idx += 1
                # 이벤트 루프에 양보 — 1 vCPU 환경에서 다른 코루틴(MJPEG placeholder yield 등) 진행 보장
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[TestStream] 영상 inference 오류: {e}")
        finally:
            cap.release()
            print(f"[TestStream] 영상 inference 종료: {os.path.basename(filepath)} "
                  f"(샘플 {frame_idx // sample_interval}회)")

    # ── 이미지 detection 백그라운드 실행 + 브로드캐스트 ────────
    async def _detect_and_broadcast_image(
        self,
        frame: np.ndarray,
        filepath: str,
        rgb_snapshot: bytes,
        thermal_snapshot: Optional[bytes],
    ) -> None:
        """이미지 프레임 detection 을 MJPEG yield 경로 밖에서(백그라운드 태스크) 실행하고
        결과 카드를 WS 로 broadcast 한다. VLM 하이브리드가 느리거나 멈춰도 라이브 스트림은
        계속 흐르게 하는 핵심 분리점. timeout 으로 좀비 태스크를 방지한다.

        rgb_snapshot/thermal_snapshot: detection 발생 프레임의 raw JPEG 를 미리 굳혀둔 값.
        broadcast 가 지연되는 사이 _current_*_jpeg 가 다음 프레임으로 갱신되어 bbox/JPEG 짝이
        어긋나는 드리프트를 방지(store_defect_frame 으로 클릭 뷰가 정확한 프레임을 보여주도록)."""
        try:
            try:
                detections = await asyncio.wait_for(
                    self._detect_all(frame, filepath),
                    timeout=settings.TEST_DETECT_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                print(f"[TestStream] 이미지 detection 타임아웃"
                      f"({settings.TEST_DETECT_TIMEOUT_SEC}s) — 스킵: {os.path.basename(filepath)}")
                return
            if not detections:
                return
            # 한 프레임의 모든 하자를 각각 broadcast → 프론트가 여러 박스/카드를 동시 표시.
            # 동일 프레임 raw 스냅샷을 공유해 클릭 뷰 프레임 드리프트 방지.
            for detection in detections:
                detection["_rgb_snapshot"] = rgb_snapshot
                if thermal_snapshot is not None:
                    detection["_thermal_snapshot"] = thermal_snapshot
                await self._broadcast_detection(detection)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[TestStream] 이미지 detection 백그라운드 오류: {e}")
        finally:
            self._image_detect_inflight = max(0, self._image_detect_inflight - 1)

    # ── 추론 (결과만 반환, 브로드캐스트 하지 않음) ────────
    async def _detect(
        self, frame: np.ndarray, filepath: str, tier: int = 3,
    ) -> Optional[dict]:
        """실제 ONNX 추론 결과만 반환 (브로드캐스트는 별도).
        모델 미로드/검출 0건/예외 → None. mock 폴백으로 거짓 라벨을 만들지 않는다.

        tier (2026-05-12 신설):
          3 = 풀(M1+M2+M3+M4 thermal+M5 geom+M6 patchcore). 이미지 단발 추론 기본.
          2 = M4(thermal) + M6(patchcore) 제외. 영상 경로 기본 — RGB 영상에 thermal U-Net은
              무의미하고 PatchCore는 무거워 60fps 동시 inference 시 결정적 병목.
          1 = M1+M2 만. 경량 (필요 시).
        Why: 디렉토리명 기반 mock 라벨이 실제 추론 자리를 가로채면 입주자 신뢰 직결 사고.

        2026-06-09: TEST MODE 도 항상 하이브리드(ONNX 후보 → VLM 판정) 경로를 탄다.
        VLM 인프라 실패(키 미설정/쿼터/네트워크)일 때만 ONNX 단독으로 폴백 — 화면이
        빈 채로 멎지 않게. 하이브리드가 정상 동작하며 0건을 반환한 경우(VLM 기각 등)는
        그 판정을 존중하여 None."""
        if not self._models_loaded:
            return None
        try:
            return await self._detect_hybrid(frame, filepath)
        except Exception as e:
            print(f"[TestStream] 하이브리드 실패 — ONNX 단독 폴백: {e}")
            return await self._detect_real(frame, filepath, tier=tier)

    async def _detect_hybrid(self, frame: np.ndarray, filepath: str) -> Optional[dict]:
        """ONNX 후보 → VLM 판정(하이브리드). 보고서 등재 가능(listable) 검출 중 bbox 있는
        최고 신뢰 1건을 골라 기존 broadcast 스키마로 매핑. 인프라 실패는 호출부(_detect)가 폴백."""
        from app.services.hybrid_detector import detect_hybrid_async
        image_bytes = await asyncio.to_thread(self._encode_jpeg, frame)
        result = await detect_hybrid_async(image_bytes)

        # TEST MODE는 데모/검증용이라 CONFIRMED 뿐 아니라 REVIEW(VLM 단독·ONNX 단독)도 노출.
        # (is_listable 는 보고서 등재 기준이라 CONFIRMED-only → 그걸 쓰면 VLM recall 검출이 다 숨음)
        usable = [d for d in result.detections
                  if d.grade in ("CONFIRMED", "REVIEW") and d.bbox_xyxy and d.localization == "bbox"]
        if not usable:
            usable = [d for d in result.detections
                      if d.grade in ("CONFIRMED", "REVIEW") and d.bbox_xyxy]
        if not usable:
            return None
        # CONFIRMED 우선(더 신뢰), 없으면 REVIEW 중 최고 신뢰 1건
        confirmed = [d for d in usable if d.grade == "CONFIRMED"]
        det = max(confirmed or usable, key=lambda d: d.conf)

        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy[:4]]
        bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        image_crop_b64 = self._crop_to_base64(frame, x1, y1, x2, y2)
        conf_rounded = round(float(det.conf), 3)
        class_name = getattr(det, "class_", "") or ""
        class_en = getattr(det, "class_display_en", "") or class_name
        grade_ko = getattr(det, "grade_display_ko", "") or ""
        label = f"{det.code} {det.class_display_ko} ({conf_rounded*100:.0f}%·{grade_ko})".strip()

        return {
            "id": uuid.uuid4().hex,
            "bbox": bbox_dict,
            "label": label,
            "severity": det.severity or "HIGH",
            "image_crop": image_crop_b64,
            "confidence": conf_rounded,
            "filepath": filepath,
            "defect_info": {
                "area": det.area,
                "category_code": det.code,
                "defect_type": det.class_display_ko,
                "severity": det.severity,
                "defect_class": class_name,
                "defect_class_display_en": class_en,
                "defect_class_display_ko": det.class_display_ko,
            },
            "source": det.source,  # onnx+vlm | vlm | onnx
        }

    def _hybrid_det_to_dict(self, det, frame: np.ndarray, filepath: str) -> dict:
        """hybrid 검출 1건 → broadcast 스키마 dict. (_detect_hybrid 의 단건 구성과 동일,
        다중 검출에서 재사용하기 위해 분리)."""
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy[:4]]
        bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        image_crop_b64 = self._crop_to_base64(frame, x1, y1, x2, y2)
        conf_rounded = round(float(det.conf), 3)
        class_name = getattr(det, "class_", "") or ""
        class_en = getattr(det, "class_display_en", "") or class_name
        grade_ko = getattr(det, "grade_display_ko", "") or ""
        label = f"{det.code} {det.class_display_ko} ({conf_rounded*100:.0f}%·{grade_ko})".strip()
        return {
            "id": uuid.uuid4().hex,
            "bbox": bbox_dict,
            "label": label,
            "severity": det.severity or "HIGH",
            "image_crop": image_crop_b64,
            "confidence": conf_rounded,
            "filepath": filepath,
            "defect_info": {
                "area": det.area,
                "category_code": det.code,
                "defect_type": det.class_display_ko,
                "severity": det.severity,
                "defect_class": class_name,
                "defect_class_display_en": class_en,
                "defect_class_display_ko": det.class_display_ko,
            },
            "source": det.source,
            # 근거 감사로그(4-4) — 분쟁 증거 + 점검자 판단 보조. hybrid_detector 가 생성.
            "grade": getattr(det, "grade", None),
            "reasoning": getattr(det, "reasoning", None),
            "onnx_conf": getattr(det, "onnx_conf", None),
            "vlm_conf": getattr(det, "vlm_conf", None),
            "agreement": getattr(det, "agreement", None),
        }

    async def _detect_hybrid_all(self, frame: np.ndarray, filepath: str) -> List[dict]:
        """하이브리드 검출 — usable(CONFIRMED+REVIEW, bbox 보유) 전부를 dict 리스트로 반환.
        최고신뢰 1건만 고르던 _detect_hybrid 의 '다중 하자 동시 표시' 버전.
        신뢰도 내림차순 상위 _MAX_DEFECTS_PER_FRAME 건으로 제한(화면 혼잡 방지)."""
        from app.services.hybrid_detector import detect_hybrid_async
        image_bytes = await asyncio.to_thread(self._encode_jpeg, frame)
        result = await detect_hybrid_async(image_bytes)

        usable = [d for d in result.detections
                  if d.grade in ("CONFIRMED", "REVIEW") and d.bbox_xyxy and d.localization == "bbox"]
        if not usable:
            usable = [d for d in result.detections
                      if d.grade in ("CONFIRMED", "REVIEW") and d.bbox_xyxy]
        if not usable:
            return []
        usable.sort(key=lambda d: d.conf, reverse=True)
        usable = usable[: self._MAX_DEFECTS_PER_FRAME]
        return [self._hybrid_det_to_dict(d, frame, filepath) for d in usable]

    async def _detect_all(
        self, frame: np.ndarray, filepath: str, tier: int = 3,
    ) -> List[dict]:
        """한 프레임의 검출을 *전부* 반환 (다중 하자). 모델 미로드/0건/예외 → [].
        하이브리드(ONNX 후보 → VLM 판정) 다중 버전; 실패 시 ONNX 단독 단건으로 폴백."""
        if not self._models_loaded:
            return []
        try:
            return await self._detect_hybrid_all(frame, filepath)
        except Exception as e:
            print(f"[TestStream] 하이브리드(다중) 실패 — ONNX 단독 폴백: {e}")
            d = await self._detect_real(frame, filepath, tier=tier)
            return [d] if d else []

    def _detect_mock(self, frame: np.ndarray, filepath: str) -> dict:
        """[DEPRECATED — 호출되지 않음] 디렉토리명 기반 가짜 라벨.
        bbox는 random crop, 클래스는 폴더명, confidence는 random.uniform —
        실제 추론 자리에 끼어들면 거짓 검출이 노출되어 신뢰성 사고. 호출 차단됨.
        시연용으로 부활하려면 별도 명시 플래그로 분리할 것."""
        defect_info = None
        path_lower = filepath.replace("\\", "/").lower()
        for dir_key, info in _DIR_TO_DEFECT.items():
            if dir_key in path_lower:
                defect_info = info
                break
        if defect_info is None:
            defect_info = _PAIRED_DEFECT

        confidence = round(random.uniform(0.60, 0.95), 3)
        image_crop_b64, bbox_dict = self._generate_random_crop(frame)
        defect_id = uuid.uuid4().hex
        label = f"{defect_info['category_code']} {defect_info['defect_type']} ({confidence*100:.0f}%)"

        return {
            "id": defect_id,
            "bbox": bbox_dict,
            "label": label,
            "severity": defect_info["severity"],
            "image_crop": image_crop_b64,
            "confidence": confidence,
            "filepath": filepath,
            "defect_info": defect_info,
            "source": "test_mock",
        }

    # ── UI 노출 conf 게이트 ─────────────────
    # 모델 학습 conf threshold(M1~M3=0.25~0.30) 와 별개로, 사용자에게 카드를 띄울 때
    # 적용하는 UI 노출 임계값. OOD 입력(사람/외부 객체 등)에서 35~43% 저신뢰 검출이
    # "방수층 들뜸/코킹 누락" 같은 거짓 라벨로 노출되어 입주자 신뢰 사고 유발 가능.
    # Precision 우선([모든 하자 엄격·신뢰 우선] 정책) — 학습 자체는 안 건드림.
    # 단열은 미탐 비용이 더 크므로([단열 결함 더 엄격하게]) 더 낮은 cutoff 유지.
    #
    # 2026-05-12 강화: 코킹·표면·도색·스크래치는 색상/패턴 의존도가 높아 OOD(밈/사람/풍경
    # 영상)에서도 52~64% 신뢰도로 거짓 검출이 통과하는 사례 확인(test_mode 첨부 회귀 사고).
    # 학습 자체는 후속 사이클로 두고 UI 노출 임계값만 우선 끌어올린다.
    _UI_CONF_GATE_DEFAULT = 0.50
    _UI_CONF_GATE_INSULATION = 0.30
    # OOD-취약 클래스(가시광 색·패턴 dependent — 사람/풍경 영상에서 거짓 양성)
    _UI_CONF_GATE_OOD_FRAGILE = 0.75

    # 키워드 기반 매칭. class_ 내부명/한글표시명 둘 중 하나라도 걸리면 적용.
    # (substring 매칭이므로 caulking_indicator / caulking_defect / "코킹 누락" 등 모두 cover)
    _OOD_FRAGILE_KEYS = (
        "caulking", "코킹",
        "scratch", "찍힘", "스크래치",
        "paint_stain", "도색",
        "surface_defect", "표면 결함",
        "baseboard", "걸레받이",
        "pollution", "오염",
    )

    @classmethod
    def _ui_conf_gate(cls, class_: Optional[str], class_ko: Optional[str]) -> float:
        s = f"{class_ or ''} {class_ko or ''}".lower()
        if "insulation" in s or "단열" in s:
            return cls._UI_CONF_GATE_INSULATION
        for key in cls._OOD_FRAGILE_KEYS:
            if key in s:
                return cls._UI_CONF_GATE_OOD_FRAGILE
        return cls._UI_CONF_GATE_DEFAULT

    async def _detect_real(
        self, frame: np.ndarray, filepath: str, tier: int = 3,
    ) -> Optional[dict]:
        """실제 ONNX 추론. 첫 번째 검출 결과를 반환. 0건/미로드/예외/저신뢰 → None.
        tier는 _detect에서 위임 — 영상 경로는 tier=2(M4 thermal/M6 patchcore 제외).
        Why: 0건일 때 mock 폴백을 쓰면 모델이 못 본 곳에 가짜 bbox + 디렉토리명 라벨이
        그려져 사용자에게 거짓 검출이 노출됨. 안전 직결 — None이 정직한 답이다."""
        try:
            from app.services.inference_pipeline_20 import pipeline20
            if not pipeline20.is_loaded:
                return None

            result = await pipeline20.detect_async(frame, tier=tier)
            if result.defect_count == 0:
                return None

            det = result.detections[0]  # 첫 번째 검출
            # bbox 없는 검출은 노출하지 않음. random crop으로 채우면 무관한 영역에 박스가
            # 그려져 거짓 위치 표시가 되므로 안전 직결 위반.
            if not det.bbox_xyxy:
                print(f"[TestStream] 검출됐으나 bbox 없음 — 노출 차단 (filepath={filepath})")
                return None

            # UI 노출 conf 게이트(클래스별) — 학습 임계값과 무관하게 사용자 노출 보호.
            ui_gate = self._ui_conf_gate(det.class_, det.class_display_ko)
            if det.conf < ui_gate:
                return None
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            image_crop_b64 = self._crop_to_base64(frame, x1, y1, x2, y2)

            conf_rounded = round(det.conf, 3)
            label = f"{det.code} {det.class_display_ko} ({conf_rounded*100:.0f}%)"

            return {
                "id": uuid.uuid4().hex,
                "bbox": bbox_dict,
                "label": label,
                "severity": det.severity,
                "image_crop": image_crop_b64,
                "confidence": conf_rounded,
                "filepath": filepath,
                "defect_info": {
                    "area": getattr(det, "area", None),
                    "category_code": det.code,
                    "defect_type": det.class_display_ko,
                    "severity": det.severity,
                    "defect_class": det.class_,
                    "defect_class_display_en": det.class_display_en,
                    "defect_class_display_ko": det.class_display_ko,
                },
                "source": det.defect_source,
            }
        except Exception as e:
            print(f"[TestStream] 추론 오류 — mock 폴백 없이 None 반환: {e}")
            return None

    # ── 브로드캐스트 (결과를 WebSocket으로 전송) ────────
    async def _broadcast_detection(self, detection: dict) -> None:
        """검출 결과를 WebSocket으로 브로드캐스트 + defect frame 저장."""
        from app.core.ws_manager import ws_manager

        now_iso = datetime.now(timezone.utc).isoformat()
        info = detection.get("defect_info", _PAIRED_DEFECT)

        # 하자 시점 프레임 저장 (클릭 시 조회용).
        # _rgb_snapshot/_thermal_snapshot은 detection이 발생한 그 프레임의 raw JPEG.
        # 이걸 명시 전달해야 _current_*_jpeg(다음 프레임으로 이미 갱신됨)와의 드리프트 방지.
        self.store_defect_frame(
            detection["id"], bbox=detection["bbox"],
            label=detection["label"], severity=detection["severity"],
            rgb_jpeg=detection.get("_rgb_snapshot"),
            thermal_jpeg=detection.get("_thermal_snapshot"),
        )

        # 브로드캐스트 데이터 구성
        data = {
            "id": detection["id"],
            "timestamp": now_iso,
            "area": info.get("area"),
            "category_code": info.get("category_code"),
            "defect_type": info.get("defect_type"),
            "severity": detection["severity"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "image_crop": detection["image_crop"],
            "defect_source": detection.get("source", "test_mock"),
            "defect_class": info.get("defect_class", "unknown"),
            "defect_class_display_en": info.get("defect_class_display_en", ""),
            "defect_class_display_ko": info.get("defect_class_display_ko", info.get("defect_type", "")),
            "frame_id": self._frame_counter,
            "source_file": os.path.basename(detection["filepath"]),
            "mode": self._detection_mode,
            # 검출 소스 채널 — 프론트가 일치하는 피드(RGB→Drone1, thermal→Drone2)에만
            # 인스펙션 뷰를 띄우도록. 영상 경로(tier=2)는 RGB 추론이라 기본 'rgb'.
            "source_channel": detection.get("_source_channel", "rgb"),
            # 근거 감사로그(4-4) — 분쟁 증거 + 점검자 판단 보조.
            "grade": detection.get("grade"),
            "reasoning": detection.get("reasoning"),
            "onnx_conf": detection.get("onnx_conf"),
            "vlm_conf": detection.get("vlm_conf"),
            "agreement": detection.get("agreement"),
        }
        # 영상 직접재생 모드일 때만 채워지는 동기화용 메타.
        # 프론트 <video>.currentTime ↔ video_timestamp_sec 비교로 SVG 오버레이 동기화.
        if "_video_timestamp_sec" in detection:
            data["video_timestamp_sec"] = detection["_video_timestamp_sec"]
        if "_frame_w" in detection and "_frame_h" in detection:
            data["frame_w"] = detection["_frame_w"]
            data["frame_h"] = detection["_frame_h"]
        await ws_manager.broadcast("defects", {"type": "defect.new", "data": data})

    # ── 라이브 오버레이 (numpy 프레임에 직접 그리기) ────────
    def _apply_live_overlay(
        self, frame: np.ndarray, detection: Optional[dict]
    ) -> np.ndarray:
        """현재 감지 모드에 따라 numpy 프레임에 오버레이를 그린다."""
        if detection is None:
            return frame

        bbox = detection.get("bbox")
        label = detection.get("label", "")
        severity = detection.get("severity", "HIGH")

        if not bbox or not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        x1 = max(0, int(bbox["x1"]))
        y1 = max(0, int(bbox["y1"]))
        x2 = min(w, int(bbox["x2"]))
        y2 = min(h, int(bbox["y2"]))

        severity_colors_bgr = {
            "HIGH": (0, 0, 255),
            "MED":  (0, 140, 255),
            "LOW":  (0, 200, 255),
        }
        color = severity_colors_bgr.get(severity, (0, 0, 255))
        color_rgb = (color[2], color[1], color[0])

        if self._detection_mode == "detection":
            # ── 객체감지 모드 ──
            # 반투명 마스크
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

            # 윤곽 강조
            if y2 > y1 and x2 > x1:
                roi = out[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_shifted = cnt.copy()
                    cnt_shifted[:, :, 0] += x1
                    cnt_shifted[:, :, 1] += y1
                    cv2.drawContours(out, [cnt_shifted], -1, color, 1)

            # 코너 마커
            bl = max(12, min(x2 - x1, y2 - y1) // 5)
            t = 3
            cv2.line(out, (x1, y1), (x1 + bl, y1), color, t)
            cv2.line(out, (x1, y1), (x1, y1 + bl), color, t)
            cv2.line(out, (x2, y1), (x2 - bl, y1), color, t)
            cv2.line(out, (x2, y1), (x2, y1 + bl), color, t)
            cv2.line(out, (x1, y2), (x1 + bl, y2), color, t)
            cv2.line(out, (x1, y2), (x1, y2 - bl), color, t)
            cv2.line(out, (x2, y2), (x2 - bl, y2), color, t)
            cv2.line(out, (x2, y2), (x2, y2 - bl), color, t)
        else:
            # ── BBox 모드 ──
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # PIL로 한글 라벨 렌더링 (두 모드 공통)
        if label:
            pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            if self._detection_mode == "detection":
                font = _get_font(15)
                font_sm = _get_font(12)
                sev_text = severity
                sev_bbox_ = draw.textbbox((0, 0), sev_text, font=font_sm)
                sw, sh = sev_bbox_[2] - sev_bbox_[0], sev_bbox_[3] - sev_bbox_[1]
                lb = draw.textbbox((0, 0), label, font=font)
                lw_, lh_ = lb[2] - lb[0], lb[3] - lb[1]
                tw = sw + 10 + lw_ + 16
                bh = max(sh, lh_) + 10
                bx, by = x1, y1 - bh - 4
                if by < 0:
                    by = y2 + 4
                draw.rectangle([bx, by, bx + tw, by + bh], fill=(30, 30, 30))
                draw.rectangle([bx + 2, by + 2, bx + sw + 10, by + bh - 2], fill=color_rgb)
                draw.text((bx + 5, by + (bh - sh) // 2 - 1), sev_text, fill=(255, 255, 255), font=font_sm)
                draw.text((bx + sw + 14, by + (bh - lh_) // 2 - 1), label, fill=(255, 255, 255), font=font)
            else:
                font = _get_font(16)
                tb = draw.textbbox((0, 0), label, font=font)
                tw_, th_ = tb[2] - tb[0], tb[3] - tb[1]
                lx, ly = x1, y1 - th_ - 8
                if ly < 0:
                    ly = y2 + 4
                draw.rectangle([lx, ly, lx + tw_ + 8, ly + th_ + 6], fill=(255, 0, 0))
                draw.text((lx + 4, ly + 2), label, fill=(255, 255, 255), font=font)

            out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return out

    # ── 이미지 크롭 유틸리티 ────────────────────
    @staticmethod
    def _generate_random_crop(frame: np.ndarray) -> Tuple[str, dict]:
        """프레임에서 랜덤 영역을 크롭하여 (base64_data_url, bbox_dict)를 반환."""
        h, w = frame.shape[:2]
        crop_w = random.randint(max(1, w // 5), max(2, w // 3))
        crop_h = random.randint(max(1, h // 5), max(2, h // 3))
        x1 = random.randint(0, max(0, w - crop_w))
        y1 = random.randint(0, max(0, h - crop_h))
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)

        crop = frame[y1:y2, x1:x2]
        thumb = cv2.resize(crop, (112, 112))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return (
            f"data:image/jpeg;base64,{b64}",
            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        )

    @staticmethod
    def _crop_to_base64(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
        """지정 bbox 영역을 크롭 → base64 data URL로 인코딩."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            # 유효하지 않은 bbox → 전체 프레임 축소
            crop = frame
        else:
            crop = frame[y1:y2, x1:x2]
        thumb = cv2.resize(crop, (112, 112))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # ── 유틸리티 ────────────────────────────────
    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.MJPEG_JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _mjpeg_boundary(jpeg_bytes: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"

    @staticmethod
    def _no_images_frame(label: str = "RGB", width: int = 640, height: int = 480) -> np.ndarray:
        frame = np.full((height, width, 3), 18, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"TEST MODE - {label}", (width // 2 - 150, height // 2 - 20),
                     font, 0.7, (100, 100, 100), 1)
        cv2.putText(frame, "No test images available", (width // 2 - 155, height // 2 + 15),
                     font, 0.55, (80, 80, 80), 1)
        return frame

    @staticmethod
    def _no_signal_frame(label: str = "THERMAL", width: int = 640, height: int = 480) -> np.ndarray:
        """쌍 데이터가 없을 때 표시할 No Signal 프레임."""
        frame = np.full((height, width, 3), 12, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "No Signal", (width // 2 - 80, height // 2 - 10),
                     font, 0.9, (60, 60, 60), 2)
        cv2.putText(frame, f"{label} - No paired data", (width // 2 - 120, height // 2 + 25),
                     font, 0.45, (50, 50, 50), 1)
        return frame

    @staticmethod
    def _watching_video_frame(filename: str, width: int = 640, height: int = 480) -> np.ndarray:
        """영상 직접재생 모드 placeholder — <img>에서 접근하면 보이는 안내 화면.
        실제 사용자는 <video> 태그로 직접 재생 중. 이 frame은 fallback/디버그용."""
        frame = np.full((height, width, 3), 14, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "DIRECT VIDEO MODE", (width // 2 - 145, height // 2 - 30),
                     font, 0.7, (80, 140, 180), 2)
        cv2.putText(frame, f"file: {filename[:38]}", (24, height // 2 + 10),
                     font, 0.5, (60, 60, 60), 1)
        cv2.putText(frame, "Frontend plays this via <video> directly.",
                    (24, height // 2 + 35), font, 0.45, (50, 50, 50), 1)
        return frame

    @staticmethod
    def _stopped_frame(label: str = "RGB", msg: str = "STOPPED", width: int = 640, height: int = 480) -> np.ndarray:
        """정지/일시중지 상태 프레임."""
        frame = np.full((height, width, 3), 15, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"TEST MODE - {label}", (width // 2 - 150, height // 2 - 30),
                     font, 0.7, (80, 80, 80), 1)
        cv2.putText(frame, msg, (width // 2 - 100, height // 2 + 15),
                     font, 0.8, (100, 100, 100), 2)
        return frame

    def stop(self) -> None:
        self.stop_playback()
        self._frame_counter = 0
        print("[TestStream] 테스트 모드 종료")


# ── 모듈 레벨 싱글톤 ─────────────────────────
test_stream_service = TestStreamService()
