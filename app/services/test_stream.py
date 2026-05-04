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
        self._frame_counter: int = 0
        self._scanned: bool = False

        # ── 하자별 프레임 저장소 (클릭 시 해당 시점 프레임 조회용) ──
        self._defect_frames: OrderedDict[str, Tuple[bytes, Optional[bytes]]] = OrderedDict()
        self._MAX_DEFECT_FRAMES = 200

        # ── 감지 시각화 모드 ──
        self._detection_mode: str = "bbox"  # 'bbox' | 'detection'

        # ── 재생 상태 ──
        self._playing: bool = False
        self._paused: bool = False

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
        print("[TestStream] ▶ 재생 시작")

    def pause_playback(self) -> None:
        self._paused = True
        print("[TestStream] ⏸ 일시중지")

    def resume_playback(self) -> None:
        self._paused = False
        print("[TestStream] ▶ 재생 재개")

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
        print("[TestStream] ⏹ 정지")

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
    async def load_models(self) -> dict:
        if self._models_loaded:
            return {"status": "already_loaded"}
        try:
            from app.services.inference_pipeline_20 import pipeline20
            await asyncio.to_thread(pipeline20.load_models)
            self._models_loaded = pipeline20.is_loaded
            print(f"[TestStream] 20종 파이프라인 로드: loaded={self._models_loaded}")
            return {"status": "loaded" if self._models_loaded else "partial"}
        except Exception as e:
            print(f"[TestStream] 모델 로드 실패 (목업 폴백 사용): {e}")
            return {"status": "fallback_mock", "error": str(e)}

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
        upload_dir = settings.TEST_UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        saved = 0
        total_size = 0
        for upload_file in files:
            ext = Path(upload_file.filename).suffix.lower()
            if ext not in ALL_EXTENSIONS:
                continue
            safe_name = f"{uuid.uuid4().hex[:8]}_{upload_file.filename}"
            dest = os.path.join(upload_dir, safe_name)
            content = await upload_file.read()
            with open(dest, "wb") as f:
                f.write(content)
            self._uploaded_files.append(dest)
            saved += 1
            total_size += len(content)
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
    ) -> None:
        """현재 RGB/Thermal raw JPEG + 메타데이터를 저장.
        조회 시 모드(bbox/detection)에 따라 시각화를 적용."""
        self._defect_frames[defect_id] = {
            "rgb": self._current_rgb_jpeg,
            "thermal": self._current_thermal_jpeg,
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
        channel: 'rgb' | 'thermal', mode: 'bbox' | 'detection'."""
        data = self._defect_frames.get(defect_id)
        if data is None:
            return None
        jpeg = data["rgb"] if channel == "rgb" else data["thermal"]
        bbox = data["bbox"]
        label = data["label"]
        severity = data["severity"]
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
        """RGB MJPEG 스트림. 추론 → 오버레이 → yield → broadcast."""
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
                async for boundary in self._stream_video_frames(filepath):
                    yield boundary
                continue

            frame = await asyncio.to_thread(cv2.imread, filepath)
            if frame is None:
                continue

            self._frame_counter += 1

            # 1) 추론 (브로드캐스트 없이 결과만 반환)
            detection = await self._detect(frame, filepath)

            # 2) RGB 프레임에 오버레이 그리기
            annotated = self._apply_live_overlay(frame, detection)
            rgb_jpeg = self._encode_jpeg(annotated)
            self._current_rgb_jpeg = rgb_jpeg

            # 3) Thermal 프레임 준비 + 오버레이
            await self._prepare_thermal_frame(test_frame, detection)
            self._frame_version += 1

            # 4) 오버레이된 프레임 전송 (이미지가 먼저 보임)
            yield self._mjpeg_boundary(rgb_jpeg)

            # 5) 렌더링 여유 후 하자 브로드캐스트
            await asyncio.sleep(0.5)
            if detection:
                await self._broadcast_detection(detection)

            await asyncio.sleep(max(0, settings.TEST_IMAGE_INTERVAL - 0.5))

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
    ) -> None:
        """RGB와 동기화된 Thermal 프레임을 준비. detection이 있으면 오버레이 적용."""
        if test_frame.thermal_path is None:
            frame = self._no_signal_frame("THERMAL")
            self._current_thermal_jpeg = self._encode_jpeg(frame)
            return

        ir_frame = await asyncio.to_thread(cv2.imread, test_frame.thermal_path)
        if ir_frame is None:
            frame = self._no_signal_frame("THERMAL")
            self._current_thermal_jpeg = self._encode_jpeg(frame)
            return

        if len(ir_frame.shape) == 3:
            gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = ir_frame
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        # Thermal에도 동일한 오버레이 적용
        if detection:
            thermal = self._apply_live_overlay(thermal, detection)

        self._current_thermal_jpeg = self._encode_jpeg(thermal)

    # ── 영상 프레임 스트리밍 ────────────────────
    async def _stream_video_frames(self, filepath: str):
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / min(fps, 10.0)

        while cap.isOpened():
            if not self._playing:
                break
            if self._paused:
                if self._current_rgb_jpeg:
                    yield self._mjpeg_boundary(self._current_rgb_jpeg)
                await asyncio.sleep(0.5)
                continue

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break

            self._frame_counter += 1

            detection = None
            if self._frame_counter % 3 == 0:
                detection = await self._detect(frame, filepath)

            annotated = self._apply_live_overlay(frame, detection) if detection else frame
            rgb_jpeg = self._encode_jpeg(annotated)
            self._current_rgb_jpeg = rgb_jpeg

            yield self._mjpeg_boundary(rgb_jpeg)

            if detection:
                await self._broadcast_detection(detection)

            await asyncio.sleep(frame_interval)

        cap.release()

    # ── 추론 (결과만 반환, 브로드캐스트 하지 않음) ────────
    async def _detect(self, frame: np.ndarray, filepath: str) -> Optional[dict]:
        """추론 또는 목업 생성. 결과 dict를 반환 (브로드캐스트는 별도)."""
        if self._models_loaded:
            return await self._detect_real(frame, filepath)
        return self._detect_mock(frame, filepath)

    def _detect_mock(self, frame: np.ndarray, filepath: str) -> dict:
        """목업 하자 데이터 생성 (브로드캐스트 없이 반환)."""
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

    async def _detect_real(self, frame: np.ndarray, filepath: str) -> Optional[dict]:
        """실제 ONNX 추론. 첫 번째 검출 결과를 반환."""
        try:
            from app.services.inference_pipeline_20 import pipeline20
            if not pipeline20.is_loaded:
                return self._detect_mock(frame, filepath)

            result = await pipeline20.detect_async(frame, tier=3)
            if result.defect_count == 0:
                return self._detect_mock(frame, filepath)

            det = result.detections[0]  # 첫 번째 검출
            bbox_dict = None
            image_crop_b64 = None
            if det.bbox_xyxy:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                image_crop_b64 = self._crop_to_base64(frame, x1, y1, x2, y2)
            else:
                image_crop_b64, bbox_dict = self._generate_random_crop(frame)

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
            print(f"[TestStream] 추론 오류: {e}")
            return self._detect_mock(frame, filepath)

    # ── 브로드캐스트 (결과를 WebSocket으로 전송) ────────
    async def _broadcast_detection(self, detection: dict) -> None:
        """검출 결과를 WebSocket으로 브로드캐스트 + defect frame 저장."""
        from app.core.ws_manager import ws_manager

        now_iso = datetime.now(timezone.utc).isoformat()
        info = detection.get("defect_info", _PAIRED_DEFECT)

        # raw 프레임 저장 (클릭 시 조회용)
        self.store_defect_frame(
            detection["id"], bbox=detection["bbox"],
            label=detection["label"], severity=detection["severity"],
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
        }
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
