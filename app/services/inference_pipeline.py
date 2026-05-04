# =============================================
# app/services/inference_pipeline.py
# 역할: 3-모델 통합 추론 오케스트레이터 (싱글톤)
#       - YOLOv8s crack_moisture (Crack, Moisture)
#       - YOLOv8s delamination
#       - ResNet50 wallpaper (19 classes, good=Burst 포함)
#       - detect_defects(image) → DetectionResult (신규 포맷)
#       - detect_defects_legacy() → 기존 DetectionResult dataclass 리스트
#       - severity 자동 계산 (HIGH/MED/LOW/null)
#
# 입력 타입 지원: bytes, numpy.ndarray, PIL.Image.Image, str(경로)
# 주의: API 응답은 xyxy(픽셀), DB 저장용은 xywhn(정규화) — 변환은 호출자 책임
# =============================================

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np

from app.config import settings
from app.schemas.detection import (
    DetectionResult,
    ImageShape,
    ModelsLoadedStatus,
    Top3Prediction,
    WallpaperPrediction,
    YoloDetection,
)
from app.services.defect_taxonomy import (
    WALLPAPER_SEVERE_CLASSES,
    get_display_names,
    map_to_legacy,
    xyxy_to_xywhn,
)
from app.services.wallpaper_classifier import wallpaper_classifier


ImageInput = Union[bytes, np.ndarray, "object", str]  # "object" = PIL.Image


# ── 레거시 호환용 DataClass ─────────────────────
# 기존 [app/services/yolo_inference.py](yolo_inference.py)의 DetectionResult와
# 동일 필드 구조. [app/services/defect_processor.py](defect_processor.py) 등
# 기존 호출자가 깨지지 않도록 shim에서 이 타입을 반환한다.
@dataclass
class LegacyDetection:
    class_id: int
    class_name: str
    category_code: str
    defect_type: str
    area: str
    severity: str
    confidence: float
    bbox_x: float   # xywhn cx
    bbox_y: float   # xywhn cy
    bbox_w: float   # xywhn w
    bbox_h: float   # xywhn h
    raw: Optional[dict] = None


class InferencePipeline:
    """
    3-모델 추론 오케스트레이터. 서버 전역 단 하나의 싱글톤.
    yolo_inference.yolo_service는 이 객체를 참조만 한다 — 모델 로드 중복 금지.
    """

    def __init__(self):
        self._yolo_thermal = None       # crack_moisture 모델
        self._yolo_delam = None         # delamination 모델
        self._device: Optional[str] = None
        self._conf_threshold: float = 0.25
        self._wallpaper_conf_threshold: float = 0.35
        self._wallpaper_margin_threshold: float = 0.15
        self._loaded = False

    # ── 상태 조회 ────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device or "cpu"

    @property
    def models_loaded(self) -> ModelsLoadedStatus:
        return ModelsLoadedStatus(
            yolo_thermal=self._yolo_thermal is not None,
            yolo_delam=self._yolo_delam is not None,
            wallpaper=wallpaper_classifier.is_loaded,
        )

    # ── 모델 로드 ────────────────────────────────
    def load_models(self) -> None:
        """
        3개 가중치 로드. 서버 시작 시 한 번만 호출 (재진입 시 스킵).

        Raises:
            FileNotFoundError: 가중치 파일 중 하나라도 없음
        """
        if self._loaded:
            print("[Pipeline] 모델 이미 로드됨 — 스킵")
            return

        from ultralytics import YOLO
        import torch

        weights_dir = settings.AEROINSPECT_WEIGHTS_DIR
        thermal_path = os.path.join(weights_dir, settings.YOLO_THERMAL_WEIGHTS)
        delam_path = os.path.join(weights_dir, settings.YOLO_DELAM_WEIGHTS)
        wallpaper_path = os.path.join(weights_dir, settings.WALLPAPER_WEIGHTS)

        # 파일 존재 검증
        missing: List[str] = []
        for label, path in [
            ("yolo_thermal", thermal_path),
            ("yolo_delam", delam_path),
            ("wallpaper", wallpaper_path),
        ]:
            if not os.path.exists(path):
                missing.append(f"  {label}: {path}")
        if missing:
            raise FileNotFoundError(
                "[Pipeline] 가중치 파일 누락:\n" + "\n".join(missing)
            )

        # 디바이스 선택
        device_pref = settings.DEVICE.lower()
        if device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_pref

        # YOLO 2개 로드
        self._yolo_thermal = YOLO(thermal_path)
        self._yolo_delam = YOLO(delam_path)
        print(f"[Pipeline] YOLO thermal 로드: {thermal_path}")
        print(f"[Pipeline] YOLO delam 로드:   {delam_path}")

        # ResNet50 벽지 분류기 로드 (싱글톤)
        wallpaper_classifier.load_model(wallpaper_path, device_pref=device_pref)

        self._conf_threshold = float(settings.YOLO_CONF_THRESHOLD)
        self._wallpaper_conf_threshold = float(settings.WALLPAPER_CONF_THRESHOLD)
        self._wallpaper_margin_threshold = float(settings.WALLPAPER_MARGIN_THRESHOLD)
        self._loaded = True

        print(
            f"[Pipeline] 3-모델 로드 완료: device={self._device}, "
            f"yolo_conf={self._conf_threshold}, "
            f"wallpaper_conf={self._wallpaper_conf_threshold}, "
            f"wallpaper_margin={self._wallpaper_margin_threshold}"
        )

    # ── 입력 정규화 ───────────────────────────────
    @staticmethod
    def _to_bgr_ndarray(image: ImageInput, is_rgb: bool = False) -> np.ndarray:
        """bytes / ndarray / PIL / path → BGR ndarray로 정규화."""
        # bytes (multipart 업로드 원본)
        if isinstance(image, (bytes, bytearray)):
            arr = np.frombuffer(image, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("이미지 디코딩 실패 (지원되지 않는 포맷).")
            return frame

        # 파일 경로
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"이미지 파일 없음: {image}")
            frame = cv2.imread(image, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"이미지 로드 실패: {image}")
            return frame

        # numpy 배열
        if isinstance(image, np.ndarray):
            frame = image
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # 입력이 RGB였다면 BGR로 변환 (OpenCV는 BGR 기준)
            if is_rgb and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame

        # PIL.Image
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                arr = np.array(image.convert("RGB"))  # RGB
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(f"지원되지 않는 이미지 타입: {type(image)}")

    # ── 메인 추론 엔트리포인트 ────────────────────
    def detect(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
        is_rgb: bool = False,
    ) -> DetectionResult:
        """
        동기 추론 (호출자는 asyncio.to_thread로 감쌀 것).

        Args:
            image: bytes / ndarray / PIL.Image / 파일경로
            conf_threshold: YOLO 신뢰도 임계값 (None이면 settings 값)
            is_rgb: ndarray 입력이 RGB인지 (기본 False = BGR)

        Returns:
            DetectionResult (신규 포맷)
        """
        if not self._loaded:
            raise RuntimeError("[Pipeline] 모델이 로드되지 않았습니다. load_models() 호출 필요.")

        conf = conf_threshold if conf_threshold is not None else self._conf_threshold
        frame_bgr = self._to_bgr_ndarray(image, is_rgb=is_rgb)
        h, w = frame_bgr.shape[:2]

        # YOLO 2개 순차 추론
        yolo_thermal_dets = self._run_yolo(self._yolo_thermal, frame_bgr, conf)
        yolo_delam_dets = self._run_yolo(self._yolo_delam, frame_bgr, conf)

        # ResNet 벽지 분류 (입력은 RGB 필요)
        wallpaper_pred: Optional[WallpaperPrediction] = None
        if wallpaper_classifier.is_loaded:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            wallpaper_pred = self._run_wallpaper(rgb)

        # 집계
        has_yolo = bool(yolo_thermal_dets or yolo_delam_dets)
        defect_count = len(yolo_thermal_dets) + len(yolo_delam_dets)
        if wallpaper_pred and wallpaper_pred.is_confident:
            defect_count += 1
        has_defect = has_yolo or (wallpaper_pred is not None and wallpaper_pred.is_confident)

        severity = self._compute_severity(has_yolo, wallpaper_pred)

        return DetectionResult(
            yolo_thermal=yolo_thermal_dets,
            yolo_delam=yolo_delam_dets,
            wallpaper_cls=wallpaper_pred,
            severity=severity,
            has_defect=has_defect,
            defect_count=defect_count,
            image_shape=ImageShape(width=w, height=h),
        )

    async def detect_async(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
        is_rgb: bool = False,
    ) -> DetectionResult:
        """비동기 래퍼 — 동기 추론을 스레드 풀에서 실행."""
        return await asyncio.to_thread(self.detect, image, conf_threshold, is_rgb)

    # ── YOLO 추론 ───────────────────────────────
    def _run_yolo(
        self,
        model,
        frame_bgr: np.ndarray,
        conf: float,
    ) -> List[YoloDetection]:
        """단일 YOLO 모델에서 탐지 → YoloDetection 리스트로 변환."""
        if model is None:
            return []

        results = model.predict(
            frame_bgr,
            conf=conf,
            verbose=False,
            device=self._device,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = result.names  # {class_id: class_name}
        detections: List[YoloDetection] = []

        xyxy_arr = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        conf_arr = result.boxes.conf.cpu().numpy()  # [N]
        cls_arr = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(cls_arr)):
            class_id = int(cls_arr[i])
            class_name = names.get(class_id, f"unknown_{class_id}")
            confidence = float(conf_arr[i])
            xyxy = [float(v) for v in xyxy_arr[i].tolist()]
            display_en, display_ko = get_display_names(class_name)

            detections.append(YoloDetection(
                **{
                    "class": class_name,
                    "class_display_en": display_en,
                    "class_display_ko": display_ko,
                    "conf": confidence,
                    "bbox_xyxy": xyxy,
                }
            ))
        return detections

    # ── 벽지 분류 ───────────────────────────────
    def _run_wallpaper(self, image_rgb: np.ndarray) -> WallpaperPrediction:
        """ResNet50 분류 → WallpaperPrediction (top1 + top3 + is_confident).

        is_confident 규칙 (val_acc 54% 대응):
          - top1_conf >= WALLPAPER_CONF_THRESHOLD (top1 절대 신뢰도)
          - AND (top1_conf - top2_conf) >= WALLPAPER_MARGIN_THRESHOLD (top2와 분리도)
          두 조건 모두 만족해야 confident. 근소차 예측은 모호로 처리.
        """
        top1_class, top1_conf, top3_raw = wallpaper_classifier.classify(image_rgb)
        top1_en, top1_ko = get_display_names(top1_class)

        top3: List[Top3Prediction] = []
        for cname, conf in top3_raw:
            en, ko = get_display_names(cname)
            top3.append(Top3Prediction(
                **{"class": cname, "class_display_en": en, "class_display_ko": ko, "conf": conf}
            ))

        top2_conf = top3_raw[1][1] if len(top3_raw) >= 2 else 0.0
        margin = top1_conf - top2_conf
        is_confident = (
            top1_conf >= self._wallpaper_conf_threshold
            and margin >= self._wallpaper_margin_threshold
        )

        return WallpaperPrediction(
            top1_class=top1_class,
            top1_class_display_en=top1_en,
            top1_class_display_ko=top1_ko,
            top1_conf=top1_conf,
            is_confident=is_confident,
            top3=top3,
        )

    # ── severity 규칙 ──────────────────────────
    @staticmethod
    def _compute_severity(
        has_yolo: bool,
        wallpaper_pred: Optional[WallpaperPrediction],
    ) -> Optional[str]:
        """
        severity 자동 계산.
          - YOLO thermal/delam 탐지 있음 → HIGH (구조·단열·방수)
          - 벽지 confident + 심각 클래스(Mold/Damage/Exploded/Defective_Joint/good) → MED
          - 벽지 confident + 그 외 → LOW
          - 그 외 (벽지 신뢰도 부족 등) → null (판단 보류)
        """
        if has_yolo:
            return "HIGH"
        if wallpaper_pred and wallpaper_pred.is_confident:
            if wallpaper_pred.top1_class in WALLPAPER_SEVERE_CLASSES:
                return "MED"
            return "LOW"
        return None


# ── 모듈 레벨 싱글톤 ─────────────────────────
pipeline = InferencePipeline()


# ── 공개 API (모듈 함수) ─────────────────────
def load_models() -> None:
    """서버 시작 시 한 번만 호출 — 재진입 시 스킵."""
    pipeline.load_models()


def detect_defects(
    image: ImageInput,
    conf_threshold: float = 0.25,
    is_rgb: bool = False,
) -> DetectionResult:
    """
    프롬프트 스펙의 공개 API.
    동기 — async 컨텍스트에서는 detect_defects_async()를 사용.
    """
    return pipeline.detect(image, conf_threshold=conf_threshold, is_rgb=is_rgb)


async def detect_defects_async(
    image: ImageInput,
    conf_threshold: float = 0.25,
    is_rgb: bool = False,
) -> DetectionResult:
    """비동기 래퍼 — 블로킹 추론을 스레드 풀로."""
    return await pipeline.detect_async(image, conf_threshold=conf_threshold, is_rgb=is_rgb)


# ── 레거시 포맷 변환 (기존 호출자용) ─────────
def detect_defects_legacy(
    image: ImageInput,
    conf_threshold: float = 0.25,
    is_rgb: bool = False,
) -> List[LegacyDetection]:
    """
    기존 [defect_processor.py](defect_processor.py) 등이 기대하는 포맷으로 반환.
    새 DetectionResult를 받아 A-E taxonomy로 매핑한 LegacyDetection 리스트 생성.
    매핑 없는 클래스는 스킵 (A-E 없으면 기존 DB 컬럼에 못 들어감).
    """
    result = pipeline.detect(image, conf_threshold=conf_threshold, is_rgb=is_rgb)
    img_w = result.image_shape.width
    img_h = result.image_shape.height

    legacy_list: List[LegacyDetection] = []

    def _push(source: str, class_name: str, conf: float, xyxy: List[float]) -> None:
        area, code, dtype = map_to_legacy(source, class_name)
        if area is None or code is None:
            return  # 레거시 체계로 매핑 불가 → 스킵
        cx, cy, bw, bh = xyxy_to_xywhn(xyxy, img_w, img_h)
        severity = "HIGH" if source in ("yolo_thermal", "yolo_delam") else (
            "MED" if class_name in WALLPAPER_SEVERE_CLASSES else "LOW"
        )
        legacy_list.append(LegacyDetection(
            class_id=0,
            class_name=class_name,
            category_code=code,
            defect_type=dtype,
            area=area,
            severity=severity,
            confidence=conf,
            bbox_x=cx, bbox_y=cy, bbox_w=bw, bbox_h=bh,
            raw={"source": source, "xyxy": xyxy, "image_shape": {"w": img_w, "h": img_h}},
        ))

    for det in result.yolo_thermal:
        _push("yolo_thermal", det.class_, det.conf, det.bbox_xyxy)
    for det in result.yolo_delam:
        _push("yolo_delam", det.class_, det.conf, det.bbox_xyxy)
    # 벽지는 bbox 없으므로 전체 프레임을 bbox로 간주 (cx=cy=0.5, w=h=1.0)
    if result.wallpaper_cls and result.wallpaper_cls.is_confident:
        cname = result.wallpaper_cls.top1_class
        conf = result.wallpaper_cls.top1_conf
        area, code, dtype = map_to_legacy("wallpaper", cname)
        if area is not None and code is not None:
            severity = "MED" if cname in WALLPAPER_SEVERE_CLASSES else "LOW"
            legacy_list.append(LegacyDetection(
                class_id=0,
                class_name=cname,
                category_code=code,
                defect_type=dtype,
                area=area,
                severity=severity,
                confidence=conf,
                bbox_x=0.5, bbox_y=0.5, bbox_w=1.0, bbox_h=1.0,
                raw={"source": "wallpaper", "image_shape": {"w": img_w, "h": img_h}},
            ))

    return legacy_list


__all__ = [
    "InferencePipeline",
    "LegacyDetection",
    "pipeline",
    "load_models",
    "detect_defects",
    "detect_defects_async",
    "detect_defects_legacy",
]
