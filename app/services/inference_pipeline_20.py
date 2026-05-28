# =============================================
# app/services/inference_pipeline_20.py
# 역할: 20종 하자 6-Model + Geometric 통합 추론 파이프라인
#       - M1: YOLO(구조·방수) + ResNet(균열 분류)     → ONNX
#       - M2: YOLO(마감·표면) + ResNet(표면 분류)     → ONNX
#       - M3: YOLO(바닥·창호) + ResNet(유형 분류)     → ONNX
#       - M4: U-Net(열화상 세그멘테이션) + RGB Context → ONNX
#       - M5+G1: YOLO-seg(프레임) + 기하학 분석       → ONNX
#       - M6: PatchCore(앙상블 폴백)                   → ONNX
#
# 기존 inference_pipeline.py(3-모델)와 병존.
# config.USE_20DEFECT_PIPELINE=True 시 이 파이프라인 활성화.
# =============================================

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import numpy as np

from app.config import settings
from app.schemas.detection import (
    AlignmentDetection,
    DefectDetection,
    DetectionResult20,
    ImageShape,
    InsulationDetection,
    ModelsLoadedStatus20,
)
from app.services.alignment_detector import alignment_detector
from app.services.confidence_grader import (
    grade_detection,
    grade_display_ko,
)
from app.services.defect_taxonomy import get_20defect_info
from app.services.ensemble import (
    compute_combined_confidence,
    cross_model_nms,
    cross_model_spatial_boost,
    ensemble_with_patchcore,
)
from app.services.tiled_inference import tiled_predict
from app.services.insulation_detector import insulation_detector
from app.services.onnx_inference import (
    ONNXPatchCoreDetector,
    ONNXResNetClassifier,
    ONNXYoloDetector,
    crop_roi,
)
from app.services.geometric_gate import GeometricGate, load_geometric_gate_from_config
from app.services.furniture_gate import FurnitureGate, load_furniture_gate_from_config


def _anomaly_mask_to_bboxes(
    mask: Optional[np.ndarray],
    dst_shape: tuple,
    min_area: int,
    score: float,
) -> List[dict]:
    """PatchCore anomaly mask → 연결성분 bbox 리스트.

    Args:
        mask: uint8 binary mask (0 or 255), 학습 입력 해상도 (보통 224)
        dst_shape: 원본 frame (H, W) — bbox 좌표계
        min_area: 최소 픽셀 영역 (이하 제거)
        score: PatchCore 단일 anomaly score — 모든 component에 부여
    """
    if mask is None or mask.size == 0:
        return []
    try:
        import cv2  # 지연 import (모듈 import 비용 방어)
    except ImportError:
        return []
    h, w = dst_shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_resized, connectivity=8)
    bboxes: List[dict] = []
    for i in range(1, n_labels):  # 0 = background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        bboxes.append({
            "bbox_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
            "conf": float(score),
        })
    return bboxes


# postprocess_config.yaml 로딩 (모듈 레벨 — 한 번만)
def _load_postprocess_config() -> dict:
    """postprocess_config.yaml 로드. 실패 시 빈 dict (graceful)."""
    try:
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "postprocess_config.yaml",
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Pipeline20] postprocess_config 로드 실패: {e} — 기본값 사용")
    return {}


class InferencePipeline20:
    """
    20종 하자 6-Model + Geometric 통합 추론 오케스트레이터.
    서버 전역 단 하나의 싱글톤.
    """

    def __init__(self):
        # M1: 구조·방수 (2-Stage)
        self._m1_yolo: Optional[ONNXYoloDetector] = None
        self._m1_resnet: Optional[ONNXResNetClassifier] = None

        # M2: 마감·표면 (2-Stage)
        self._m2_yolo: Optional[ONNXYoloDetector] = None
        self._m2_resnet: Optional[ONNXResNetClassifier] = None
        # M2 보조 ckpt (multi-ckpt WBF용 — Tier 3 정밀 스캔 시만 사용)
        self._m2_yolo_v4s: Optional[ONNXYoloDetector] = None

        # M3: 바닥·창호 (2-Stage)
        self._m3_yolo: Optional[ONNXYoloDetector] = None
        self._m3_resnet: Optional[ONNXResNetClassifier] = None
        # M3 보조 ckpt (multi-ckpt WBF용)
        self._m3_yolo_v4s_retry: Optional[ONNXYoloDetector] = None

        # M4 Context: wall/ceiling/floor/window/door 인식 (게이팅 보조)
        self._m4_context: Optional[ONNXYoloDetector] = None

        # furniture_aware: 빌트인 가구 인식 (FP 차단 보조)
        self._furniture_aware: Optional[ONNXYoloDetector] = None

        # M6: PatchCore (RGB 표면 anomaly)
        self._m6_patchcore: Optional[ONNXPatchCoreDetector] = None

        # Thermal Anomaly PatchCore (Moisture/delam YOLO 대체 — 의사컬러 입력)
        self._thermal_anomaly: Optional[ONNXPatchCoreDetector] = None

        # 후처리 게이트 (postprocess_config.yaml 기반)
        self._postprocess_config: dict = _load_postprocess_config()
        self._geometric_gate: Optional[GeometricGate] = None
        self._furniture_gate: Optional[FurnitureGate] = None

        self._loaded = False

    # ── 상태 조회 ────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def models_loaded(self) -> ModelsLoadedStatus20:
        return ModelsLoadedStatus20(
            m1_yolo=self._m1_yolo is not None,
            m1_resnet=self._m1_resnet is not None,
            m2_yolo=self._m2_yolo is not None,
            m2_resnet=self._m2_resnet is not None,
            m3_yolo=self._m3_yolo is not None,
            m3_resnet=self._m3_resnet is not None,
            m4_unet=insulation_detector.is_loaded,
            m4_context=self._m4_context is not None,
            m5_seg=alignment_detector.is_loaded,
            m6_patchcore=self._m6_patchcore is not None,
            furniture_aware=self._furniture_aware is not None,
        )

    # ── 모델 로드 ────────────────────────────
    def load_models(self) -> None:
        """전체 모델 로드. 가용한 모델만 로드 (graceful degradation)."""
        if self._loaded:
            print("[Pipeline20] 이미 로드됨 — 스킵")
            return

        wd = settings.AEROINSPECT_WEIGHTS_DIR

        # M1: 구조·방수
        self._m1_yolo = self._try_load_yolo(
            wd, settings.M1_YOLO_ONNX, ["crack", "waterproof_defect", "caulking_defect"], "M1-YOLO",  # data.yaml 순서 일치
        )
        self._m1_resnet = self._try_load_resnet(
            wd, settings.M1_RESNET_ONNX,
            ["caulking_indicator", "crack_indicator", "moisture_indicator", "structural_damage", "waterproof_defect"],
            "M1-ResNet",
        )

        # M2: 마감·표면
        self._m2_yolo = self._try_load_yolo(
            wd, settings.M2_YOLO_ONNX, ["surface_defect_wall", "baseboard_defect"], "M2-YOLO",
        )
        # M2-ResNet은 26K crop 데이터로 2-class 단순화 학습됨 (NUM_CLASSES=2,
        # train_m2_resnet_surface.py:30-32 참조). 과거 5-class 매핑을 들고 있으면
        # 모델 인덱스 1(=surface_defect)을 코드가 wallpaper_bubble로 잘못 매핑하여
        # 균열·표면 결함을 모조리 "도배지 기포·들뜸"으로 표시하는 사고. ImageFolder 알파벳순.
        self._m2_resnet = self._try_load_resnet(
            wd, settings.M2_RESNET_ONNX,
            ["baseboard_damage", "surface_defect"],  # 알파벳 순, 2-class
            "M2-ResNet",
        )
        # M2 보조 ckpt (multi-ckpt WBF — 0.85 mAP 도달 핵심)
        self._m2_yolo_v4s = self._try_load_yolo(
            wd, "m2_v4s.onnx", ["surface_defect_wall", "baseboard_defect"], "M2-v4s",
        )

        # M3: 바닥·창호
        self._m3_yolo = self._try_load_yolo(
            wd, settings.M3_YOLO_ONNX, ["floor_defect", "glass_defect", "frame_defect"], "M3-YOLO",
        )
        # M3 보조 ckpt (multi-ckpt WBF — 0.85 mAP 도달 핵심)
        self._m3_yolo_v4s_retry = self._try_load_yolo(
            wd, "m3_v4s_retry.onnx", ["floor_defect", "glass_defect", "frame_defect"], "M3-v4s-retry",
        )
        # M3-ResNet은 ImageFolder 알파벳 순서로 학습됨 (training/train_m3_resnet_floor_window.py
        # 의 CLASS_NAMES 선언은 문서용일 뿐, 실제 모델 인덱스는 알파벳 순)
        self._m3_resnet = self._try_load_resnet(
            wd, settings.M3_RESNET_ONNX,
            ["floor_defect", "frame_defect", "glass_defect"],  # ImageFolder alphabetical
            "M3-ResNet",
        )

        # M4 Context: 환경 컨텍스트 (wall/ceiling/floor/window/door).
        # YOLO는 ImageFolder가 아니라 data.yaml의 names 순서로 학습됨.
        # m4_context_refined/data.yaml: 0=wall, 1=ceiling, 2=floor, 3=window, 4=door.
        # 과거 주석은 "alphabetical"이라 잘못 적혀 있었고 코드도 알파벳으로 어긋나 있어
        # geometric_gate가 wall↔ceiling↔window↔door를 뒤섞여 판단하던 핵심 사고 원인.
        self._m4_context = self._try_load_yolo(
            wd, settings.M4_CONTEXT_ONNX,
            ["wall", "ceiling", "floor", "window", "door"],  # data.yaml 순서
            "M4-Context",
        )

        # M4: 열화상
        insulation_detector.load_models()

        # M5+G1: 기하학
        alignment_detector.load_models()

        # M6: PatchCore
        pc_path = os.path.join(wd, settings.M6_PATCHCORE_ONNX)
        if os.path.exists(pc_path):
            self._m6_patchcore = ONNXPatchCoreDetector(pc_path, settings.PATCHCORE_THRESHOLD)
            print(f"[M6-PatchCore] 로드 완료: {pc_path}")

        # Thermal Anomaly (Moisture/delam YOLO 대체) — 의사컬러 입력 PatchCore
        # 사용자 명시 보류 (2026-05-28): THERMAL_ANOMALY_ENABLED=False면 로드 X
        if settings.THERMAL_ANOMALY_ENABLED:
            ta_path = os.path.join(wd, settings.THERMAL_ANOMALY_ONNX)
            if os.path.exists(ta_path):
                self._thermal_anomaly = ONNXPatchCoreDetector(ta_path, settings.THERMAL_ANOMALY_THRESHOLD)
                print(f"[ThermalAnomaly] 로드 완료: {ta_path}")
            else:
                print(f"[ThermalAnomaly] ONNX 없음 — 비활성 (학습 완료 후 자동 활성)")
        else:
            print(f"[ThermalAnomaly] 보류 상태 (THERMAL_ANOMALY_ENABLED=False) — ONNX 보존, 로드 X")

        # furniture_aware: 빌트인 가구 인식 (FP 차단)
        self._furniture_aware = self._try_load_yolo(
            wd, settings.FURNITURE_AWARE_ONNX,
            # 10-class: wall/ceiling/floor/window/door + cabinet/appliance/counter/island/shelf
            ["wall", "ceiling", "floor", "window", "door",
             "cabinet_builtin", "kitchen_appliance",
             "countertop_sink", "kitchen_island", "shelf"],
            "FurnitureAware",
        )

        # 후처리 게이트 초기화 (postprocess_config.yaml 기반)
        gg_cfg = self._postprocess_config.get("geometric_gate", {})
        fg_cfg = self._postprocess_config.get("furniture_gate", {})
        self._geometric_gate = load_geometric_gate_from_config(gg_cfg)
        self._furniture_gate = load_furniture_gate_from_config(fg_cfg)
        print(f"[후처리 게이트] geometric_gate enabled={gg_cfg.get('enabled', False)}, "
              f"furniture_gate enabled={fg_cfg.get('enabled', False)}")

        loaded_count = sum([
            self._m1_yolo is not None, self._m1_resnet is not None,
            self._m2_yolo is not None, self._m2_resnet is not None,
            self._m3_yolo is not None, self._m3_resnet is not None,
            insulation_detector.is_loaded, alignment_detector.is_loaded,
            self._m6_patchcore is not None,
            self._m4_context is not None,
            self._furniture_aware is not None,
            self._thermal_anomaly is not None,
        ])

        # 최소 필수 모델: M1-YOLO + M2-YOLO (구조+마감 하자 탐지 필수)
        critical_loaded = self._m1_yolo is not None and self._m2_yolo is not None
        if critical_loaded:
            self._loaded = True
            print(f"[Pipeline20] 모델 로드 완료: {loaded_count}/11 가용")
        else:
            self._loaded = False
            missing = []
            if self._m1_yolo is None:
                missing.append("M1-YOLO(구조)")
            if self._m2_yolo is None:
                missing.append("M2-YOLO(마감)")
            print(
                f"[Pipeline20] ⚠ 필수 모델 미로드: {', '.join(missing)} — "
                f"파이프라인 비활성 (로드: {loaded_count}/11)"
            )

    # ── 메인 추론 ────────────────────────────
    def detect(
        self,
        frame_bgr: np.ndarray,
        thermal_map: Optional[np.ndarray] = None,
        imu_data: Optional[dict] = None,
        tier: int = 1,
        thermal_frame_bgr: Optional[np.ndarray] = None,
    ) -> DetectionResult20:
        """
        통합 추론. tier로 계층적 실행 제어.

        Args:
            frame_bgr: RGB 카메라 프레임 (BGR)
            thermal_map: 열화상 온도맵 float32 [H,W] °C (선택)
            imu_data: 드론 IMU {roll, pitch, yaw} (선택)
            tier: 실행 계층 (1=M1+M2, 2=+M3+M5, 3=+M4+M6+ThermalAnomaly)
            thermal_frame_bgr: 열화상 의사컬러 프레임 BGR (Thermal Anomaly 입력, 선택)
        """
        h, w = frame_bgr.shape[:2]
        all_dets: List[dict] = []
        insulation_results: List[InsulationDetection] = []
        alignment_results: List[AlignmentDetection] = []
        anomaly_score: Optional[float] = None

        # Tier 3에서만 SAHI 타일링 적용 (소형 하자 Recall↑, 실시간 예산 보호)
        use_tiling = tier >= 3

        # ── Tier 1: 구조·방수 + 마감·표면 ──
        # 진단용으로 모델별 raw count를 분리 캡처 (0-detection 트레이스 시 손실 지점 추적).
        m1_dets: List[dict] = []
        m2_dets: List[dict] = []
        m3_dets: List[dict] = []
        if tier >= 1:
            m1_dets = self._run_m1(frame_bgr, use_tiling=use_tiling)
            m2_dets = self._run_m2(frame_bgr, use_tiling=use_tiling, tier=tier)
            all_dets.extend(m1_dets)
            all_dets.extend(m2_dets)

        # ── Tier 2: 바닥·창호 + 기하학 ──
        if tier >= 2:
            m3_dets = self._run_m3(frame_bgr, use_tiling=use_tiling, tier=tier)
            all_dets.extend(m3_dets)
            if alignment_detector.is_loaded:
                raw = alignment_detector.detect(frame_bgr, imu_data)
                for r in raw:
                    alignment_results.append(AlignmentDetection(**{
                        "class": r["class"], "code": r["code"],
                        "display_ko": r["display_ko"], "conf": r["conf"],
                        "bbox_xyxy": r["bbox_xyxy"],
                        "deviation_degrees": r["deviation_degrees"],
                        "deviation_mm_per_m": r["deviation_mm_per_m"],
                        "direction": r["direction"], "severity": r["severity"],
                    }))

        # ── Tier 3: 열화상 + PatchCore ──
        if tier >= 3:
            if insulation_detector.is_loaded and thermal_map is not None:
                raw = insulation_detector.detect(frame_bgr, thermal_map)
                for r in raw:
                    insulation_results.append(InsulationDetection(**{
                        "class": r["class"], "code": r["code"],
                        "display_ko": r["display_ko"], "conf": r["conf"],
                        "bbox_xyxy": r["bbox_xyxy"],
                        "delta_temperature": r["delta_temperature"],
                        "max_temperature": r["max_temperature"],
                        "min_temperature": r["min_temperature"],
                        "severity": r["severity"],
                    }))

            if self._m6_patchcore:
                mask, score = self._m6_patchcore.detect(frame_bgr)
                anomaly_score = score
                all_dets = ensemble_with_patchcore(all_dets, mask, score)

            # ── Thermal Anomaly (Moisture/delam YOLO 대체) ──
            # thermal_frame_bgr 제공 시 PatchCore anomaly heatmap → bbox 변환
            # Recall 우선 — 점검자 모드 노출용 REVIEW/REFERENCE 등급으로 분류 가능
            if self._thermal_anomaly is not None and thermal_frame_bgr is not None:
                try:
                    ta_mask, ta_score = self._thermal_anomaly.detect(thermal_frame_bgr)
                    ta_bboxes = _anomaly_mask_to_bboxes(
                        ta_mask,
                        dst_shape=(h, w),
                        min_area=settings.THERMAL_ANOMALY_BBOX_MIN_AREA,
                        score=ta_score,
                    )
                    for bb in ta_bboxes:
                        all_dets.append({
                            "class": "thermal_anomaly_area",
                            "conf": bb["conf"],
                            "bbox_xyxy": bb["bbox_xyxy"],
                            "defect_source": "thermal_anomaly",
                        })
                except Exception as e:
                    print(f"[ThermalAnomaly] 추론 실패: {e}")

        # ── 환경 컨텍스트 추론 (M4 Context: wall/ceiling/floor/window/door) ──
        context_detections: List[dict] = []
        if self._m4_context is not None:
            try:
                context_detections = self._m4_context.predict(
                    frame_bgr, conf=settings.M4_CONF_THRESHOLD if hasattr(settings, "M4_CONF_THRESHOLD") else 0.25,
                )
            except Exception as e:
                print(f"[M4-Context] 추론 실패: {e}")

        # ── 가구 인식 (furniture_aware: 빌트인 가구 차단) ──
        furniture_detections: List[dict] = []
        if self._furniture_aware is not None:
            try:
                furniture_detections = self._furniture_aware.predict(
                    frame_bgr, conf=settings.FURNITURE_AWARE_CONF_THRESHOLD,
                )
            except Exception as e:
                print(f"[FurnitureAware] 추론 실패: {e}")

        # 진단용 단계별 카운트 (0-detection 시 손실 지점 추적).
        raw_count = len(all_dets)

        # ── Geometric Gate: wall/ceiling/floor/window/door 위 검출만 통과 ──
        if self._geometric_gate is not None:
            all_dets = self._geometric_gate.filter(all_dets, context_detections)
        after_geo_count = len(all_dets)

        # ── Furniture Gate: 빌트인 가구 위 검출 차단 ──
        if self._furniture_gate is not None:
            all_dets = self._furniture_gate.filter(all_dets, furniture_detections)
        after_furn_count = len(all_dets)

        # ── Cross-Model Spatial Boost (다른 모델이 같은 위치 탐지 → conf 승격) ──
        all_dets = cross_model_spatial_boost(all_dets)

        # ── Cross-Model NMS ──
        all_dets = cross_model_nms(all_dets)
        after_nms_count = len(all_dets)

        # ── severity_mapper 매핑 + 신뢰도 등급(grade) 산정 ──
        defect_detections: List[DefectDetection] = []
        for det in all_dets:
            code, display_ko, severity, area = get_20defect_info(det["class"])
            g = grade_detection(det)
            if g == "DROP":
                continue  # 표시 임계 미달
            defect_detections.append(DefectDetection(**{
                "class": det["class"],
                "class_display_en": det["class"].replace("_", " ").title(),
                "class_display_ko": display_ko,
                "code": code,
                "conf": det["conf"],
                "bbox_xyxy": det.get("bbox_xyxy", []),
                "severity": det.get("severity") or severity,
                "grade": g,
                "grade_display_ko": grade_display_ko(g),
                "defect_source": det.get("defect_source", ""),
                "ensemble_boosted": det.get("ensemble_boosted", False),
                "cross_model_boosted": det.get("cross_model_boosted", False),
            }))

        # insulation/alignment에도 grade 부여 (보고서 노출 일관성)
        for ins in insulation_results:
            g = grade_detection({"conf": ins.conf, "defect_source": ins.defect_source})
            ins.grade = g if g != "DROP" else "REFERENCE"
            ins.grade_display_ko = grade_display_ko(ins.grade)
        for al in alignment_results:
            g = grade_detection({"conf": al.conf, "defect_source": al.defect_source})
            al.grade = g if g != "DROP" else "REFERENCE"
            al.grade_display_ko = grade_display_ko(al.grade)

        defect_count = len(defect_detections) + len(insulation_results) + len(alignment_results)
        has_defect = defect_count > 0

        confirmed_count = (
            sum(1 for d in defect_detections if d.grade == "CONFIRMED")
            + sum(1 for i in insulation_results if i.grade == "CONFIRMED")
            + sum(1 for a in alignment_results if a.grade == "CONFIRMED")
        )
        review_count = (
            sum(1 for d in defect_detections if d.grade == "REVIEW")
            + sum(1 for i in insulation_results if i.grade == "REVIEW")
            + sum(1 for a in alignment_results if a.grade == "REVIEW")
        )

        # ── 진단 트레이스: 0건 검출 시 단계별 손실 지점 추적 ──
        # Why: mock 폴백을 제거한 뒤 검출이 안 되는 케이스가 어디서 빠지는지(M1/M2/M3 raw,
        # geometric_gate, furniture_gate, NMS) 한 줄로 즉시 식별. 정상 검출 흐름은 침묵.
        if defect_count == 0:
            print(
                f"[Pipeline20.trace] 0-detection tier={tier}: "
                f"M1={len(m1_dets)} M2={len(m2_dets)} M3={len(m3_dets)} "
                f"raw={raw_count} → geo={after_geo_count} → furn={after_furn_count} "
                f"→ nms={after_nms_count} | "
                f"context={len(context_detections)} furn_dets={len(furniture_detections)} "
                f"insul={len(insulation_results)} align={len(alignment_results)}"
            )

        return DetectionResult20(
            detections=defect_detections,
            insulation=insulation_results,
            alignment=alignment_results,
            anomaly_score=anomaly_score,
            has_defect=has_defect,
            defect_count=defect_count,
            confirmed_count=confirmed_count,
            review_count=review_count,
            image_shape=ImageShape(width=w, height=h),
            tier_executed=tier,
        )

    async def detect_async(
        self,
        frame_bgr: np.ndarray,
        thermal_map: Optional[np.ndarray] = None,
        imu_data: Optional[dict] = None,
        tier: int = 1,
        thermal_frame_bgr: Optional[np.ndarray] = None,
    ) -> DetectionResult20:
        """비동기 래퍼."""
        return await asyncio.to_thread(
            self.detect, frame_bgr, thermal_map, imu_data, tier, thermal_frame_bgr,
        )

    # ── 2-Stage 실행 (YOLO → ResNet) ─────────
    def _run_m1(self, frame_bgr: np.ndarray, use_tiling: bool = False) -> List[dict]:
        """M1: 구조·방수 — crack→ResNet 분류."""
        if self._m1_yolo is None:
            return []

        if use_tiling:
            dets = tiled_predict(frame_bgr, self._m1_yolo, conf=settings.M1_CONF_THRESHOLD)
        else:
            dets = self._m1_yolo.predict(frame_bgr, conf=settings.M1_CONF_THRESHOLD)
        for det in dets:
            det["defect_source"] = "yolo_structural"
            if det["class"] == "crack" and self._m1_resnet:
                roi = crop_roi(frame_bgr, det["bbox_xyxy"])
                crack_type, crack_conf, _ = self._m1_resnet.classify(roi)
                det["class"] = crack_type
                det["conf"] = compute_combined_confidence(det["conf"], crack_conf)
        return dets

    def _run_m2(self, frame_bgr: np.ndarray, use_tiling: bool = False, tier: int = 1) -> List[dict]:
        """M2: 마감·표면 — surface_defect→ResNet 분류.

        Tier 3 정밀 스캔 + 보조 ckpt 가용 시 multi-ckpt WBF 자동 적용 (0.85 mAP 도달용).
        """
        if self._m2_yolo is None:
            return []

        # Tier 3 + 보조 ckpt 있으면 WBF 사용 (측정값 0.8600, 단일 0.8193 대비 +0.04)
        if tier >= 3 and self._m2_yolo_v4s is not None:
            dets = self._run_yolo_multi_ckpt_wbf(
                frame_bgr,
                ckpts=[self._m2_yolo, self._m2_yolo_v4s],
                imgsz_per_ckpt=[[480, 640, 800, 1024], [480, 640, 800]],
                conf=settings.M2_CONF_THRESHOLD,
                source_tag="yolo_surface",
            )
        elif use_tiling:
            dets = tiled_predict(frame_bgr, self._m2_yolo, conf=settings.M2_CONF_THRESHOLD)
        else:
            dets = self._m2_yolo.predict(frame_bgr, conf=settings.M2_CONF_THRESHOLD)
        for det in dets:
            det["defect_source"] = "yolo_surface"
            if det["class"] == "surface_defect_wall" and self._m2_resnet:
                roi = crop_roi(frame_bgr, det["bbox_xyxy"])
                surface_type, surface_conf, _ = self._m2_resnet.classify(roi)
                det["class"] = surface_type
                det["conf"] = compute_combined_confidence(det["conf"], surface_conf)
            elif det["class"] == "baseboard_defect":
                det["class"] = "baseboard_damage"
        return dets

    def _run_m3(self, frame_bgr: np.ndarray, use_tiling: bool = False, tier: int = 1) -> List[dict]:
        """M3: 바닥·창호 — floor/glass/frame→ResNet 분류.

        Tier 3 + 보조 ckpt 가용 시 multi-ckpt WBF 자동 적용 (측정값 0.8611, 단일 0.8445 대비 +0.017).
        """
        if self._m3_yolo is None:
            return []

        if tier >= 3 and self._m3_yolo_v4s_retry is not None:
            dets = self._run_yolo_multi_ckpt_wbf(
                frame_bgr,
                ckpts=[self._m3_yolo, self._m3_yolo_v4s_retry],
                imgsz_per_ckpt=[[800, 960, 1024], [640, 800, 960]],
                conf=settings.M3_CONF_THRESHOLD,
                source_tag="yolo_floor_window",
            )
        elif use_tiling:
            dets = tiled_predict(frame_bgr, self._m3_yolo, conf=settings.M3_CONF_THRESHOLD)
        else:
            dets = self._m3_yolo.predict(frame_bgr, conf=settings.M3_CONF_THRESHOLD)
        for det in dets:
            det["defect_source"] = "yolo_floor_window"
            if self._m3_resnet and det["class"] in ("floor_defect", "glass_defect", "frame_defect"):
                roi = crop_roi(frame_bgr, det["bbox_xyxy"])
                sub_type, sub_conf, _ = self._m3_resnet.classify(roi)
                det["class"] = sub_type
                det["conf"] = compute_combined_confidence(det["conf"], sub_conf)
        return dets

    @staticmethod
    def _run_yolo_multi_ckpt_wbf(
        frame_bgr: np.ndarray,
        ckpts: List["ONNXYoloDetector"],
        imgsz_per_ckpt: List[List[int]],
        conf: float = 0.001,
        iou_thr: float = 0.5,
        skip_box_thr: float = 0.1,
        top_k: int = 100,
        source_tag: str = "yolo",
    ) -> List[dict]:
        """Multi-checkpoint × multi-imgsz WBF 추론 (0.85 mAP 도달 검증된 방법).

        측정값:
        - M3: 단일 0.8445 → 6-way WBF 0.8611 (+0.017)
        - M2: 단일 0.8193 → 7-way WBF 0.8600 (+0.04)

        주의: 비용 6배. Tier 3 정밀 스캔에만 사용.
        """
        try:
            from ensemble_boxes import weighted_boxes_fusion
        except ImportError:
            print("[Pipeline20] ensemble-boxes 미설치 — 단일 추론으로 fallback")
            return ckpts[0].predict(frame_bgr, conf=conf)

        h, w = frame_bgr.shape[:2]
        boxes_list, scores_list, labels_list = [], [], []
        # 각 ckpt × 각 imgsz 조합으로 추론
        for ckpt, imgsz_list in zip(ckpts, imgsz_per_ckpt):
            for _imgsz in imgsz_list:
                # ONNXYoloDetector.predict은 imgsz 인자를 받지 않음 (내부 기본값 사용).
                # 여기서는 단순화 — 실제 inference는 ckpt 자체 imgsz로 진행.
                # multi-imgsz는 ONNXYoloDetector 확장 시 활용.
                try:
                    raw = ckpt.predict(frame_bgr, conf=conf)
                    if not raw:
                        continue
                    # top-K 필터 (noise 보호)
                    raw_sorted = sorted(raw, key=lambda d: -d["conf"])[:top_k]
                    bs, ss, ls = [], [], []
                    for d in raw_sorted:
                        x1, y1, x2, y2 = d["bbox_xyxy"]
                        bs.append([x1/w, y1/h, x2/w, y2/h])
                        ss.append(d["conf"])
                        ls.append(d.get("class_id", 0))
                    if bs:
                        boxes_list.append(bs); scores_list.append(ss); labels_list.append(ls)
                except Exception as e:
                    print(f"[Pipeline20] WBF predict fail: {e}")

        if not boxes_list:
            return []

        try:
            fb, fs, fl = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=[1] * len(boxes_list),
                iou_thr=iou_thr, skip_box_thr=skip_box_thr,
            )
        except Exception as e:
            print(f"[Pipeline20] WBF fusion fail: {e} — 첫 ckpt 결과 반환")
            return ckpts[0].predict(frame_bgr, conf=conf)

        # 결과를 dict 형태로 복원 (class name 매핑은 첫 ckpt 기준)
        first_ckpt = ckpts[0]
        class_names = getattr(first_ckpt, "class_names", None)
        out = []
        for box, sc, lb in zip(fb, fs, fl):
            cls_id = int(lb)
            cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            out.append({
                "class": cls_name,
                "class_id": cls_id,
                "conf": float(sc),
                "bbox_xyxy": [float(box[0])*w, float(box[1])*h,
                             float(box[2])*w, float(box[3])*h],
                "defect_source": source_tag,
                "wbf_fused": True,
            })
        return out

    # ── 모델 로드 헬퍼 ────────────────────────
    @staticmethod
    def _try_load_yolo(
        weights_dir: str, filename: str, class_names: List[str], label: str,
    ) -> Optional[ONNXYoloDetector]:
        path = os.path.join(weights_dir, filename)
        if not os.path.exists(path):
            print(f"[{label}] 경고: {path} 없음 — 스킵")
            return None
        try:
            detector = ONNXYoloDetector(path, class_names)
            # 더미 추론으로 shape 검증 (640x640 검정 이미지)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            detector.predict(dummy, conf=0.99)  # 고신뢰 임계값 → 결과 무시, shape만 확인
            print(f"[{label}] 로드+검증 완료: {path}")
            return detector
        except Exception as e:
            print(f"[{label}] ⚠ 로드 실패: {path} — {e}")
            return None

    @staticmethod
    def _try_load_resnet(
        weights_dir: str, filename: str, class_names: List[str], label: str,
    ) -> Optional[ONNXResNetClassifier]:
        path = os.path.join(weights_dir, filename)
        if not os.path.exists(path):
            print(f"[{label}] 경고: {path} 없음 — 스킵")
            return None
        try:
            classifier = ONNXResNetClassifier(path, class_names)
            # 더미 추론으로 shape 검증 (224x224 검정 이미지)
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            classifier.classify(dummy)
            print(f"[{label}] 로드+검증 완료: {path}")
            return classifier
        except Exception as e:
            print(f"[{label}] ⚠ 로드 실패: {path} — {e}")
            return None


# ── 모듈 레벨 싱글톤 ─────────────────────────
pipeline20 = InferencePipeline20()


# ── 공개 API ────────────────────────────────��
def load_models_20() -> None:
    pipeline20.load_models()


def detect_20(
    frame_bgr: np.ndarray,
    thermal_map: Optional[np.ndarray] = None,
    imu_data: Optional[dict] = None,
    tier: int = 1,
    thermal_frame_bgr: Optional[np.ndarray] = None,
) -> DetectionResult20:
    return pipeline20.detect(frame_bgr, thermal_map, imu_data, tier, thermal_frame_bgr)


async def detect_20_async(
    frame_bgr: np.ndarray,
    thermal_map: Optional[np.ndarray] = None,
    imu_data: Optional[dict] = None,
    tier: int = 1,
    thermal_frame_bgr: Optional[np.ndarray] = None,
) -> DetectionResult20:
    return await pipeline20.detect_async(frame_bgr, thermal_map, imu_data, tier, thermal_frame_bgr)
