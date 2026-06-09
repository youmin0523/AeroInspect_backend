# =============================================
# app/services/hybrid_detector.py
# 역할: ONNX 제안 + 비전 LLM 판정 캐스케이드 (상업용 검출 확정 로직)
#
# 프로세스:
#   1) ONNX 파이프라인이 후보 검출 (위치 정밀 + 고recall)
#   2) VLM이 1회 호출로 후보 검증/종류교정/기각 + 누락 보완 (전체 맥락 활용)
#   3) 고conf 충돌 시 1회 한정 재판정(토론-lite, 설정 가드)
#   4) 결정론적 병합 규칙으로 등급 확정 + provenance 기록
#
# 상업용 핵심 원칙 (분쟁 책임 대비):
#   - 단일 엔진 단독으로는 CONFIRMED(보고서 등재) 불가.
#   - ONNX+VLM 합의/종류교정 → CONFIRMED 가능 (위치=ONNX, 종류=VLM 권위).
#   - ONNX 단독 / VLM 단독 → REVIEW 상한 (점검자 확인).
#   - VLM 기각 → REFERENCE (감사 로그 보존).
#   - 모든 판정에 onnx_conf/vlm_conf/agreement/reasoning 기록.
#
# 재사용: confidence_grader.grade_detection (결정론적 등급), severity_mapper,
#         inference_pipeline(3-모델) / inference_pipeline_20(20종), vlm_detector.
# =============================================

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.schemas.detection import (
    HybridDetection,
    HybridDetectionResult,
    ImageShape,
)
from app.services.box_refiner import refine_box
from app.services.confidence_grader import (
    grade_detection,
    grade_display_ko,
    is_listable,
)
from app.services.vlm_detector import vlm_detector
from app.utils.severity_mapper import get_severity_by_code

logger = logging.getLogger(__name__)

# 3-모델 파이프라인 내부 클래스 → 20종 taxonomy class_name 브리지
# (3-모델은 Crack/Moisture/delamination만 위치검출. 20종 파이프라인은 직접 매칭됨)
_3MODEL_BRIDGE = {
    "Crack": "crack_structural",        # A-02
    "Moisture": "waterproof_defect",    # B-04
    "delamination": "wall_insulation_gap",  # B-02
}

# 등급 순위 (상한 캡 계산용)
_GRADE_RANK = {"DROP": 0, "REFERENCE": 1, "REVIEW": 2, "CONFIRMED": 3}
_RANK_GRADE = {v: k for k, v in _GRADE_RANK.items()}


class HybridDetector:
    """ONNX 제안 → VLM 판정 캐스케이드 오케스트레이터 (싱글톤)."""

    async def detect(
        self,
        image_bytes: bytes,
        *,
        thermal_map: Optional[np.ndarray] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> HybridDetectionResult:
        t0 = time.perf_counter()
        provider = (provider or settings.VLM_PROVIDER).lower()
        model = model or settings.VLM_MODEL

        frame = self._decode(image_bytes)
        h, w = frame.shape[:2]
        shape = ImageShape(width=int(w), height=int(h))

        # ── 1) ONNX 후보 ──
        engine, candidates = await self._run_onnx(frame, thermal_map)

        # ── 2) VLM 판정 (1회) ──
        vlm_calls = 0
        verdicts_by_id: Dict[int, Dict[str, Any]] = {}
        missed: List[Dict[str, Any]] = []
        if vlm_detector and (candidates or settings.VLM_DETECTION_ENABLED):
            adj = await vlm_detector.adjudicate(
                image_bytes, candidates, provider=provider, model=model
            )
            vlm_calls = 1
            verdicts_by_id = {
                int(v["id"]): v for v in adj["verdicts"] if "id" in v
            }
            missed = adj["missed"]

            # ── 3) 충돌 재판정 (토론-lite, 1회 한정) ──
            conflict_ids = self._find_conflicts(candidates, verdicts_by_id)
            if conflict_ids and settings.VLM_ADJUDICATE_CONFLICTS:
                adj2 = await vlm_detector.adjudicate(
                    image_bytes, candidates, provider=provider, model=model,
                    conflict_ids=conflict_ids,
                )
                vlm_calls = 2
                for v in adj2["verdicts"]:
                    if "id" in v and int(v["id"]) in conflict_ids:
                        verdicts_by_id[int(v["id"])] = v  # 재심 결과로 덮어씀

        # ── 4) 결정론적 병합 (+ VLM 박스 CV 보정) ──
        detections = self._merge(candidates, verdicts_by_id, missed, frame)

        confirmed = sum(1 for d in detections if d.grade == "CONFIRMED")
        review = sum(1 for d in detections if d.grade == "REVIEW")
        rejected = sum(1 for d in detections if d.status == "rejected")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return HybridDetectionResult(
            detections=detections,
            has_defect=any(d.grade in ("CONFIRMED", "REVIEW") for d in detections),
            defect_count=len(detections),
            confirmed_count=confirmed,
            review_count=review,
            rejected_count=rejected,
            onnx_engine=engine,
            vlm_provider=provider,
            vlm_model=model,
            vlm_calls=vlm_calls,
            latency_ms=round(latency_ms, 1),
            image_shape=shape,
        )

    # ── ONNX 실행 + 후보 정규화 ──────────────
    async def _run_onnx(
        self, frame: np.ndarray, thermal_map: Optional[np.ndarray]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """활성 파이프라인 실행 → 정규화 후보 리스트 {id, class_name, conf, bbox_xyxy}."""
        # 20종 파이프라인 우선 (taxonomy 직접 매칭).
        # USE_20DEFECT_PIPELINE 플래그가 꺼져 있어도 이미 로드돼 있으면 사용 —
        # TEST MODE 가 lazy-load 해둔 파이프라인을 하이브리드에서도 활용(플래그는 startup 자동로드 여부만 제어).
        try:
            from app.services.inference_pipeline_20 import pipeline20
            if pipeline20.is_loaded:
                result = await pipeline20.detect_async(
                    frame, thermal_map=thermal_map, tier=3
                )
                cands = []
                for i, d in enumerate(result.detections):
                    if not d.bbox_xyxy:
                        continue
                    cands.append({
                        "id": i,
                        "class_name": d.class_,
                        "conf": float(d.conf),
                        "bbox_xyxy": list(d.bbox_xyxy),
                    })
                return "pipeline20", cands
        except Exception as e:
            logger.warning("pipeline20 실행 실패, 3-모델로 폴백: %s", e)

        # 3-모델 파이프라인 (Crack/Moisture/delamination → 브리지)
        try:
            from app.services.inference_pipeline import pipeline
            if pipeline.is_loaded:
                result = await __import__("asyncio").to_thread(
                    pipeline.detect, frame, None, False
                )
                cands = []
                i = 0
                for det in list(result.yolo_thermal) + list(result.yolo_delam):
                    bridged = _3MODEL_BRIDGE.get(det.class_, det.class_)
                    cands.append({
                        "id": i,
                        "class_name": bridged,
                        "conf": float(det.conf),
                        "bbox_xyxy": list(det.bbox_xyxy),
                    })
                    i += 1
                return "pipeline3", cands
        except Exception as e:
            logger.warning("3-모델 파이프라인 실행 실패: %s", e)

        return "none", []

    # ── 충돌 탐지 ─────────────────────────────
    @staticmethod
    def _find_conflicts(
        candidates: List[Dict[str, Any]], verdicts: Dict[int, Dict[str, Any]]
    ) -> List[int]:
        """ONNX 고신뢰 검출인데 VLM이 기각 → 재판정 대상."""
        out = []
        thr = settings.VLM_CONFLICT_ONNX_CONF
        for c in candidates:
            v = verdicts.get(c["id"])
            if v and v.get("verdict") == "reject" and float(c.get("conf", 0)) >= thr:
                out.append(c["id"])
        return out

    # ── 결정론적 병합 ─────────────────────────
    def _merge(
        self,
        candidates: List[Dict[str, Any]],
        verdicts: Dict[int, Dict[str, Any]],
        missed: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> List[HybridDetection]:
        height, width = frame.shape[:2]
        out: List[HybridDetection] = []

        # ── ONNX 후보 기반 ──
        for c in candidates:
            onnx_conf = float(c.get("conf", 0.0))
            bbox = c.get("bbox_xyxy") or []
            v = verdicts.get(c["id"])

            if v is None:
                # VLM 미언급 — 미검증 단일 엔진
                final_class = c["class_name"]
                status, agreement, boosted = "onnx_only", False, False
                grade_conf, vlm_conf, reason = onnx_conf, None, ""
            else:
                verdict = str(v.get("verdict", "")).lower()
                vlm_conf = self._clip(v.get("conf", 0.5))
                reason = str(v.get("reason", ""))[:300]
                if verdict == "reject":
                    final_class = c["class_name"]
                    status, agreement, boosted = "rejected", False, False
                    grade_conf = vlm_conf
                elif verdict == "reclassify":
                    final_class = self._valid_class(v.get("class_name"), c["class_name"])
                    status, agreement, boosted = "reclassified", True, True
                    grade_conf = vlm_conf  # 종류는 VLM 권위
                else:  # confirm (기본)
                    final_class = self._valid_class(v.get("class_name"), c["class_name"])
                    status, agreement, boosted = "confirmed_by_both", True, True
                    grade_conf = max(onnx_conf, vlm_conf)

            det = self._build_detection(
                final_class=final_class, bbox=bbox, localization=("bbox" if len(bbox) == 4 else "image_level"),
                status=status, grade_conf=grade_conf, boosted=boosted, agreement=agreement,
                onnx_conf=onnx_conf, vlm_conf=vlm_conf, source=("onnx+vlm" if v else "onnx"),
                reasoning=reason, single_engine=(status in ("onnx_only",)),
            )
            if det is not None:
                out.append(det)

        # ── VLM 누락 보완 ──
        for m in missed:
            cls = m.get("class_name")
            if not cls:
                continue
            vlm_conf = self._clip(m.get("conf", 0.5))
            box = m.get("box_2d")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                bbox = vlm_detector._box2d_to_xyxy(box, width, height)
                loc = "bbox"
                # VLM 거친 박스 → 실제 하자 픽셀에 스냅 보정 (실패 시 원본 유지)
                if settings.VLM_BOX_REFINE:
                    refined, _method = refine_box(frame, bbox, cls)
                    if refined is not None:
                        bbox = refined
            else:
                bbox = [0.0, 0.0, float(width), float(height)]
                loc = "image_level"
            det = self._build_detection(
                final_class=cls, bbox=bbox, localization=loc,
                status="vlm_only", grade_conf=vlm_conf, boosted=False, agreement=False,
                onnx_conf=None, vlm_conf=vlm_conf, source="vlm",
                reasoning=str(m.get("reason", ""))[:300], single_engine=True,
            )
            if det is not None:
                out.append(det)

        return out

    def _build_detection(
        self, *, final_class: str, bbox: List[float], localization: str, status: str,
        grade_conf: float, boosted: bool, agreement: bool,
        onnx_conf: Optional[float], vlm_conf: Optional[float], source: str,
        reasoning: str, single_engine: bool,
    ) -> Optional[HybridDetection]:
        info = get_severity_by_code(final_class)
        if info.get("code") == "X-00":
            logger.debug("하이브리드: 목록 외 클래스 폐기 %s", final_class)
            return None

        # 결정론적 등급 산정
        base = grade_detection({
            "conf": grade_conf,
            "defect_source": source,
            "cross_model_boosted": boosted,
        })

        if status == "rejected":
            grade = "REFERENCE"          # 기각은 감사 보존용
        elif single_engine:
            grade = self._cap(base, "REVIEW")   # 단일 엔진 미검증 → REVIEW 상한
        else:
            grade = base                 # 합의/교정 → CONFIRMED 가능

        if grade == "DROP":
            return None

        return HybridDetection(
            **{"class": info["class_name"]},
            class_display_ko=info.get("name", ""),
            code=info.get("code", ""),
            area=info.get("area", ""),
            conf=round(grade_conf, 3),
            severity=info.get("severity"),
            bbox_xyxy=[round(float(x), 1) for x in bbox],
            localization=localization,
            status=status,
            grade=grade,
            grade_display_ko=grade_display_ko(grade),
            listable=is_listable(grade),
            onnx_conf=round(onnx_conf, 3) if onnx_conf is not None else None,
            vlm_conf=round(vlm_conf, 3) if vlm_conf is not None else None,
            agreement=agreement,
            source=source,
            reasoning=reasoning,
        )

    # ── 유틸 ─────────────────────────────────
    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지 디코딩 실패 (지원되지 않는 포맷).")
        return frame

    @staticmethod
    def _clip(x: Any) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _valid_class(candidate: Any, fallback: str) -> str:
        """VLM이 교정한 class_name이 20종 목록에 있으면 채택, 아니면 fallback."""
        if candidate and get_severity_by_code(str(candidate)).get("code") != "X-00":
            return str(candidate)
        return fallback

    @staticmethod
    def _cap(grade: str, ceiling: str) -> str:
        """grade를 ceiling 등급으로 상한 제한."""
        if _GRADE_RANK.get(grade, 0) > _GRADE_RANK[ceiling]:
            return ceiling
        return grade


# 싱글톤
hybrid_detector = HybridDetector()


async def detect_hybrid_async(
    image_bytes: bytes,
    *,
    thermal_map: Optional[np.ndarray] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> HybridDetectionResult:
    """공개 API — ONNX+VLM 하이브리드 검출."""
    return await hybrid_detector.detect(
        image_bytes, thermal_map=thermal_map, provider=provider, model=model
    )
