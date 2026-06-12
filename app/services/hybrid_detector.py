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

import asyncio
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
        # VLM 주도 모드: VLM(grounding)이 1차 검출을 주도하고 ONNX가 교차검증.
        # (ONNX recall 약점 보완 — VLM∩ONNX=CONFIRMED, VLM단독=REVIEW, ONNX단독=REVIEW)
        if settings.VLM_PRIMARY:
            return await self._detect_vlm_primary(image_bytes, thermal_map=thermal_map)

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
                result = await asyncio.to_thread(
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

    # ── VLM 주도 검출 + ONNX 교차검증 ─────────────────
    async def _detect_vlm_primary(
        self, image_bytes: bytes, *, thermal_map: Optional[np.ndarray] = None,
    ) -> HybridDetectionResult:
        """VLM(grounding) 앙상블이 1차 검출을 주도하고 ONNX가 교차검증.
        합치: VLM∩ONNX 겹침 → CONFIRMED(박스=ONNX 정밀, 종류=VLM 권위),
              VLM 단독 → box_refiner 보정 후 REVIEW, ONNX 단독 → REVIEW.
        VLM 전원 실패해도 ONNX 단독 결과는 그대로 노출(예외 없이 degrade)."""
        t0 = time.perf_counter()
        frame = self._decode(image_bytes)
        h, w = frame.shape[:2]
        shape = ImageShape(width=int(w), height=int(h))

        specs = self._parse_ensemble()
        # ONNX 후보 + VLM 앙상블(grounding) 병렬 실행
        results = await asyncio.gather(
            self._run_onnx(frame, thermal_map),
            *[self._vlm_detect_one(image_bytes, p, m) for p, m in specs],
            return_exceptions=True,
        )
        onnx_res, vlm_res = results[0], results[1:]

        if isinstance(onnx_res, Exception):
            logger.warning("VLM-primary: ONNX 실패 — VLM 단독 진행: %s", onnx_res)
            engine, onnx_cands = "none", []
        else:
            engine, onnx_cands = onnx_res

        dets_by_provider: List[Tuple[str, List[Any]]] = []
        for (p, _m), r in zip(specs, vlm_res):
            if isinstance(r, Exception):
                logger.warning("VLM-primary: provider %s 검출 실패: %s", p, r)
                continue
            dets_by_provider.append(
                (p, [d for d in r.detections if d.bbox_xyxy and d.localization == "bbox"])
            )
        vlm_calls = len(dets_by_provider)

        vlm_consensus = self._cluster_vlm(dets_by_provider)
        detections = self._merge_vlm_primary(vlm_consensus, onnx_cands, frame)

        confirmed = sum(1 for d in detections if d.grade == "CONFIRMED")
        review = sum(1 for d in detections if d.grade == "REVIEW")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        providers_used = "+".join(p for p, _ in dets_by_provider) or settings.VLM_PROVIDER
        return HybridDetectionResult(
            detections=detections,
            has_defect=any(d.grade in ("CONFIRMED", "REVIEW") for d in detections),
            defect_count=len(detections),
            confirmed_count=confirmed,
            review_count=review,
            rejected_count=0,
            onnx_engine=engine,
            vlm_provider=providers_used,
            vlm_model=("ensemble" if vlm_calls > 1 else (specs[0][1] if specs else "")),
            vlm_calls=vlm_calls,
            latency_ms=round(latency_ms, 1),
            image_shape=shape,
        )

    async def _vlm_detect_one(self, image_bytes: bytes, provider: str, model: str):
        """단일 provider grounding 검출 (앙상블 한 갈래)."""
        return await vlm_detector.detect(
            image_bytes, mode="grounding", provider=provider, model=(model or None)
        )

    @staticmethod
    def _parse_ensemble() -> List[Tuple[str, str]]:
        """VLM_ENSEMBLE("provider:model,provider:model") → [(provider, model)].
        앙상블 OFF면 단일 provider/model 한 갈래."""
        if not settings.VLM_ENSEMBLE_ENABLED:
            return [(settings.VLM_PROVIDER.lower(), settings.VLM_MODEL)]
        out: List[Tuple[str, str]] = []
        for part in (settings.VLM_ENSEMBLE or "").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                p, m = part.split(":", 1)
                out.append((p.strip().lower(), m.strip()))
            else:
                out.append((part.lower(), settings.VLM_MODEL))
        return out or [(settings.VLM_PROVIDER.lower(), settings.VLM_MODEL)]

    def _cluster_vlm(
        self, dets_by_provider: List[Tuple[str, List[Any]]]
    ) -> List[Dict[str, Any]]:
        """여러 provider 의 VLM 검출을 IoU 로 묶어 합의 클러스터 생성.
        agree_count = 그 위치를 본 provider 수(앙상블 합의 강도)."""
        clusters: List[Dict[str, Any]] = []
        for provider, dets in dets_by_provider:
            for d in dets:
                box = [float(x) for x in d.bbox_xyxy[:4]]
                placed = False
                for c in clusters:
                    if self._iou_xyxy(box, c["box"]) >= 0.4:
                        c["items"].append((provider, d))
                        c["providers"].add(provider)
                        placed = True
                        break
                if not placed:
                    clusters.append({"box": box, "items": [(provider, d)], "providers": {provider}})

        out: List[Dict[str, Any]] = []
        for c in clusters:
            best = max(c["items"], key=lambda pd: float(pd[1].conf))[1]
            conf = sum(float(pd[1].conf) for pd in c["items"]) / len(c["items"])
            out.append({
                "class_name": best.class_,
                "conf": conf,
                "bbox": [float(x) for x in best.bbox_xyxy[:4]],
                "agree_count": len(c["providers"]),
                "reason": "; ".join(f"{p}:{(pd.reasoning or '')[:60]}" for p, pd in c["items"])[:300],
            })
        return out

    def _merge_vlm_primary(
        self, vlm_consensus: List[Dict[str, Any]],
        onnx_cands: List[Dict[str, Any]], frame: np.ndarray,
    ) -> List[HybridDetection]:
        out: List[HybridDetection] = []
        used_onnx: set = set()

        for vc in vlm_consensus:
            vbox = vc["bbox"]
            best_i, best_iou = -1, 0.0
            for i, oc in enumerate(onnx_cands):
                ob = [float(x) for x in (oc.get("bbox_xyxy") or [])[:4]]
                if len(ob) != 4:
                    continue
                iou = self._iou_xyxy(vbox, ob)
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_i >= 0 and best_iou >= settings.VLM_PRIMARY_IOU:
                # VLM∩ONNX 합의 → CONFIRMED. 박스=ONNX(정밀), 종류=VLM 권위.
                oc = onnx_cands[best_i]
                used_onnx.add(best_i)
                final_class = self._valid_class(vc["class_name"], oc.get("class_name"))
                det = self._build_detection(
                    final_class=final_class,
                    bbox=[float(x) for x in oc["bbox_xyxy"][:4]],
                    localization="bbox", status="confirmed_by_both",
                    grade_conf=max(float(oc.get("conf", 0.0)), vc["conf"]),
                    boosted=True, agreement=True,
                    onnx_conf=float(oc.get("conf", 0.0)), vlm_conf=vc["conf"],
                    source="onnx+vlm", reasoning=vc["reason"], single_engine=False,
                )
            else:
                # VLM 단독 → 박스 보정 후 REVIEW (단일 엔진 상한).
                bbox = vbox
                if settings.VLM_BOX_REFINE:
                    refined, _method = refine_box(frame, bbox, vc["class_name"])
                    if refined is not None:
                        bbox = refined
                det = self._build_detection(
                    final_class=vc["class_name"], bbox=bbox, localization="bbox",
                    status="vlm_only", grade_conf=vc["conf"], boosted=False, agreement=False,
                    onnx_conf=None, vlm_conf=vc["conf"], source="vlm",
                    reasoning=vc["reason"], single_engine=True,
                )
            if det is not None:
                out.append(det)

        # ONNX 단독(VLM 미검출) 처리 — 클래스별 차등.
        #  • 균열·구조(crack/structural/rebar): ONNX(M1/M3)가 강하고 안전직결(미탐 비용 큼) →
        #    VLM 미확인이어도 유지. 균열난 면엔 균열이 조밀히 잡혀야 함(사용자 레퍼런스 기대치).
        #  • 그 외 노이즈성(코킹·방수·걸레받이·표면·얼룩): VLM 미확인이면 폐기 — ONNX 가 conf 1.0
        #    으로 아무 데나 남발하는 '중구난방' 차단. VLM 권위 우선.
        #  KEEP_ONNX_ONLY=True 면 전부 유지(과거 동작).
        for i, oc in enumerate(onnx_cands):
            if i in used_onnx:
                continue
            ob = [float(x) for x in (oc.get("bbox_xyxy") or [])[:4]]
            if len(ob) != 4:
                continue
            cls = str(oc.get("class_name", "")).lower()
            trusted = any(k in cls for k in ("crack", "structural", "rebar", "균열", "철근"))
            if not settings.VLM_PRIMARY_KEEP_ONNX_ONLY and not trusted:
                continue
            det = self._build_detection(
                final_class=oc.get("class_name"), bbox=ob, localization="bbox",
                status="onnx_only", grade_conf=float(oc.get("conf", 0.0)),
                boosted=False, agreement=False, onnx_conf=float(oc.get("conf", 0.0)),
                vlm_conf=None, source="onnx", reasoning="", single_engine=True,
            )
            if det is not None:
                out.append(det)
        return out

    @staticmethod
    def _iou_xyxy(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        u = aa + bb - inter
        return inter / u if u > 0 else 0.0

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
