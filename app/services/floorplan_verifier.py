# =============================================
# app/services/floorplan_verifier.py
# 역할: 사전모델(CAD/평면도 폴리곤) ↔ SLAM occupancy 정합 검증
#
# 알고리즘:
#   1) 사전모델 폴리곤 리스트 → SLAM 과 동일 해상도 raster (cv2.fillPoly)
#   2) phase correlation 으로 (tx, ty) 초기 정합 (cv2.phaseCorrelate)
#   3) 미세 yaw 정합: ±10° 범위 sweep (1° 간격) → IoU 최댓값 yaw 채택
#   4) 정합 적용 후 픽셀 IoU 산출
#   5) 차이영역 추출:
#       - SLAM 에 있고 사전모델에 없음(추가)  : occ AND NOT prior
#       - 사전모델에 있고 SLAM 에 없음(누락)  : prior AND NOT occ
#      각각 connected components → 폴리곤 + 면적, area_min 임계 필터
#   6) verdict 결정:
#       IoU >= iou_ok       → OK
#       IoU >= iou_marginal → MARGINAL (mission_orchestrator 가 추가 MAPPING 1회)
#       그 외               → DIVERGENT (사용자 확인 요구)
#
# 좌표계: occupancy 픽셀 (col, row) → 월드 (col*res, row*res). 차이영역 폴리곤은 월드 좌표(미터).
# scipy 미의존, opencv + numpy 만 사용.
# =============================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


Polygon2D = Sequence[Tuple[float, float]]


class VerificationVerdict(str, Enum):
    OK = "ok"
    MARGINAL = "marginal"
    DIVERGENT = "divergent"
    NO_PRIOR_MODEL = "no_prior_model"


@dataclass
class AlignmentTransform:
    tx: float = 0.0
    ty: float = 0.0
    yaw_rad: float = 0.0
    scale: float = 1.0


@dataclass
class DiscrepancyRegion:
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    kind: str = "added"     # "added" | "missing"
    area_m2: float = 0.0


@dataclass
class VerificationResult:
    verdict: VerificationVerdict
    iou: float = 0.0
    alignment: AlignmentTransform = field(default_factory=AlignmentTransform)
    discrepancies: List[DiscrepancyRegion] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "iou": round(self.iou, 4),
            "alignment": {
                "tx": self.alignment.tx, "ty": self.alignment.ty,
                "yaw_rad": self.alignment.yaw_rad, "scale": self.alignment.scale,
            },
            "discrepancies": [
                {
                    "polygon": [list(p) for p in d.polygon],
                    "kind": d.kind,
                    "area_m2": round(d.area_m2, 3),
                }
                for d in self.discrepancies
            ],
            "detail": self.detail,
        }


@dataclass
class VerifierParams:
    iou_ok: float = 0.70
    iou_marginal: float = 0.50
    yaw_search_deg: float = 10.0
    yaw_step_deg: float = 1.0
    discrepancy_min_area_m2: float = 0.25
    poly_simplify_px: float = 2.0


class FloorplanVerifier:
    def __init__(self, params: VerifierParams | None = None) -> None:
        self.params = params or VerifierParams()

    # ── 폴리곤 → raster ──────────────────
    @staticmethod
    def _rasterize_polygons(
        polygons: List[Polygon2D],
        shape: Tuple[int, int],
        resolution_m_per_px: float,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """폴리곤 영역을 1로 채운 uint8 마스크. shape=(H, W)."""
        h, w = shape
        canvas = np.zeros((h, w), dtype=np.uint8)
        ox, oy = origin_xy
        for poly in polygons:
            if len(poly) < 3:
                continue
            pts = np.array([
                [int(round((x - ox) / resolution_m_per_px)),
                 int(round((y - oy) / resolution_m_per_px))]
                for (x, y) in poly
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], 1)
        return canvas

    @staticmethod
    def _occupancy_to_free_mask(occupancy_grid: np.ndarray) -> np.ndarray:
        """occupancy(0=free, 1=occ, -1=unknown) → 자유공간 마스크(1=free)."""
        return (occupancy_grid == 0).astype(np.uint8)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        a_b = (a.astype(bool)) & (b.astype(bool))
        a_or = (a.astype(bool)) | (b.astype(bool))
        denom = int(a_or.sum())
        if denom == 0:
            return 0.0
        return float(a_b.sum()) / float(denom)

    # ── 정합 ─────────────────────────────
    def _phase_correlate(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """phaseCorrelate 는 float32 입력. 반환 (dx, dy) 픽셀."""
        af = a.astype(np.float32)
        bf = b.astype(np.float32)
        try:
            (dx, dy), _ = cv2.phaseCorrelate(af, bf)
            return float(dx), float(dy)
        except cv2.error as e:
            logger.warning("verifier.phase_corr_failed", error=str(e))
            return 0.0, 0.0

    def _rotate_mask(self, mask: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = mask.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def _translate_mask(self, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        h, w = mask.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def _align(
        self, prior_mask: np.ndarray, slam_free: np.ndarray
    ) -> Tuple[np.ndarray, AlignmentTransform, float]:
        """prior 를 slam_free 좌표계로 정렬. (정렬된 prior, transform, IoU) 반환."""
        # 1) yaw sweep — 0 부터 시작해 ±yaw_search_deg
        best_iou = -1.0
        best_aligned = prior_mask
        best_tf = AlignmentTransform()

        yaws = np.arange(
            -self.params.yaw_search_deg,
            self.params.yaw_search_deg + 1e-6,
            self.params.yaw_step_deg,
        )

        for yaw_deg in yaws:
            rot = self._rotate_mask(prior_mask, yaw_deg) if abs(yaw_deg) > 1e-6 else prior_mask
            dx, dy = self._phase_correlate(rot, slam_free)
            shifted = self._translate_mask(rot, dx, dy)
            iou = self._iou(shifted, slam_free)
            if iou > best_iou:
                best_iou = iou
                best_aligned = shifted
                best_tf = AlignmentTransform(
                    tx=dx, ty=dy, yaw_rad=math.radians(float(yaw_deg)), scale=1.0,
                )
        return best_aligned, best_tf, max(0.0, best_iou)

    # ── 차이영역 폴리곤 추출 ──────────────
    def _extract_discrepancies(
        self,
        diff_mask: np.ndarray,
        kind: str,
        resolution_m_per_px: float,
        origin_xy: Tuple[float, float],
    ) -> List[DiscrepancyRegion]:
        if diff_mask.sum() == 0:
            return []
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask.astype(np.uint8), connectivity=8)
        out: List[DiscrepancyRegion] = []
        min_px = max(
            1,
            int(round(self.params.discrepancy_min_area_m2 / (resolution_m_per_px ** 2))),
        )
        for lbl in range(1, n_labels):
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < min_px:
                continue
            mask = (labels == lbl).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, self.params.poly_simplify_px, closed=True)
            pts = approx.reshape(-1, 2)
            if len(pts) < 3:
                continue
            polygon_world = [
                (float(p[0]) * resolution_m_per_px + origin_xy[0],
                 float(p[1]) * resolution_m_per_px + origin_xy[1])
                for p in pts
            ]
            out.append(DiscrepancyRegion(
                polygon=polygon_world,
                kind=kind,
                area_m2=area_px * (resolution_m_per_px ** 2),
            ))
        return out

    # ── 메인 ──────────────────────────────
    def verify(
        self,
        occupancy_grid: Optional[np.ndarray],
        resolution_m_per_px: float,
        prior_polygons: Optional[List[Polygon2D]] = None,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> VerificationResult:
        if not prior_polygons:
            logger.info("verifier.skip", reason="no_prior_model")
            return VerificationResult(verdict=VerificationVerdict.NO_PRIOR_MODEL)

        if occupancy_grid is None or occupancy_grid.size == 0:
            logger.warning("verifier.empty_occupancy")
            return VerificationResult(
                verdict=VerificationVerdict.DIVERGENT,
                detail={"reason": "empty_occupancy"},
            )

        slam_free = self._occupancy_to_free_mask(occupancy_grid)
        prior_mask = self._rasterize_polygons(
            list(prior_polygons), shape=slam_free.shape,
            resolution_m_per_px=resolution_m_per_px, origin_xy=origin_xy,
        )

        if prior_mask.sum() == 0:
            logger.warning("verifier.empty_prior_raster")
            return VerificationResult(
                verdict=VerificationVerdict.DIVERGENT,
                detail={"reason": "empty_prior_after_rasterize"},
            )

        aligned_prior, tf, iou = self._align(prior_mask, slam_free)

        # 차이영역
        added = (slam_free.astype(bool) & ~aligned_prior.astype(bool)).astype(np.uint8)
        missing = (aligned_prior.astype(bool) & ~slam_free.astype(bool)).astype(np.uint8)

        discrepancies: List[DiscrepancyRegion] = []
        discrepancies.extend(self._extract_discrepancies(
            added, kind="added",
            resolution_m_per_px=resolution_m_per_px, origin_xy=origin_xy,
        ))
        discrepancies.extend(self._extract_discrepancies(
            missing, kind="missing",
            resolution_m_per_px=resolution_m_per_px, origin_xy=origin_xy,
        ))

        # verdict
        if iou >= self.params.iou_ok:
            verdict = VerificationVerdict.OK
        elif iou >= self.params.iou_marginal:
            verdict = VerificationVerdict.MARGINAL
        else:
            verdict = VerificationVerdict.DIVERGENT

        result = VerificationResult(
            verdict=verdict, iou=iou, alignment=tf,
            discrepancies=discrepancies,
            detail={
                "iou_ok_th": self.params.iou_ok,
                "iou_marginal_th": self.params.iou_marginal,
                "discrepancy_count": len(discrepancies),
            },
        )
        logger.info(
            "verifier.done",
            verdict=verdict.value, iou=round(iou, 4),
            tx=round(tf.tx, 2), ty=round(tf.ty, 2),
            yaw_rad=round(tf.yaw_rad, 4),
            discrepancies=len(discrepancies),
        )
        return result
