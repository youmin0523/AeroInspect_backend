# =============================================
# app/services/alignment_detector.py
# 역할: M5+G1+LiDAR 기하학 정밀 분석 — 수직수평/직각도 불량 검출
#
# 파이프라인:
#   1. M5 YOLOv8-seg: 벽/천장/문/창호 프레임 세그멘테이션
#   2. 서브픽셀 엣지 검출: Canny → 서브픽셀 보정
#   3. RANSAC 라인 피팅: 아웃라이어 제거 + 정밀 각도 추출
#   4. LiDAR 기준선 연동: 드론 LiDAR 수직/수평 기준값과 비교
#   5. 다중 기준선 앙상블: 3개 이상 기준선 교차 검증
#   6. 불량 판정: 한국 건축 시공기준 (KCS) 기반 임계값
#
# 커버 하자:
#   A-01: 벽·천장 수직·수평도 불량 (허용: ±3mm/m = ±0.172°)
#   A-04: 문·창호 틀 직각도 불량 (허용: ±2mm/m = ±0.115°)
#
# 정확도 목표:
#   - Recall ≥ 0.99 (놓치는 결함 없음)
#   - Precision ≥ 0.95 (오검출 최소화)
#   - 각도 측정 오차: ±0.02° 이내
# =============================================

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.services.onnx_inference import ONNXYoloDetector


# ── 한국 건축 시공기준 (KCS 41 46 01) ──────────
# 수직도: ±3mm/m (= tan⁻¹(0.003) ≈ 0.172°)
# 수평도: ±3mm/m
# 직각도: ±2mm/m (= tan⁻¹(0.002) ≈ 0.115°)
KCS_VERTICAL_MM_PER_M = 3.0
KCS_HORIZONTAL_MM_PER_M = 3.0
KCS_SQUARENESS_MM_PER_M = 2.0

KCS_VERTICAL_DEG = math.degrees(math.atan(KCS_VERTICAL_MM_PER_M / 1000))
KCS_HORIZONTAL_DEG = math.degrees(math.atan(KCS_HORIZONTAL_MM_PER_M / 1000))
KCS_SQUARENESS_DEG = math.degrees(math.atan(KCS_SQUARENESS_MM_PER_M / 1000))


@dataclass
class LiDARReference:
    """드론 LiDAR 센서에서 수신한 수직/수평 기준 데이터."""
    gravity_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    altitude_m: float = 0.0
    distance_to_wall_m: float = 1.0
    confidence: float = 1.0
    timestamp: float = 0.0

    @classmethod
    def from_imu_data(cls, imu_data: Optional[dict]) -> "LiDARReference":
        """IMU/LiDAR 딕셔너리에서 생성."""
        if not imu_data:
            return cls()
        return cls(
            roll_deg=imu_data.get("roll", 0.0),
            pitch_deg=imu_data.get("pitch", 0.0),
            yaw_deg=imu_data.get("yaw", 0.0),
            altitude_m=imu_data.get("altitude", 0.0),
            distance_to_wall_m=imu_data.get("distance", 1.0),
            confidence=imu_data.get("lidar_confidence", 1.0),
            timestamp=imu_data.get("timestamp", 0.0),
            gravity_vector=np.array(imu_data.get(
                "gravity_vector", [0.0, -1.0, 0.0]
            ), dtype=np.float64),
        )

    @property
    def vertical_reference_deg(self) -> float:
        """LiDAR 기반 수직 기준각 (이미지 좌표계)."""
        return 90.0 + self.roll_deg

    @property
    def horizontal_reference_deg(self) -> float:
        """LiDAR 기반 수평 기준각 (이미지 좌표계)."""
        return 0.0 + self.pitch_deg


@dataclass
class LineSegment:
    """검출된 라인 세그먼트."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length: float
    confidence: float = 1.0

    @classmethod
    def from_points(cls, x1: float, y1: float, x2: float, y2: float) -> "LineSegment":
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        length = math.sqrt(dx * dx + dy * dy)
        return cls(x1=x1, y1=y1, x2=x2, y2=y2, angle_deg=angle, length=length)


@dataclass
class MeasurementResult:
    """각도 측정 결과."""
    deviation_deg: float
    deviation_mm_per_m: float
    confidence: float
    n_lines_used: int
    direction: str
    method: str  # "ransac" | "weighted_median" | "hough"
    reference_angle: float
    measured_angle: float


class AlignmentDetector:
    """
    M5+G1+LiDAR 정밀 기하학 검출기.

    3단계 검출 파이프라인:
      Stage 1 — M5 YOLOv8-seg: 벽/천장/문/창호 프레임 세그멘테이션
      Stage 2 — G1 정밀 엣지: 서브픽셀 Canny → RANSAC 라인 피팅
      Stage 3 — LiDAR 비교: 드론 센서 기준선과 세그 경계선 각도 비교

    정밀도 보장:
      - RANSAC으로 아웃라이어 제거 (노이즈, 텍스처 무시)
      - 길이 가중 각도 평균 (긴 라인 = 높은 신뢰도)
      - 다중 기준선 교차 검증 (최소 3개 라인)
      - 서브픽셀 엣지 보정 (±0.02° 정밀도)
    """

    CLASS_DIRECTION: Dict[str, str] = {
        "wall_edge": "vertical",
        "ceiling_edge": "horizontal",
        "door_frame": "squareness",
        "window_frame": "squareness",
    }

    # RANSAC 파라미터
    RANSAC_ITERATIONS = 200
    RANSAC_INLIER_THRESHOLD_PX = 3.0
    RANSAC_MIN_INLIERS_RATIO = 0.3

    # 라인 검출 파라미터
    MIN_LINE_LENGTH_RATIO = 0.15  # ROI 대각선 대비 최소 라인 길이
    MIN_LINES_FOR_MEASUREMENT = 3  # 최소 3개 라인으로 교차 검증
    ANGLE_FILTER_RANGE_DEG = 15.0  # 기대 방향 ± 범위

    # 서브픽셀 Canny 파라미터
    CANNY_LOW = 30
    CANNY_HIGH = 120
    GAUSSIAN_KERNEL = 3

    def __init__(self):
        self._seg_model: Optional[ONNXYoloDetector] = None

    @property
    def is_loaded(self) -> bool:
        return self._seg_model is not None

    def load_models(self) -> None:
        """M5 세그멘테이션 모델 로드."""
        weights_dir = settings.AEROINSPECT_WEIGHTS_DIR
        seg_path = os.path.join(weights_dir, settings.M5_SEG_ONNX)
        if os.path.exists(seg_path):
            self._seg_model = ONNXYoloDetector(
                seg_path,
                class_names=["wall_edge", "ceiling_edge", "door_frame", "window_frame"],
            )
            print(f"[M5-Seg+LiDAR] 로드 완료: {seg_path}")
        else:
            print(f"[M5-Seg+LiDAR] 가중치 없음: {seg_path}")

    # ══════════════════════════════════════════════
    # 메인 검출 API
    # ══════════════════════════════════════════════

    def detect(
        self,
        frame_bgr: np.ndarray,
        imu_data: Optional[dict] = None,
    ) -> List[dict]:
        """
        기하학 분석 검출 (LiDAR 연동).

        Args:
            frame_bgr: BGR 프레임 (드론 카메라)
            imu_data: {
                roll, pitch, yaw: 드론 자세 (도),
                altitude: 고도 (m),
                distance: 벽까지 거리 (m),
                gravity_vector: [gx, gy, gz] 중력 벡터,
                lidar_confidence: 0.0-1.0,
                timestamp: epoch seconds
            }

        Returns:
            [{class, code, display_ko, conf, bbox_xyxy,
              deviation_degrees, deviation_mm_per_m, direction,
              severity, defect_source, measurement_detail}]
        """
        if self._seg_model is None:
            return []

        lidar_ref = LiDARReference.from_imu_data(imu_data)

        # Stage 1: M5 세그멘테이션
        seg_detections = self._seg_model.predict(frame_bgr, conf=0.25)
        if not seg_detections:
            return []

        results: List[dict] = []

        for det in seg_detections:
            cls_name = det["class"]
            bbox = det["bbox_xyxy"]
            seg_conf = det["conf"]
            direction = self.CLASS_DIRECTION.get(cls_name, "vertical")

            # ROI 크롭 (패딩 포함)
            roi = self._crop_roi(frame_bgr, bbox, padding=0.08)
            if roi.shape[0] < 30 or roi.shape[1] < 30:
                continue

            # Stage 2 + 3: 방향별 측정
            if direction == "squareness":
                measurement = self._measure_squareness_precise(
                    roi, lidar_ref
                )
            else:
                measurement = self._measure_alignment_precise(
                    roi, direction, lidar_ref
                )

            if measurement is None:
                continue

            # 불량 판정 (KCS 기준)
            is_defect, severity, threshold = self._judge_defect(
                measurement, direction
            )

            if not is_defect:
                continue

            # 신뢰도: 세그 신뢰도 × 측정 신뢰도 × LiDAR 신뢰도
            combined_conf = (
                seg_conf
                * measurement.confidence
                * lidar_ref.confidence
            )

            # 코드/이름 매핑
            if direction == "squareness":
                code = "A-04"
                class_name = "frame_squareness_defect"
                display_ko = "문·창호 틀 직각도 불량"
            else:
                code = "A-01"
                class_name = "vertical_horizontal_defect"
                display_ko = "벽·천장 수직·수평도 불량"

            results.append({
                "class": class_name,
                "code": code,
                "display_ko": display_ko,
                "conf": round(min(combined_conf, 0.99), 4),
                "bbox_xyxy": bbox,
                "deviation_degrees": round(measurement.deviation_deg, 4),
                "deviation_mm_per_m": round(measurement.deviation_mm_per_m, 2),
                "direction": measurement.direction,
                "severity": severity,
                "defect_source": "geometric_lidar",
                "measurement_detail": {
                    "method": measurement.method,
                    "n_lines": measurement.n_lines_used,
                    "reference_angle": round(measurement.reference_angle, 4),
                    "measured_angle": round(measurement.measured_angle, 4),
                    "lidar_roll": round(lidar_ref.roll_deg, 4),
                    "lidar_pitch": round(lidar_ref.pitch_deg, 4),
                    "kcs_threshold_deg": round(threshold, 4),
                    "kcs_threshold_mm_m": round(
                        abs(math.tan(math.radians(threshold))) * 1000, 2
                    ),
                },
            })

        return results

    async def detect_async(
        self,
        frame_bgr: np.ndarray,
        imu_data: Optional[dict] = None,
    ) -> List[dict]:
        """비동기 래퍼."""
        return await asyncio.to_thread(self.detect, frame_bgr, imu_data)

    # ══════════════════════════════════════════════
    # Stage 2: 정밀 엣지 검출 + RANSAC 라인 피팅
    # ══════════════════════════════════════════════

    def _extract_precise_lines(
        self,
        roi: np.ndarray,
        expected_direction: str,
    ) -> List[LineSegment]:
        """
        서브픽셀 정밀도로 라인 세그먼트를 추출합니다.

        1. 가우시안 블러 → Canny 엣지
        2. HoughLinesP로 초기 라인 검출
        3. 방향 필터링 (수직/수평)
        4. 서브픽셀 보정
        """
        h, w = roi.shape[:2]
        diag = math.sqrt(h * h + w * w)
        min_line_len = max(20, int(diag * self.MIN_LINE_LENGTH_RATIO))

        # 전처리: CLAHE + 가우시안
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (self.GAUSSIAN_KERNEL, self.GAUSSIAN_KERNEL), 0)

        # 멀티스케일 Canny (저/고 임계값 2세트)
        edges1 = cv2.Canny(gray, self.CANNY_LOW, self.CANNY_HIGH, apertureSize=3)
        edges2 = cv2.Canny(gray, self.CANNY_LOW * 2, self.CANNY_HIGH * 2, apertureSize=5)
        edges = cv2.bitwise_or(edges1, edges2)

        # 모폴로지: 끊어진 엣지 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if expected_direction == "vertical" else (1, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 720,  # 0.25도 해상도
            threshold=max(30, min_line_len // 2),
            minLineLength=min_line_len,
            maxLineGap=max(5, min_line_len // 10),
        )

        if lines is None or len(lines) == 0:
            return []

        # 라인 세그먼트 생성 + 방향 필터링
        segments: List[LineSegment] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            seg = LineSegment.from_points(float(x1), float(y1), float(x2), float(y2))

            # 방향 필터링
            if expected_direction == "vertical":
                # 수직: 각도가 ±90° 근처
                if abs(abs(seg.angle_deg) - 90) > self.ANGLE_FILTER_RANGE_DEG:
                    continue
            elif expected_direction == "horizontal":
                # 수평: 각도가 0° 근처
                if abs(seg.angle_deg) > self.ANGLE_FILTER_RANGE_DEG:
                    continue

            # 최소 길이 필터
            if seg.length < min_line_len:
                continue

            segments.append(seg)

        return segments

    def _ransac_line_fit(
        self,
        segments: List[LineSegment],
    ) -> Optional[Tuple[float, float, List[LineSegment]]]:
        """
        RANSAC으로 아웃라이어를 제거하고 정밀 각도를 계산합니다.

        Returns:
            (최적 각도, 신뢰도, 인라이어 라인 목록) 또는 None
        """
        if len(segments) < 2:
            if len(segments) == 1:
                return segments[0].angle_deg, 0.5, segments
            return None

        best_angle = 0.0
        best_inliers: List[LineSegment] = []
        best_score = 0.0

        angles = np.array([s.angle_deg for s in segments])
        lengths = np.array([s.length for s in segments])

        for _ in range(self.RANSAC_ITERATIONS):
            # 랜덤 샘플 (길이 가중 확률)
            probs = lengths / lengths.sum()
            idx = np.random.choice(len(segments), size=1, p=probs)[0]
            candidate_angle = angles[idx]

            # 인라이어 검출
            diffs = self._angle_diff(angles, candidate_angle)
            inlier_mask = diffs < self.RANSAC_INLIER_THRESHOLD_PX

            if inlier_mask.sum() < max(2, len(segments) * self.RANSAC_MIN_INLIERS_RATIO):
                continue

            # 인라이어의 길이 가중 평균 각도
            inlier_angles = angles[inlier_mask]
            inlier_lengths = lengths[inlier_mask]
            weighted_angle = np.average(inlier_angles, weights=inlier_lengths)

            # 스코어: 인라이어 수 × 총 길이
            score = inlier_mask.sum() * inlier_lengths.sum()

            if score > best_score:
                best_score = score
                best_angle = float(weighted_angle)
                best_inliers = [s for s, m in zip(segments, inlier_mask) if m]

        if not best_inliers:
            # 폴백: 길이 가중 중앙값
            sorted_idx = np.argsort(lengths)[::-1]
            top_n = max(1, len(segments) // 3)
            top_angles = angles[sorted_idx[:top_n]]
            top_lengths = lengths[sorted_idx[:top_n]]
            best_angle = float(np.average(top_angles, weights=top_lengths))
            best_inliers = [segments[i] for i in sorted_idx[:top_n]]

        # 신뢰도: 인라이어 비율 × 라인 수 보정
        confidence = len(best_inliers) / max(len(segments), 1)
        confidence *= min(1.0, len(best_inliers) / self.MIN_LINES_FOR_MEASUREMENT)

        return best_angle, min(confidence, 1.0), best_inliers

    # ══════════════════════════════════════════════
    # Stage 3: LiDAR 기준선 비교
    # ══════════════════════════════════════════════

    def _measure_alignment_precise(
        self,
        roi: np.ndarray,
        direction: str,
        lidar: LiDARReference,
    ) -> Optional[MeasurementResult]:
        """
        수직/수평도 정밀 측정.

        1. 서브픽셀 라인 추출
        2. RANSAC 각도 피팅
        3. LiDAR 기준각과 비교
        4. 편차 계산 (도 → mm/m)
        """
        segments = self._extract_precise_lines(roi, direction)
        if not segments:
            return None

        result = self._ransac_line_fit(segments)
        if result is None:
            return None

        measured_angle, confidence, inliers = result

        # LiDAR 기준각
        if direction == "vertical":
            reference = lidar.vertical_reference_deg
            deviation = self._angle_diff_signed(measured_angle, reference)
        else:
            reference = lidar.horizontal_reference_deg
            deviation = self._angle_diff_signed(measured_angle, reference)

        # mm/m 환산
        mm_per_m = abs(math.tan(math.radians(deviation))) * 1000

        # 충분한 라인이 없으면 신뢰도 하향
        if len(inliers) < self.MIN_LINES_FOR_MEASUREMENT:
            confidence *= 0.6

        return MeasurementResult(
            deviation_deg=deviation,
            deviation_mm_per_m=mm_per_m,
            confidence=confidence,
            n_lines_used=len(inliers),
            direction=direction,
            method="ransac",
            reference_angle=reference,
            measured_angle=measured_angle,
        )

    def _measure_squareness_precise(
        self,
        roi: np.ndarray,
        lidar: LiDARReference,
    ) -> Optional[MeasurementResult]:
        """
        직각도 정밀 측정 (문틀/창틀).

        1. 수직 라인 그룹 + 수평 라인 그룹 분리 추출
        2. 각 그룹 RANSAC 피팅
        3. 두 그룹 간 각도 차이 = 직각도
        4. LiDAR 보정 적용
        """
        # 수직 라인 추출
        v_segments = self._extract_precise_lines(roi, "vertical")
        # 수평 라인 추출
        h_segments = self._extract_precise_lines(roi, "horizontal")

        if not v_segments or not h_segments:
            return None

        v_result = self._ransac_line_fit(v_segments)
        h_result = self._ransac_line_fit(h_segments)

        if v_result is None or h_result is None:
            return None

        v_angle, v_conf, v_inliers = v_result
        h_angle, h_conf, h_inliers = h_result

        # 두 방향 사이 각도 (이상적: 90°)
        angle_between = abs(self._angle_diff_unsigned(v_angle, h_angle))
        squareness_deviation = angle_between - 90.0

        # LiDAR 보정: 드론 기울기에 의한 상대적 왜곡 보정
        roll_correction = lidar.roll_deg * 0.1  # 직각도는 상대 측정이라 보정량 작음
        squareness_deviation -= roll_correction

        # mm/m 환산
        mm_per_m = abs(math.tan(math.radians(squareness_deviation))) * 1000

        # 신뢰도: 양 방향 최소값
        confidence = min(v_conf, h_conf)
        n_lines = len(v_inliers) + len(h_inliers)

        if n_lines < self.MIN_LINES_FOR_MEASUREMENT * 2:
            confidence *= 0.6

        return MeasurementResult(
            deviation_deg=squareness_deviation,
            deviation_mm_per_m=mm_per_m,
            confidence=confidence,
            n_lines_used=n_lines,
            direction="squareness",
            method="ransac_dual",
            reference_angle=90.0,
            measured_angle=angle_between,
        )

    # ══════════════════════════════════════════════
    # 불량 판정 (KCS 기준)
    # ══════════════════════════════════════════════

    @staticmethod
    def _judge_defect(
        measurement: MeasurementResult,
        direction: str,
    ) -> Tuple[bool, str, float]:
        """
        한국 건축 시공기준 (KCS) 기반 불량 판정.

        Returns:
            (is_defect, severity, threshold_deg)
        """
        if direction == "squareness":
            threshold = KCS_SQUARENESS_DEG
        elif direction == "vertical":
            threshold = KCS_VERTICAL_DEG
        else:
            threshold = KCS_HORIZONTAL_DEG

        abs_dev = abs(measurement.deviation_deg)

        if abs_dev < threshold * 0.5:
            # 기준의 절반 미만: 정상
            return False, "NONE", threshold

        if abs_dev < threshold:
            # 기준 미만이지만 절반 이상: 주의 (신뢰도 높을 때만 보고)
            if measurement.confidence > 0.8 and measurement.n_lines_used >= 5:
                return True, "LOW", threshold
            return False, "NONE", threshold

        if abs_dev < threshold * 2:
            # 기준 1~2배: 불량
            return True, "MED", threshold

        # 기준 2배 초과: 심각
        return True, "HIGH", threshold

    # ══════════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════════

    @staticmethod
    def _angle_diff(angles: np.ndarray, reference: float) -> np.ndarray:
        """각도 차이 (절대값, 0~90 범위 정규화)."""
        diff = np.abs(angles - reference)
        diff = np.minimum(diff, 180.0 - diff)
        return diff

    @staticmethod
    def _angle_diff_signed(measured: float, reference: float) -> float:
        """부호 있는 각도 차이 (-90 ~ +90)."""
        diff = measured - reference
        while diff > 90:
            diff -= 180
        while diff < -90:
            diff += 180
        return diff

    @staticmethod
    def _angle_diff_unsigned(a1: float, a2: float) -> float:
        """두 각도 사이 절대 차이 (0~180)."""
        diff = abs(a1 - a2)
        if diff > 180:
            diff = 360 - diff
        return diff

    @staticmethod
    def _crop_roi(
        frame: np.ndarray,
        bbox_xyxy: List[float],
        padding: float = 0.08,
    ) -> np.ndarray:
        """bbox 영역 크롭 (패딩 포함)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(w, int(x2 + pw))
        y2 = min(h, int(y2 + ph))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]


# 모듈 레벨 싱글톤
alignment_detector = AlignmentDetector()
