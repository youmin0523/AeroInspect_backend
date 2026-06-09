# =============================================
# app/services/onnx_inference.py
# 역할: ONNX Runtime 기반 통합 추론 래퍼
#       - ONNXYoloDetector: YOLO ONNX bbox 검출 (Stage 1)
#       - ONNXResNetClassifier: ResNet50 ONNX ROI 분류 (Stage 2)
#       - ONNXUNetSegmenter: U-Net ONNX 열화상 세그멘테이션
#       - ONNXPatchCoreDetector: PatchCore ONNX 이상 탐지
#
# 전 모델 ONNX Runtime 추론으로 통일:
#   - PyTorch/ultralytics 런타임 의존성 제거
#   - CUDA EP / TensorRT EP / CPU EP 자동 전환
#   - FP16 양자화 지원
# =============================================

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # 테스트 환경에서 onnxruntime 미설치 시 graceful 처리


# ═══════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════

def _create_session(onnx_path: str) -> "ort.InferenceSession":
    """CUDA → CPU 자동 선택으로 ONNX 세션 생성."""
    if ort is None:
        raise ImportError(
            "onnxruntime이 설치되지 않았습니다. "
            "pip install onnxruntime-gpu 또는 pip install onnxruntime"
        )

    providers: list = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }))
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    return ort.InferenceSession(onnx_path, sess_options, providers=providers)


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """NumPy NMS 구현. boxes: [N, 4] xyxy, scores: [N]."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# ═══════════════════════════════════════════════
# ONNXYoloDetector — Stage 1 (위치 검출)
# ═══════════════════════════════════════════════

class ONNXYoloDetector:
    """
    YOLOv8 ONNX 검출기.
    ultralytics ONNX export 포맷: 입력 [1,3,H,W], 출력 [1, 4+nc, num_anchors]
    """

    def __init__(self, onnx_path: str, class_names: List[str], input_size: int = 640):
        self.session = _create_session(onnx_path)
        self.class_names = class_names
        self.nc = len(class_names)
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def preprocess(
        self, frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int, int, int]:
        """BGR 이미지 → YOLO 입력 텐서 + letterbox 스케일 정보."""
        h0, w0 = frame_bgr.shape[:2]
        scale = min(self.input_size / h0, self.input_size / w0)
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8,
        )
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR→RGB, HWC→CHW, 0-255→0-1
        blob = (
            canvas[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, pad_w, pad_h, w0, h0

    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[dict]:
        """YOLO 출력 → NMS → 검출 리스트."""
        out = output[0].T  # [num_anchors, 4+nc(+32 mask coef for seg)]
        boxes = out[:, :4]
        # seg 모델은 4+nc 뒤에 32개 mask 계수가 붙음 → nc개만 슬라이스(mask coef 제외).
        # detect 모델은 채널이 정확히 4+nc라 [4:4+nc] == [4:] 로 동일하게 안전.
        scores = out[:, 4:4 + self.nc]

        max_scores = scores.max(axis=1)
        mask = max_scores >= conf_threshold
        boxes, scores, max_scores = boxes[mask], scores[mask], max_scores[mask]

        if len(boxes) == 0:
            return []

        class_ids = scores.argmax(axis=1)

        # xywh → xyxy
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # letterbox 역변환
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # 클래스별 NMS: 서로 다른 클래스의 겹치는 박스가 상호 억제되지 않도록
        # 클래스 id마다 큰 좌표 오프셋을 더해 단일 NMS로 처리(표준 YOLO 트릭).
        # 단일 클래스(nc==1)면 오프셋이 모두 0 → 기존 동작과 동일.
        max_coord = float(max(orig_w, orig_h)) + 1.0
        offset = class_ids.astype(np.float32) * max_coord
        xyxy_offset = xyxy + offset[:, None]
        keep = _nms_numpy(xyxy_offset, max_scores, iou_threshold)

        results = []
        for i in keep:
            results.append({
                "class": self.class_names[class_ids[i]],
                "class_id": int(class_ids[i]),
                "conf": float(max_scores[i]),
                "bbox_xyxy": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })
        return results

    @property
    def supports_batch(self) -> bool:
        """ONNX 입력의 batch 축이 동적(N>1 허용)인지 여부.

        ultralytics를 dynamic=True 로 export 하면 입력 shape[0] 이 정수 1 이
        아니라 심볼릭 문자열('batch')로 남는다. 이 경우 여러 타일/프레임을
        하나의 배치로 묶어 단일 추론 호출이 가능하다.
        고정 export(shape[0]==1)면 진짜 배치가 불가능하므로 False.
        """
        try:
            n = self.session.get_inputs()[0].shape[0]
        except Exception:
            return False
        # 정수 1 이면 고정 batch=1 export → 배치 불가.
        # 심볼릭('batch') 또는 정수 >1 이면 동적 → 배치 가능.
        return not (isinstance(n, int) and n == 1)

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> List[dict]:
        """전체 파이프라인: 전처리 → ONNX 추론 → NMS → 결과."""
        blob, scale, pad_w, pad_h, orig_w, orig_h = self.preprocess(frame_bgr)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return self.postprocess(output, scale, pad_w, pad_h, orig_w, orig_h, conf, iou)

    def predict_batch(
        self,
        frames_bgr: List[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> List[List[dict]]:
        """여러 프레임을 단일 ONNX 추론으로 일괄 처리.

        각 프레임을 letterbox 전처리(모두 input_size×input_size 로 정규화)한 뒤
        [N,3,S,S] 로 스택하여 ONNX 세션을 *한 번만* 호출한다. 출력 [N, 4+nc, A]
        를 프레임별로 분리해 각자의 letterbox 스케일/패딩으로 postprocess 한다.

        입력 batch 축이 고정(batch=1)인 모델이면 진짜 배치가 불가능하므로
        프레임별 순차 predict 로 폴백한다(supports_batch 로 판단).

        Returns:
            프레임별 검출 리스트의 리스트 (입력 순서 보존).
        """
        if not frames_bgr:
            return []

        # 고정 batch=1 export → 진짜 배치 불가. 순차 폴백.
        if not self.supports_batch:
            return [self.predict(f, conf=conf, iou=iou) for f in frames_bgr]

        blobs: List[np.ndarray] = []
        metas: List[Tuple[float, int, int, int, int]] = []
        for frame in frames_bgr:
            blob, scale, pad_w, pad_h, orig_w, orig_h = self.preprocess(frame)
            blobs.append(blob)  # blob: [1,3,S,S]
            metas.append((scale, pad_w, pad_h, orig_w, orig_h))

        # [N,3,S,S] 단일 텐서로 결합 → ONNX 1회 호출
        batch_blob = np.concatenate(blobs, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch_blob})[0]

        results: List[List[dict]] = []
        for i, (scale, pad_w, pad_h, orig_w, orig_h) in enumerate(metas):
            # postprocess 는 output[0] 으로 첫 배치 요소를 집으므로,
            # i번째 요소만 잘라 다시 [1, 4+nc, A] 형태로 넣어 재사용.
            single = outputs[i : i + 1]
            results.append(
                self.postprocess(
                    single, scale, pad_w, pad_h, orig_w, orig_h, conf, iou,
                )
            )
        return results


# ═══════════════════════════════════════════════
# ONNXResNetClassifier — Stage 2 (ROI 정밀 분류)
# ═══════════════════════════════════════════════

class ONNXResNetClassifier:
    """
    ResNet50 ONNX 분류기.
    YOLO Stage 1이 검출한 ROI 크롭 → ImageNet 정규화 → 분류.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str,
        class_names: List[str],
        input_size: int = 224,
    ):
        self.session = _create_session(onnx_path)
        self.class_names = class_names
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        """ROI 크롭 → ResNet 입력 텐서 [1, 3, 224, 224]."""
        resized = cv2.resize(roi_bgr, (self.input_size, self.input_size))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD
        blob = rgb.transpose(2, 0, 1)
        return np.expand_dims(blob, axis=0).astype(np.float32)

    def classify(
        self, roi_bgr: np.ndarray,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        ROI 크롭 분류 → (top1_class, top1_conf, top3_list).

        모델의 output class 수가 self.class_names보다 많은 경우
        (예: 모델은 3 클래스 출력인데 매핑은 1 클래스만 제공) 안전 처리:
        - 매핑된 인덱스만 추려서 top3 구성
        - class_names 범위 밖 인덱스는 'unknown_<idx>' 폴백
        """
        blob = self.preprocess(roi_bgr)
        logits = self.session.run([self.output_name], {self.input_name: blob})[0][0]

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        n_classes_available = len(self.class_names)
        n_top = min(3, len(probs))
        top_idx = probs.argsort()[::-1][:n_top]
        top: List[Tuple[str, float]] = []
        for i in top_idx:
            if i < n_classes_available:
                top.append((self.class_names[int(i)], float(probs[i])))
            else:
                top.append((f"unknown_class_{int(i)}", float(probs[i])))

        return top[0][0], top[0][1], top


# ═══════════════════════════════════════════════
# ONNXUNetSegmenter — 열화상 세그멘테이션
# ═══════════════════════════════════════════════

class ONNXUNetSegmenter:
    """
    U-Net ONNX 세그멘테이션 (열화상 단열 검출).
    입력: 온도맵 float32 → 정규화 → 3ch 복제 → EfficientNet 호환
    출력: 멀티클래스 세그멘테이션 마스크
    """

    def __init__(self, onnx_path: str, class_names: List[str]):
        self.session = _create_session(onnx_path)
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def segment(
        self, temp_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        온도맵 → (class_mask, prob_map).

        Args:
            temp_map: float32 [H, W] 온도맵 (°C)

        Returns:
            class_mask: uint8 [H, W] 클래스 인덱스 (0=배경)
            prob_map: float32 [C, H, W] 클래스별 확률
        """
        temp_norm = (temp_map - temp_map.mean()) / (temp_map.std() + 1e-6)
        input_3ch = np.stack([temp_norm] * 3, axis=0).astype(np.float32)
        blob = np.expand_dims(input_3ch, axis=0)

        logits = self.session.run([self.output_name], {self.input_name: blob})[0][0]

        # Softmax → 확률맵
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        prob_map = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        class_mask = prob_map.argmax(axis=0).astype(np.uint8)
        return class_mask, prob_map


# ═══════════════════════════════════════════════
# ONNXPatchCoreDetector — 이상 탐지
# ═══════════════════════════════════════════════

class ONNXPatchCoreDetector:
    """
    PatchCore ONNX 이상 탐지 (Anomalib export).
    정상 패턴과 비교하여 이상 점수 + 이상 마스크 생성.
    """

    def __init__(self, onnx_path: str, threshold: float = 0.5):
        self.session = _create_session(onnx_path)
        self.threshold = threshold
        self.input_name = self.session.get_inputs()[0].name
        # 모델의 실제 입력 H/W를 그래프에서 자동 추출 — 하드코딩 시 export shape와
        # 어긋나면 매 frame INVALID_ARGUMENT로 추론 전체 실패. shape: [N, C, H, W].
        # dynamic axis(str)면 fallback 224 (anomalib 기본 backbone).
        shape = self.session.get_inputs()[0].shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 224
        self.input_w = shape[3] if isinstance(shape[3], int) else 224

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def detect(
        self, frame_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        이상 탐지 → (anomaly_mask, anomaly_score).

        Args:
            frame_bgr: BGR 입력 이미지

        Returns:
            anomaly_mask: uint8 이상 영역 이진 마스크 (0 또는 255)
            anomaly_score: float 전체 이상 점수 (0.0~1.0)
        """
        # 전처리: 모델 입력 shape에 맞춰 resize + normalize.
        # cv2.resize는 (W, H) 순서. self.input_w/h는 ONNX 그래프에서 추출.
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        blob = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: blob})

        # anomalib export 버전별 출력 형식 분기
        if len(outputs) >= 4:
            # anomalib 2.x (surface.onnx): [pred_score, pred_label, anomaly_map, pred_mask]
            s = outputs[0]
            score = float(s.item() if s.size == 1 else s.flatten()[0])
            anomaly_map = outputs[2][0]
        elif len(outputs) >= 2:
            # anomalib 1.x: [anomaly_map, pred_score]
            anomaly_map = outputs[0][0]
            score = float(outputs[1][0]) if outputs[1].size == 1 else float(outputs[1][0][0])
        else:
            anomaly_map = outputs[0][0]
            score = float(anomaly_map.max())

        # anomaly_map → 이진 마스크
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[0]
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        mask = (norm_map > self.threshold).astype(np.uint8) * 255

        # score 는 계약상 0.0~1.0 (docstring). export 버전/폴백(raw max)에서 범위를 벗어나면
        # grading 에서 conf>=CONFIRMED_STRONG 로 잘못 자동 CONFIRM 될 수 있어 클램프.
        score = max(0.0, min(1.0, score))
        return mask, score


# ═══════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════

def crop_roi_xyxy(
    frame: np.ndarray,
    bbox_xyxy: List[float],
    padding: float = 0.1,
) -> np.ndarray:
    """bbox(xyxy 픽셀) 영역을 패딩과 함께 크롭. 빈 크롭 시 원본 반환.

    주의: app.utils.image_utils.crop_roi 는 xywh-normalized 규약의 *다른* 함수다.
    혼동 방지를 위해 이 픽셀-xyxy 버전은 _xyxy 접미사로 구분한다.
    """
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


__all__ = [
    "ONNXYoloDetector",
    "ONNXResNetClassifier",
    "ONNXUNetSegmenter",
    "ONNXPatchCoreDetector",
    "crop_roi",
]
