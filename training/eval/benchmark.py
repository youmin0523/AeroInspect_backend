# =============================================
# eval/benchmark.py
# ONNX 모델별 + 전체 파이프라인 추론 속도 벤치마크
#
# 사용법:
#   cd backend/training
#   python eval/benchmark.py
# =============================================

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime 미설치. pip install onnxruntime-gpu")
    exit(1)


WEIGHTS_DIR = Path("../models_weights")

MODELS = {
    "M1-YOLO": ("m1_yolo_structural.onnx", [1, 3, 640, 640]),
    "M1-ResNet": ("m1_resnet_crack_classifier.onnx", [1, 3, 224, 224]),
    "M2-YOLO": ("m2_yolo_surface.onnx", [1, 3, 640, 640]),
    "M2-ResNet": ("m2_resnet_surface_classifier.onnx", [1, 3, 224, 224]),
    "M3-YOLO": ("m3_yolo_floor_window.onnx", [1, 3, 640, 640]),
    "M3-ResNet": ("m3_resnet_floor_window_classifier.onnx", [1, 3, 224, 224]),
    "M4-UNet": ("m4_unet_thermal_insulation.onnx", [1, 3, 192, 256]),
    "M5-Seg": ("m5_yolo_seg_frames.onnx", [1, 3, 640, 640]),
    "M6-PatchCore": ("m6_patchcore_surface.onnx", [1, 3, 256, 256]),
}

NUM_RUNS = 100
WARMUP = 10


def benchmark_model(onnx_path: str, input_shape: list) -> dict:
    """단일 ONNX 모델 추론 속도 측정."""
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(WARMUP):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "p50_ms": round(np.percentile(times, 50), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
        "p99_ms": round(np.percentile(times, 99), 2),
    }


def main():
    device = "GPU" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"
    print(f"=== ONNX 추론 속도 벤치마크 (device={device}, runs={NUM_RUNS}) ===\n")
    print(f"{'모델':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-" * 60)

    total_mean = 0.0
    for name, (filename, shape) in MODELS.items():
        path = WEIGHTS_DIR / filename
        if not path.exists():
            print(f"{name:<20} {'SKIP':>8} (파일 없음)")
            continue

        result = benchmark_model(str(path), shape)
        total_mean += result["mean_ms"]
        print(
            f"{name:<20} {result['mean_ms']:>7.1f}ms "
            f"{result['p50_ms']:>7.1f}ms "
            f"{result['p95_ms']:>7.1f}ms "
            f"{result['p99_ms']:>7.1f}ms"
        )

    print(f"\n{'합계 (순차)':<20} {total_mean:>7.1f}ms")
    print(f"{'Tier1 (M1+M2)':<20} {total_mean * 0.4:>7.1f}ms (추정)")
    print(f"\n벤치마크 완료.")


if __name__ == "__main__":
    main()
