# =============================================
# app/core/metrics.py
# 역할: Prometheus 메트릭 수집 (prometheus_client)
#       - HTTP 요청 수/지연 (method, path, status 별)
#       - 스트림 추론 워커 카운터 (submitted/processed/dropped)
#       - 결함 탐지 이벤트 누적 (severity 별)
#       - LiDAR / telemetry 헬스 gauge
#
# /metrics 엔드포인트에서 일반 텍스트(OpenMetrics)로 노출.
# Grafana → Prometheus datasource 붙이면 바로 그래프 뜸.
#
# 주의: Counter 는 영원히 증가만 함. Gauge 는 현재값.
# =============================================

from __future__ import annotations

import time
from typing import Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# ── 커스텀 레지스트리 (앱 전역 싱글톤) ──────────────
# default registry 대신 명시적 registry 를 쓰면 테스트 격리가 쉬워짐.
registry = CollectorRegistry()


# ── HTTP 메트릭 ─────────────────────────────────────
http_requests_total = Counter(
    "aeroinspect_http_requests_total",
    "Total number of HTTP requests handled",
    labelnames=("method", "path", "status"),
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "aeroinspect_http_request_duration_seconds",
    "HTTP request processing time in seconds",
    labelnames=("method", "path"),
    registry=registry,
    # 대시보드에 적당한 버킷 — 0.01초 ~ 10초
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ── 스트림 추론 워커 ────────────────────────────────
stream_frames_submitted = Counter(
    "aeroinspect_stream_frames_submitted_total",
    "WebSocket으로 수신된 총 프레임 수 (스킵 포함)",
    registry=registry,
)

stream_frames_processed = Counter(
    "aeroinspect_stream_frames_processed_total",
    "실제로 추론된 프레임 수",
    registry=registry,
)

stream_frames_dropped = Counter(
    "aeroinspect_stream_frames_dropped_total",
    "드롭 큐 가득 차서 버려진 프레임 수",
    registry=registry,
)


# ── 결함 탐지 ───────────────────────────────────────
defect_detected_total = Counter(
    "aeroinspect_defect_detected_total",
    "AI 파이프라인이 탐지한 결함 누적 (심각도별)",
    labelnames=("severity",),  # HIGH / MED / LOW
    registry=registry,
)


# ── LLM 호출 (외부 의존성 중 가장 비싸고 실패 잦음) ──
llm_calls_total = Counter(
    "aeroinspect_llm_calls_total",
    "외부 LLM 호출 누적 (provider/operation/status 별)",
    labelnames=("provider", "operation", "status"),  # status: success | error
    registry=registry,
)

llm_call_duration_seconds = Histogram(
    "aeroinspect_llm_call_duration_seconds",
    "외부 LLM 호출 지연 (초)",
    labelnames=("provider", "operation"),
    registry=registry,
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0),
)

llm_tokens_total = Counter(
    "aeroinspect_llm_tokens_total",
    "외부 LLM 토큰 사용 누적 (provider/operation/kind 별)",
    labelnames=("provider", "operation", "kind"),  # kind: prompt | completion
    registry=registry,
)


class track_llm_call:
    """LLM 호출 지연/성공·실패를 기록하는 async 컨텍스트 매니저.

    사용:
        async with track_llm_call("openai", "chat"):
            ... await client.create(...) ...
    예외가 발생하면 status=error 로, 정상 종료면 success 로 카운트한다.
    """

    def __init__(self, provider: str, operation: str) -> None:
        self.provider = provider
        self.operation = operation
        self._start = 0.0

    async def __aenter__(self):
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self._start
        llm_call_duration_seconds.labels(self.provider, self.operation).observe(duration)
        status = "success" if exc_type is None else "error"
        llm_calls_total.labels(self.provider, self.operation, status).inc()
        return False  # 예외 전파


def record_llm_tokens(provider: str, operation: str, *, prompt: int = 0, completion: int = 0) -> None:
    """LLM 토큰 사용량 기록 (응답 usage 가 있을 때 호출)."""
    if prompt:
        llm_tokens_total.labels(provider, operation, "prompt").inc(prompt)
    if completion:
        llm_tokens_total.labels(provider, operation, "completion").inc(completion)


# ── 센서 헬스 (Gauge: 현재값) ───────────────────────
lidar_distance_meters = Gauge(
    "aeroinspect_lidar_distance_meters",
    "최신 LiDAR 거리 (m). 미수신 시 -1.",
    registry=registry,
)

telemetry_cache_age_seconds = Gauge(
    "aeroinspect_telemetry_cache_age_seconds",
    "텔레메트리 캐시 마지막 갱신 후 경과 (초). 미수신 시 -1.",
    registry=registry,
)

stream_worker_queue_size = Gauge(
    "aeroinspect_stream_worker_queue_size",
    "현재 추론 큐 적재량 (maxsize=1)",
    registry=registry,
)


# ── HTTP 미들웨어 ────────────────────────────────────
class PrometheusMiddleware(BaseHTTPMiddleware):
    """요청 수/지연 자동 기록."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # /metrics 자체는 메트릭에서 제외 (Prom 스크래핑 자가 반영 방지)
        if request.url.path == "/metrics":
            return await call_next(request)

        # 라우트 템플릿 (e.g. /defects/{defect_id}) 을 라벨로 — cardinality 폭증 방지
        route = request.scope.get("route")
        path_label = route.path if route is not None else request.url.path

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        http_requests_total.labels(
            method=request.method,
            path=path_label,
            status=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            path=path_label,
        ).observe(duration)

        return response


# ── 스냅샷 동기화 (sensor Gauge 갱신) ─────────────────
def refresh_sensor_gauges():
    """/metrics 호출 시마다 최신 센서값을 Gauge 에 반영."""
    # 순환 import 방지 위해 함수 내부 import
    from app.core.stream_inference import stream_inference_worker
    from app.services.lidar import lidar_service
    from app.services.telemetry_cache import telemetry_cache

    dist = lidar_service.latest_distance_m
    lidar_distance_meters.set(dist if dist is not None else -1)

    age = telemetry_cache.age_sec
    telemetry_cache_age_seconds.set(age if age is not None else -1)

    stats = stream_inference_worker.stats
    stream_worker_queue_size.set(stats.get("queue_size", 0))


# ── /metrics 엔드포인트 응답 생성 ────────────────────
def render_metrics() -> Response:
    """Prometheus 스크래퍼용 텍스트 응답."""
    refresh_sensor_gauges()
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
