# =============================================
# app/core/inference_proxy.py
# 역할: Fly(항상 켜짐·모델 없음) → GCP GPU VM(모델·검출) 추론 프록시 미들웨어.
#   - settings.INFERENCE_PROXY_URL 설정 시, `/api/v1/stream/test/*` 요청을 GPU VM 으로 전달
#     → 운영 사이트(aeroinspect.site→Fly)에서도 검출이 동작.
#   - GPU VM 이 꺼져 있으면 503 + 안내(관리자 페이지에서 GPU 시작 유도).
#   - 어떤 오류든 로컬 처리로 fallthrough — 프록시가 프로덕션을 절대 막지 않는다(fail-safe).
#   - INFERENCE_PROXY_URL 미설정이면 완전 무동작(기존 동작 그대로, 무회귀).
#
# 주의(미완성/후속):
#   - 검출 결과 WS(defect.new): GPU VM 의 ws_manager 가 broadcast 하므로, Fly WS 클라이언트
#     (프론트)에 닿게 하려면 (a) 공유 Redis(WS_BACKEND=redis) 또는 (b) Fly→GCP WS 릴레이 필요.
#     이 모듈은 HTTP 경로만 프록시한다. WS 다리는 활성화 런북 참고.
#   - 인증 정합: 프록시되는 제어 엔드포인트(/test/start 등)는 GPU VM 에서 토큰을 검증하므로
#     Fly·GCP 의 JWT_SECRET 이 동일해야 한다(런북 참고). MJPEG/active/upload-file 은 public.
#   - 업로드(대용량)는 현재 전체 버퍼링 — 매우 큰 파일 동시 업로드 시 RAM 주의(후속: 스트리밍).
# =============================================

from __future__ import annotations

import asyncio
import json
import time

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from app.config import settings

# 프록시 대상 경로 prefix (테스트모드 검출 서브시스템)
_PROXY_PREFIX = "/api/v1/stream/test"

# GPU 상태 캐시 — 매 요청마다 GCP Compute API 호출(느림·쿼터) 방지
_gpu_cache = {"running": False, "at": 0.0}
_GPU_TTL_SEC = 10.0

# 전달하지 않을 hop-by-hop 헤더
_HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "proxy-authenticate", "proxy-authorization", "te", "trailers",
}


async def _gpu_running() -> bool:
    """GPU VM 이 RUNNING 인지(캐시 10초). 실패 시 False(=꺼짐으로 간주, 503 안내)."""
    now = time.monotonic()
    if now - _gpu_cache["at"] < _GPU_TTL_SEC:
        return _gpu_cache["running"]
    running = False
    try:
        from app.services.gcp_compute import gcp_compute
        data = await gcp_compute.get_status()
        running = str(data.get("status", "")).upper() == "RUNNING"
    except Exception:
        running = False
    _gpu_cache["running"] = running
    _gpu_cache["at"] = now
    return running


class InferenceProxyMiddleware(BaseHTTPMiddleware):
    """`/api/v1/stream/test/*` 를 GPU VM 으로 프록시(설정 시). fail-safe."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        target = settings.INFERENCE_PROXY_URL
        # 미설정 또는 비대상 경로 → 즉시 통과(무영향)
        if not target or not path.startswith(_PROXY_PREFIX):
            return await call_next(request)
        try:
            if not await _gpu_running():
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "추론 서버(GPU)가 꺼져 있습니다. 관리자 페이지에서 GPU를 시작하세요.",
                        "gpu_required": True,
                    },
                )
            return await _proxy_request(request, target, path)
        except Exception as e:
            # 프록시 경로에서 어떤 오류가 나도 로컬 처리로 진행 — 프로덕션을 절대 막지 않는다.
            print(f"[InferenceProxy] 프록시 실패 — 로컬 fallthrough: {e}")
            return await call_next(request)


async def _proxy_request(request: Request, target: str, path: str) -> Response:
    """요청을 GPU VM 으로 전달하고 응답을 스트리밍으로 되돌린다(MJPEG 등 장기 스트림 대응)."""
    url = target.rstrip("/") + path
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    body = await request.body()  # 후속: 대용량 업로드 스트리밍

    # read=None: MJPEG 장기 스트림 무한 대기 허용. connect 만 짧게.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=None, write=None, pool=None)
    )
    req = client.build_request(
        request.method, url,
        params=dict(request.query_params),
        headers=fwd_headers,
        content=body,
    )
    upstream = await client.send(req, stream=True)
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS}

    async def _body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _body_iter(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ── WS 릴레이: GPU VM 의 검출 결과를 Fly WS 클라이언트(프론트)로 중계 ────────────
# 검출(defect.new)은 GPU VM 의 ws_manager 가 broadcast 한다. 프론트는 Fly 의 WS 를 보므로,
# 운영에서 검출 카드를 받으려면 GPU→Fly WS 다리가 필요. defects 채널은 공개(토큰 불필요)라
# Fly 가 GPU 의 /ws?channels=defects 에 붙어 defect.new 를 받아 Fly ws_manager 로 재broadcast 한다.
_relay_task: "asyncio.Task | None" = None


async def _ws_relay_loop() -> None:
    """GPU VM defects WS → Fly ws_manager 중계. 설정/ GPU 상태 따라 동작, 끊기면 재시도."""
    import websockets
    from app.core.ws_manager import ws_manager

    while True:
        try:
            target = settings.INFERENCE_PROXY_URL
            if not target:
                await asyncio.sleep(30)
                continue
            if not await _gpu_running():
                await asyncio.sleep(15)
                continue
            ws_url = (
                target.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
                + "/api/v1/ws?channels=defects"
            )
            async with websockets.connect(ws_url, ping_interval=20, open_timeout=15) as ws:
                print(f"[InferenceRelay] GPU defects WS 연결됨: {ws_url}")
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("type") == "defect.new":
                            await ws_manager.broadcast("defects", data)
                    except Exception:
                        pass  # 개별 메시지 파싱 실패는 무시
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[InferenceRelay] 연결 실패/끊김 — 5s 후 재시도: {e}")
            await asyncio.sleep(5)


def start_ws_relay() -> None:
    """lifespan startup 에서 호출 — INFERENCE_PROXY_URL 설정 시에만 릴레이 기동."""
    global _relay_task
    if not settings.INFERENCE_PROXY_URL:
        return
    if _relay_task is None or _relay_task.done():
        _relay_task = asyncio.create_task(_ws_relay_loop())
        print("[InferenceRelay] WS 릴레이 태스크 기동")
