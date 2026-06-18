# =============================================
# app/core/inference_proxy.py
# 역할: Fly(항상 켜짐·모델 없음) → GCP GPU VM(모델·검출) 추론 프록시 미들웨어.
#   - settings.INFERENCE_PROXY_URL 설정 시, `/api/v1/stream/test/*` 요청을 GPU VM 으로 전달
#     → 운영 사이트(aeroinspect.site→Fly)에서도 검출이 동작.
#   - GPU VM 이 꺼져 있으면 503 + 안내(관리자 페이지에서 GPU 시작 유도).
#   - 어떤 오류든 로컬 처리로 fallthrough — 프록시가 프로덕션을 절대 막지 않는다(fail-safe).
#   - INFERENCE_PROXY_URL 미설정이면 완전 무동작(기존 동작 그대로, 무회귀).
#
# 검출 결과 WS 다리(구현됨 — 아래 _ws_relay_loop):
#   GPU VM 의 ws_manager 가 검출 이벤트를 broadcast 하고 프론트는 Fly 의 WS 를 보므로,
#   Fly 가 GPU 의 /ws?channels=defects 에 붙어(=공개 채널, 토큰 불요) 수신 이벤트를 Fly
#   ws_manager 로 재broadcast 한다. defect.new 뿐 아니라 thermal.screening 등 defects 채널의
#   모든 이벤트를 중계한다(start_ws_relay 는 INFERENCE_PROXY_URL 설정 시에만 기동).
#
# 주의(운영 전제):
#   - 인증 정합: 프록시되는 제어 엔드포인트(/test/start 등)는 GPU VM 에서 토큰을 검증하므로
#     Fly·GCP 의 JWT_SECRET 이 동일해야 한다(불일치 시 401). MJPEG/active/upload-file 은 public.
#   - 멀티머신: Fly 가 여러 머신으로 뜨면 릴레이 머신과 프론트 WS 머신이 달라질 수 있으므로
#     WS_BACKEND=redis(공유) 권장. 단일 머신이면 memory 백엔드로도 동작.
#   - 업로드(대용량)는 클라이언트 수신 스트림을 그대로 GPU VM 으로 흘려보냄(버퍼링 X) → RAM 상수.
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
#   known: 한 번이라도 실제 조회에 성공했는지. 실패를 '꺼짐'으로 단정하지 않기 위함.
#   ttl:   정상 조회는 10초, 조회 실패 시 3초로 줄여 빠르게 회복 시도.
_gpu_cache = {"running": False, "at": 0.0, "known": False, "ttl": 10.0}
_GPU_TTL_SEC = 10.0
_GPU_ERROR_TTL_SEC = 3.0

# 전달하지 않을 hop-by-hop 헤더
_HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "proxy-authenticate", "proxy-authorization", "te", "trailers",
}

# upstream(GPU VM) 응답에서 제거할 CORS 헤더 — CORS 는 Fly 의 CORSMiddleware(가장 바깥)가
# 단독으로 설정한다. upstream 도 보내면 Access-Control-Allow-Origin 이 중복되어 브라우저가 차단.
_STRIP_RESP_HEADERS = _HOP_HEADERS | {
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-expose-headers",
    "access-control-max-age",
    "vary",
}


async def _gpu_running() -> bool:
    """GPU VM 이 RUNNING 인지(캐시).

    조회 실패(GCP 일시 장애·쿼터 등)를 곧장 '꺼짐'으로 단정하지 않는다 —
    직전에 알려진 상태가 있으면 그 값을 유지하고 짧은 TTL 로 빠르게 재시도한다.
    (실패→False 로 덮으면 멀쩡히 켜진 GPU 가 꺼진 것처럼 보여 불필요한 재시작·비용 유발.)
    """
    now = time.monotonic()
    if now - _gpu_cache["at"] < _gpu_cache["ttl"]:
        return _gpu_cache["running"]
    try:
        from app.services.gcp_compute import gcp_compute
        data = await gcp_compute.get_status()
        running = str(data.get("status", "")).upper() == "RUNNING"
        _gpu_cache.update(running=running, at=now, known=True, ttl=_GPU_TTL_SEC)
        return running
    except Exception as e:
        print(f"[InferenceProxy] GPU 상태 조회 실패 — 직전 상태 유지(짧은 재시도): {e}")
        # 직전 성공 상태가 있으면 유지, 한 번도 성공한 적 없으면 보수적으로 False.
        _gpu_cache.update(at=now, ttl=_GPU_ERROR_TTL_SEC)
        return _gpu_cache["running"] if _gpu_cache["known"] else False


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

    # ── 계측(진단): 지연이 'Fly 홉(중계)' 때문인지 'GPU 처리' 때문인지 데이터로 분리 ──
    #   t0      : 프록시 진입
    #   t_sent  : client.send 반환 = 업로드 본문(Fly→GPU) 전송 완료 + GPU 첫 응답 헤더 도착
    #   t_done  : 응답 본문 전부 프론트로 중계 완료(_body_iter 종료)
    # 업로드 throughput = up_bytes / (t_sent - t0) → Fly→GPU 회선 속도 가늠.
    # 짧은 응답(upload-file API 등)은 (t_sent-t0)가 사실상 업로드 전송 시간.
    # MJPEG/영상 직재생 등 장기 스트림은 t_done 이 매우 커도 정상(연결 유지 시간).
    t0 = time.monotonic()
    up_bytes = 0

    # 대용량 업로드(영상)는 전체 버퍼링하면 Fly 1GB RAM 을 압박하고 전송 지연으로 연결이 끊긴다.
    # 클라이언트 수신 스트림을 그대로 GPU VM 으로 흘려보내 메모리 상수·끊김 없이 중계한다.
    # (content-length 는 hop 헤더라 제거됨 → httpx 가 chunked transfer-encoding 으로 전송)
    async def _counting_stream():
        nonlocal up_bytes
        async for chunk in request.stream():
            up_bytes += len(chunk)
            yield chunk

    # read=None: MJPEG 장기 스트림 무한 대기 허용. write=None: 대용량 업로드 전송 시간 무제한.
    # connect 만 짧게(10s) — GPU VM 미응답 시 빠르게 fallthrough.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=None, write=None, pool=None)
    )
    req = client.build_request(
        request.method, url,
        params=dict(request.query_params),
        headers=fwd_headers,
        content=_counting_stream(),
    )
    upstream = await client.send(req, stream=True)
    t_sent = time.monotonic()
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _STRIP_RESP_HEADERS}

    # Server-Timing: 브라우저 DevTools Network 탭에서 Fly 구간을 직접 확인.
    #   fwd = Fly→GPU 업로드 전송 + GPU 첫 응답까지(= client.send 소요).
    #   (브라우저가 잰 전체 시간) - fwd ≈ 브라우저↔Fly 회선 구간.
    # 참고: JS(PerformanceObserver)로 읽으려면 CORS expose-headers 필요. DevTools 표시는 무관.
    fwd_ms = (t_sent - t0) * 1000.0
    resp_headers["Server-Timing"] = f"fwd;dur={fwd_ms:.0f}"

    async def _body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            total_ms = (time.monotonic() - t0) * 1000.0
            if up_bytes > 0:
                mb = up_bytes / (1024 * 1024)
                thru = mb / max(t_sent - t0, 1e-6)
                print(
                    f"[InferenceProxy] {request.method} {path} "
                    f"up={mb:.1f}MB fwd={fwd_ms:.0f}ms total={total_ms:.0f}ms thru={thru:.1f}MB/s"
                )
            else:
                print(
                    f"[InferenceProxy] {request.method} {path} "
                    f"fwd={fwd_ms:.0f}ms total={total_ms:.0f}ms"
                )

    return StreamingResponse(
        _body_iter(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ── WS 릴레이: GPU VM 의 검출 결과를 Fly WS 클라이언트(프론트)로 중계 ────────────
# 검출 이벤트(defect.new, thermal.screening 등)는 GPU VM 의 ws_manager 가 broadcast 한다.
# 프론트는 Fly 의 WS 를 보므로, 운영에서 검출 카드/오버레이를 받으려면 GPU→Fly WS 다리가 필요.
# defects 채널은 공개(토큰 불필요)라 Fly 가 GPU 의 /ws?channels=defects 에 붙어 그 채널의
# 모든 이벤트를 받아 Fly ws_manager 로 재broadcast 한다.
_relay_task: "asyncio.Task | None" = None


# 재연결 백오프: 연속 실패 시 5s→최대 60s 로 증가, 연결 성공하면 5s 로 리셋.
# (고정 5s 재시도는 GPU/네트워크 장기 장애 시 로그·소켓 시도를 과도하게 스팸)
_RELAY_BACKOFF_MIN = 5.0
_RELAY_BACKOFF_MAX = 60.0
# broadcast 가 멈춰도 릴레이 read 루프가 무한 블로킹되지 않도록 상한.
_RELAY_BROADCAST_TIMEOUT = 5.0


async def _ws_relay_loop() -> None:
    """GPU VM defects WS → Fly ws_manager 중계. 설정/ GPU 상태 따라 동작, 끊기면 백오프 재시도."""
    import websockets
    from app.core.ws_manager import ws_manager

    backoff = _RELAY_BACKOFF_MIN
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
                backoff = _RELAY_BACKOFF_MIN  # 연결 성공 → 백오프 리셋
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        # GPU 가 defects 채널로 쏘는 '모든' 이벤트를 그대로 중계한다.
                        # 과거엔 defect.new 만 중계 → thermal.screening(열화상 단열 스크리닝
                        # 오버레이 + 프론트 분석게이트 프런티어)이 프로덕션에서 누락돼, 열화상
                        # 영상이 운영 사이트에서 오버레이/재생게이트가 약해지는 사고가 있었다.
                        # 이 릴레이는 GPU 의 defects 채널만 구독하고(=루프 없음), 그 채널 이벤트는
                        # 전부 Fly 프론트로 가야 하므로 type 보유 dict 은 모두 재broadcast 한다
                        # (향후 추가 이벤트 타입도 자동 포함). defect 리뷰/삭제 등은 Fly 가
                        # 직접 처리·broadcast 하므로 GPU defects 채널엔 안 흘러와 중복 없음.
                        if isinstance(data, dict) and data.get("type"):
                            # broadcast 가 멈춰도 read 루프 전체가 막히지 않도록 타임아웃.
                            await asyncio.wait_for(
                                ws_manager.broadcast("defects", data),
                                timeout=_RELAY_BROADCAST_TIMEOUT,
                            )
                    except asyncio.TimeoutError:
                        print("[InferenceRelay] broadcast 타임아웃 — 해당 메시지 건너뜀")
                    except Exception:
                        pass  # 개별 메시지 파싱/전파 실패는 무시
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[InferenceRelay] 연결 실패/끊김 — {backoff:.0f}s 후 재시도: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RELAY_BACKOFF_MAX)


def start_ws_relay() -> None:
    """lifespan startup 에서 호출 — INFERENCE_PROXY_URL 설정 시에만 릴레이 기동."""
    global _relay_task
    if not settings.INFERENCE_PROXY_URL:
        return
    if _relay_task is None or _relay_task.done():
        _relay_task = asyncio.create_task(_ws_relay_loop())
        print("[InferenceRelay] WS 릴레이 태스크 기동")
