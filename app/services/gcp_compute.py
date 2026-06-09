# =============================================
# app/services/gcp_compute.py
# 역할: GCP Compute Engine 인스턴스 원격 제어 (start/stop/status)
#       - 서비스 계정 JSON 키로 RS256 JWT 발급 → OAuth2 토큰 교환
#       - 발급한 access_token 으로 Compute Engine REST API 호출
#       - 토큰은 만료 30초 전까지 메모리 캐시
# 사용처: app/api/admin_gpu.py (관리자 GPU 제어 엔드포인트)
# =============================================

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from jose import jwt as jose_jwt

from app.config import settings

GCP_TOKEN_URL = "https://oauth2.googleapis.com/token"
GCP_COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
SCOPE = "https://www.googleapis.com/auth/compute"


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float  # epoch seconds


class GcpComputeError(RuntimeError):
    """GCP Compute 호출 실패 — 라우터에서 5xx 로 변환."""


class GcpComputeClient:
    """GCP Compute Engine 인스턴스 제어 클라이언트.

    싱글톤으로 사용 (모듈 하단 `gcp_compute` 인스턴스).
    토큰은 만료 30초 전까지 캐시하여 매 요청마다 재발급하지 않는다.
    """

    def __init__(self) -> None:
        self._token_cache: Optional[_CachedToken] = None
        self._token_lock = asyncio.Lock()
        self._sa_cache: Optional[dict] = None
        # 호출마다 새 TLS 커넥션을 맺지 않도록 httpx 클라이언트를 재사용 (커넥션 풀 유지).
        self._http: Optional[httpx.AsyncClient] = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=20.0)
        return self._http

    async def aclose(self) -> None:
        """앱 종료 시 호출 — 커넥션 정리."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    # ── 서비스 계정 키 파싱 ──────────────────
    def _load_service_account(self) -> dict:
        if self._sa_cache is not None:
            return self._sa_cache

        raw = settings.GCP_SERVICE_ACCOUNT_JSON.strip()
        if not raw:
            raise GcpComputeError(
                "GCP_SERVICE_ACCOUNT_JSON 환경변수가 비어있습니다. Fly.io secrets 에 등록 필요."
            )

        # base64 인코딩이면 디코드, 아니면 JSON 원문으로 간주
        if not raw.startswith("{"):
            try:
                raw = base64.b64decode(raw).decode("utf-8")
            except Exception as e:
                raise GcpComputeError(f"GCP_SERVICE_ACCOUNT_JSON base64 디코딩 실패: {e}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise GcpComputeError(f"GCP_SERVICE_ACCOUNT_JSON 파싱 실패: {e}")

        for key in ("client_email", "private_key"):
            if key not in data:
                raise GcpComputeError(f"서비스 계정 JSON 에 '{key}' 누락")

        self._sa_cache = data
        return data

    # ── 토큰 발급 (JWT → access_token) ────────
    async def _get_access_token(self) -> str:
        now = time.time()
        cached = self._token_cache
        if cached and cached.expires_at > now + 30:
            return cached.access_token

        async with self._token_lock:
            cached = self._token_cache
            if cached and cached.expires_at > now + 30:
                return cached.access_token

            sa = self._load_service_account()
            iat = int(time.time())
            exp = iat + 3600
            assertion = jose_jwt.encode(
                {
                    "iss": sa["client_email"],
                    "scope": SCOPE,
                    "aud": GCP_TOKEN_URL,
                    "iat": iat,
                    "exp": exp,
                },
                sa["private_key"],
                algorithm="RS256",
            )

            resp = await self._get_http().post(
                GCP_TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
                timeout=15.0,
            )
            if resp.status_code != 200:
                raise GcpComputeError(
                    f"GCP 토큰 발급 실패 ({resp.status_code}): {resp.text}"
                )
            payload = resp.json()
            token = payload["access_token"]
            ttl = int(payload.get("expires_in", 3600))
            self._token_cache = _CachedToken(access_token=token, expires_at=time.time() + ttl)
            return token

    # ── 인스턴스 제어 ────────────────────────
    def _instance_url(self) -> str:
        if not settings.GCP_PROJECT_ID:
            raise GcpComputeError("GCP_PROJECT_ID 환경변수가 비어있습니다.")
        return (
            f"{GCP_COMPUTE_BASE}/projects/{settings.GCP_PROJECT_ID}"
            f"/zones/{settings.GCP_GPU_ZONE}/instances/{settings.GCP_GPU_INSTANCE}"
        )

    async def _call(self, method: str, path_suffix: str = "") -> dict:
        token = await self._get_access_token()
        url = self._instance_url() + path_suffix
        resp = await self._get_http().request(
            method,
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code >= 400:
            raise GcpComputeError(
                f"GCP API 실패 ({resp.status_code}): {resp.text}"
            )
        if not resp.content:
            return {}
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw": resp.text}

    async def get_status(self) -> dict:
        """인스턴스 상태 조회 — RUNNING / TERMINATED / STOPPING / PROVISIONING 등."""
        data = await self._call("GET")
        return {
            "name": data.get("name"),
            "status": data.get("status"),
            "zone": (data.get("zone") or "").split("/")[-1] or settings.GCP_GPU_ZONE,
            "machine_type": (data.get("machineType") or "").split("/")[-1],
            "last_start_at": data.get("lastStartTimestamp"),
            "last_stop_at": data.get("lastStopTimestamp"),
        }

    async def start(self) -> dict:
        """인스턴스 시작. 시간당 과금 시작."""
        op = await self._call("POST", "/start")
        return {"operation": op.get("name"), "status": op.get("status")}

    async def stop(self) -> dict:
        """인스턴스 정지. GPU 시간당 과금 중단 (디스크/IP 만 ~$13/월 유지)."""
        op = await self._call("POST", "/stop")
        return {"operation": op.get("name"), "status": op.get("status")}


gcp_compute = GcpComputeClient()
