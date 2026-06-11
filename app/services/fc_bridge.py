# =============================================
# app/services/fc_bridge.py
# 역할: Pi Zero 2 W ↔ 백엔드 양방향 WebSocket 게이트웨이
#       - 백엔드 명령(MSP V2) → Pi → FC UART
#       - FC 텔레메트리 → Pi → 백엔드 (telemetry 채널 브로드캐스트)
#       - Pi heartbeat 200ms 추적, 누락 시 safety_monitor 통지
#       - 본 모듈은 명령 직렬화/큐잉만 담당. 실제 UART는 Pi 측 fc_bridge.py
# =============================================
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── MSP V2 명령 타입 (mission_orchestrator → fc_bridge) ─────
class MspCommand:
    """
    mission_orchestrator 가 발행하는 추상 명령. fc_bridge 가 MSP V2 패킷으로 직렬화.
    Pi 측 fc_bridge.py 가 UART 로 바이트 송신.
    """
    SET_RAW_RC = "set_raw_rc"          # payload: {ch1..ch16: int 1000..2000} 또는 {vx,vy,vz,yaw_rate}
    LOAD_MISSION_WP = "load_mission_wp"  # payload: {waypoints: [{lat,lon,alt,...}]}
    SET_NAV_MODE = "set_nav_mode"        # payload: {mode: "POSHOLD" | "WP" | "RTH" | "LAND"}
    SET_ATTITUDE = "set_attitude"        # payload: {roll_rad, pitch_rad, yaw_rad, thrust_norm}
    ARM = "arm"
    DISARM = "disarm"
    LAND = "land"
    RTH = "rth"


@dataclass
class TelemetryFrame:
    """Pi 가 MSP 응답을 디코딩해 보낸 텔레메트리 1건."""
    ts: float
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    pos_z: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    vel_x: Optional[float] = None
    vel_y: Optional[float] = None
    vel_z: Optional[float] = None
    battery_pct: Optional[float] = None
    voltage: Optional[float] = None
    flight_mode: Optional[str] = None
    is_armed: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


TelemetryCallback = Callable[[TelemetryFrame], Awaitable[None]]


class FcBridge:
    """
    Pi 가 클라이언트로 접속(reverse-WS). 단일 Pi 가정(드론 1대) — 다중 드론은 v1.2.
    """
    HEARTBEAT_TIMEOUT_SEC = 2.0

    def __init__(self) -> None:
        self._pi_ws: Optional[WebSocket] = None
        self._command_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=64)
        self._telemetry_callbacks: list[TelemetryCallback] = []
        self._last_heartbeat_ts: float = 0.0
        self._send_task: Optional[asyncio.Task] = None
        self._heartbeat_watch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── WS 연결 라이프사이클 ───────────────────
    async def attach(self, ws: WebSocket) -> None:
        """Pi 클라이언트 접속 시 호출 (FastAPI WS 엔드포인트 핸들러에서 위임)."""
        async with self._lock:
            if self._pi_ws is not None:
                logger.warning("fc_bridge.duplicate_pi", action="reject")
                await ws.close(code=4409, reason="another pi already attached")
                return
            self._pi_ws = ws
            self._last_heartbeat_ts = time.time()

        self._send_task = asyncio.create_task(self._sender_loop())
        self._heartbeat_watch_task = asyncio.create_task(self._heartbeat_watcher())
        logger.info("fc_bridge.pi_attached")

        try:
            await self._receiver_loop()
        except WebSocketDisconnect:
            logger.warning("fc_bridge.pi_disconnected")
        finally:
            await self._detach()

    async def _detach(self) -> None:
        async with self._lock:
            self._pi_ws = None
        if self._send_task:
            self._send_task.cancel()
            self._send_task = None
        if self._heartbeat_watch_task:
            self._heartbeat_watch_task.cancel()
            self._heartbeat_watch_task = None

    # ── 명령 송신 ─────────────────────────────
    async def send(self, command: str, payload: Optional[Dict[str, Any]] = None) -> None:
        msg = {"type": "command", "name": command, "payload": payload or {}, "ts": time.time()}
        try:
            self._command_queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.error("fc_bridge.queue_full", dropped=command)

    # ── 텔레메트리 구독 ───────────────────────
    def subscribe_telemetry(self, cb: TelemetryCallback) -> None:
        self._telemetry_callbacks.append(cb)

    # ── 내부 루프 ─────────────────────────────
    async def _sender_loop(self) -> None:
        while True:
            msg = await self._command_queue.get()
            ws = self._pi_ws
            if ws is None:
                logger.warning("fc_bridge.send_no_pi", dropped=msg.get("name"))
                continue
            try:
                await ws.send_json(msg)
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.error("fc_bridge.send_failed", error=str(e))

    async def _receiver_loop(self) -> None:
        ws = self._pi_ws
        assert ws is not None
        while True:
            data = await ws.receive_json()
            kind = data.get("type")
            if kind == "heartbeat":
                self._last_heartbeat_ts = time.time()
            elif kind == "telemetry":
                frame = TelemetryFrame(
                    ts=data.get("ts", time.time()),
                    pos_x=data.get("pos_x"), pos_y=data.get("pos_y"), pos_z=data.get("pos_z"),
                    roll=data.get("roll"), pitch=data.get("pitch"), yaw=data.get("yaw"),
                    vel_x=data.get("vel_x"), vel_y=data.get("vel_y"), vel_z=data.get("vel_z"),
                    battery_pct=data.get("battery_pct"), voltage=data.get("voltage"),
                    flight_mode=data.get("flight_mode"), is_armed=bool(data.get("is_armed", False)),
                    extra=data.get("extra", {}),
                )
                for cb in list(self._telemetry_callbacks):
                    try:
                        await cb(frame)
                    except Exception as e:
                        logger.error("fc_bridge.telemetry_cb_failed", error=str(e))
            else:
                logger.debug("fc_bridge.unknown_msg", kind=kind)

    async def _heartbeat_watcher(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            gap = time.time() - self._last_heartbeat_ts
            if gap > self.HEARTBEAT_TIMEOUT_SEC and self._pi_ws is not None:
                logger.error("fc_bridge.heartbeat_lost", gap_sec=gap)
                # safety_monitor 가 다음 check() 에서 comm_loss 판단하도록
                # last_pi_heartbeat_ts 는 mission_orchestrator 의 TelemetrySnapshot 빌더가 이 값을 사용
                # ESTOP은 직접 트리거하지 않음 — safety_monitor 에 의사결정 일원화

    @property
    def last_heartbeat_ts(self) -> float:
        return self._last_heartbeat_ts

    @property
    def is_attached(self) -> bool:
        return self._pi_ws is not None


# ── 싱글톤 인스턴스 ──────────────────────────────
fc_bridge = FcBridge()
