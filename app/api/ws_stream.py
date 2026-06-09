# =============================================
# app/api/ws_stream.py
# 역할: 드론 실시간 프레임 수신용 WebSocket 엔드포인트
#       - WS /ws/stream — 클라이언트가 바이너리 JPEG 프레임을 송신
#       - 수신 즉시 asyncio.to_thread(cv2.imdecode, ...)로 디코딩 (블로킹 방지)
#       - stream_inference_worker.submit(frame) → 드롭 큐
#       - 클라이언트 → 서버 텍스트 제어: {"type":"ping"} → {"type":"pong"}
#       - 서버 → 클라이언트: 추론 결과는 stream_inference_worker에서 broadcast
#
# 이 엔드포인트는 구독 전용 /ws?channel=stream 과는 별도 — 바이너리 프레임 수신이 목적.
# 추론 결과는 ws_manager.broadcast("stream", ...)로 모든 /ws/stream 및
# /ws?channel=stream 구독자에게 동시 전송됨.
# =============================================

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.stream_inference import stream_inference_worker
from app.core.ws_manager import ws_manager

router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    드론 영상 프레임 수신 WebSocket.

    프로토콜:
      클라이언트 → 서버:
        - bytes: JPEG 인코딩된 프레임 (drone_cam.jpg raw)
        - text:  {"type":"ping"}  (하트비트 유지)
      서버 → 클라이언트:
        - text:  {"type":"pong"}  (ping 응답)
        - text:  {"type":"detection", "timestamp":.., "frame_id":.., "result":{DetectionResult}}
                 (추론 워커가 브로드캐스트)
    """
    await ws_manager.connect(websocket, "stream")
    await ws_manager.send_personal(websocket, {
        "type": "connection.established",
        "data": {"channel": "stream", "message": "스트림 연결 완료. 바이너리 JPEG 프레임을 송신하세요."},
    })

    try:
        while True:
            msg = await websocket.receive()

            # WebSocketDisconnect 감지 (starlette는 msg.type으로 구분)
            if msg.get("type") == "websocket.disconnect":
                break

            # 텍스트 제어 메시지 (ping/pong 등)
            text = msg.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await ws_manager.send_personal(websocket, {"type": "error", "message": "JSON 파싱 실패"})
                    continue

                if payload.get("type") == "ping":
                    await ws_manager.send_personal(websocket, {"type": "pong"})
                # 기타 제어 메시지는 현재 미사용
                continue

            # 바이너리 프레임
            data = msg.get("bytes")
            if data is None or len(data) == 0:
                continue

            # FRAME_SKIP 게이트를 디코드 *이전*에 적용 — 버릴 프레임은 디코드하지 않는다.
            # (skip=3 기준 JPEG 디코드 CPU 약 1/3로 절감)
            if not stream_inference_worker.will_enqueue():
                continue

            # JPEG 디코딩도 블로킹이라 스레드로
            frame = await asyncio.to_thread(_decode_jpeg, data)
            if frame is None:
                await ws_manager.send_personal(websocket, {
                    "type": "error",
                    "message": "JPEG 디코딩 실패",
                })
                continue

            # 드롭 큐에 submit (will_enqueue 가 통과한 프레임만)
            stream_inference_worker.submit(frame)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws_stream] 예외: {e}")
    finally:
        ws_manager.disconnect(websocket, "stream")


def _decode_jpeg(data: bytes):
    """bytes → BGR ndarray. 실패 시 None."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
