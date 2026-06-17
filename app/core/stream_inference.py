# =============================================
# app/core/stream_inference.py
# 역할: WebSocket 실시간 스트림 추론 워커 (드롭 큐 + 프레임 스킵)
#       - asyncio.Queue(maxsize=1) — 최신 프레임 1개만 유지
#       - put_nowait 실패 시 그냥 드롭 (추론 워커가 바쁠 때)
#       - FRAME_SKIP: N프레임 중 1프레임만 추론
#       - 추론은 asyncio.to_thread(pipeline.detect) — 이벤트 루프 블로킹 방지
#       - ByteTrack 객체 추적 → Temporal Filter 오탐 제거 → 브로드캐스트
#       - Hard Example Mining: 불확실 프레임 자동 수집 (Active Learning)
#       - 결과는 "stream" + "defects" 두 채널에 broadcast
#       - 별도 태스크로 영구 실행 (main.py lifespan에서 start/stop)
#
# 드론 IRC-256CA 스트림(15~30fps) + T4 GPU 추론(80~150ms/frame) 환경에서
# 모든 프레임을 처리 못 하므로 드롭 큐 패턴이 필수.
# =============================================

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.core.ws_manager import ws_manager
from app.services.active_learning import hard_example_miner
from app.services.defect_persistence import defect_persistence
from app.services.defect_taxonomy import map_to_legacy, xyxy_to_xywhn
from app.services.inference_pipeline import pipeline
from app.services.lidar import lidar_service
from app.services.object_tracker import defect_tracker
from app.services.telemetry_cache import telemetry_cache
from app.services.temporal_filter import TemporalFilter
from app.services.hybrid_detector import detect_hybrid_async
from app.services.vlm_detector import VLMQuotaExceeded, detect_vlm_async


@dataclass
class QueuedFrame:
    """큐에 들어갈 단일 프레임."""
    frame_bgr: np.ndarray
    frame_id: int
    submitted_at: float  # epoch seconds
    thermal_map: Optional[np.ndarray] = None   # 20종 파이프라인: 열화상 온도맵 float32 °C
    imu_data: Optional[dict] = None            # 20종 파이프라인: 드론 IMU {roll, pitch, yaw}
    thermal_frame_bgr: Optional[np.ndarray] = None  # Thermal Anomaly: 열화상 의사컬러 BGR


class StreamInferenceWorker:
    """
    WebSocket 프레임 → 추론 → 추적 → 필터 → 브로드캐스트 파이프라인.
    싱글톤. main.py lifespan에서 start()/stop() 호출.
    """

    def __init__(self):
        # 최신 프레임 1개만 유지 (maxsize=1)
        self._queue: asyncio.Queue[QueuedFrame] = asyncio.Queue(maxsize=1)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._submitted_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        # FRAME_SKIP은 submit 단계에서 적용 (워커는 큐에서 꺼낸 건 무조건 처리)
        self._frame_skip: int = 3
        self._error_count: int = 0          # 연속 추론 실패 카운터
        self._total_errors: int = 0         # 총 추론 실패 횟수
        # ── VLM 키프레임 오버레이 (근실시간 비전 LLM 검출) ──
        # 30fps ONNX 경로와 별개로, N초마다 최신 프레임 1장만 VLM에 비동기 제출.
        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_frame_id: int = 0  # _last_frame_bgr 에 대응하는 frame_id (VLM 이벤트 정합)
        self._vlm_task: Optional[asyncio.Task] = None
        self._vlm_count: int = 0            # VLM 키프레임 처리 횟수
        self._vlm_errors: int = 0           # VLM 실패 횟수
        # 시간 일관성 필터 (IoU 기반 공간 매칭 + Noisy-OR 누적)
        self._temporal_filter = TemporalFilter(
            window_size=settings.TEMPORAL_FILTER_WINDOW,
            min_detections=settings.TEMPORAL_FILTER_MIN_DETECTIONS,
            instant_threshold=settings.TEMPORAL_INSTANT_THRESHOLD,
            iou_threshold=settings.TEMPORAL_FILTER_IOU,
        )
        # ByteTrack 추적기 설정 반영 (드론 환경: 동적 frame_rate 계산)
        defect_tracker.min_hits = settings.TRACKER_MIN_HITS
        defect_tracker.max_age = settings.TRACKER_MAX_AGE
        defect_tracker.iou_threshold = settings.TRACKER_IOU_THRESHOLD
        defect_tracker.reconfigure(
            camera_fps=settings.RECORDING_FPS,   # 카메라 원본 FPS
            frame_skip=settings.FRAME_SKIP,      # 프레임 스킵 값
        )
        # Hard Example Mining 설정 반영
        hard_example_miner.enabled = settings.HARD_EXAMPLE_ENABLED
        hard_example_miner.output_dir = settings.HARD_EXAMPLE_DIR
        hard_example_miner.low_conf_min = settings.HARD_EXAMPLE_LOW_CONF_MIN
        hard_example_miner.low_conf_max = settings.HARD_EXAMPLE_LOW_CONF_MAX
        hard_example_miner.save_interval = settings.HARD_EXAMPLE_SAVE_INTERVAL

    # ── 상태 조회 ────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._running and self._worker_task is not None and not self._worker_task.done()

    @property
    def stats(self) -> dict:
        return {
            "submitted": self._submitted_count,
            "processed": self._processed_count,
            "dropped": self._dropped_count,
            "queue_size": self._queue.qsize(),
            "frame_skip": self._frame_skip,
            "errors": {
                "consecutive": self._error_count,
                "total": self._total_errors,
            },
            "tracker": {
                "available": defect_tracker.is_available,
                "active_tracks": defect_tracker.active_track_count,
                "confirmed_tracks": defect_tracker.confirmed_track_count,
            },
            "hard_examples": hard_example_miner.stats,
            "db_persistence": defect_persistence.stats,
            "vlm_keyframe": {
                "enabled": settings.VLM_DETECTION_ENABLED,
                "running": self._vlm_task is not None and not self._vlm_task.done(),
                "processed": self._vlm_count,
                "errors": self._vlm_errors,
                "interval_sec": settings.VLM_KEYFRAME_INTERVAL_SEC,
            },
        }

    # ── 생명주기 ─────────────────────────────────
    async def start(self) -> None:
        """워커 태스크 시작 (이벤트 루프 내에서 호출)."""
        if self.is_running:
            return
        self._frame_skip = max(1, int(settings.FRAME_SKIP))
        self._running = True
        # 새 세션 시작 시 추적·필터 상태 초기화
        defect_tracker.reset()
        self._temporal_filter.reset()
        hard_example_miner.reset()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="stream_inference_worker")
        # VLM 키프레임 오버레이 태스크 (활성화 시에만)
        if settings.VLM_DETECTION_ENABLED:
            self._vlm_task = asyncio.create_task(self._vlm_keyframe_loop(), name="vlm_keyframe_loop")
            print(f"[StreamInfer] VLM 키프레임 오버레이 ON "
                  f"(provider={settings.VLM_PROVIDER}, every {settings.VLM_KEYFRAME_INTERVAL_SEC}s)")
        print(f"[StreamInfer] 워커 시작 (frame_skip={self._frame_skip}, tracker={'ON' if defect_tracker.is_available else 'OFF'})")

    async def stop(self) -> None:
        """워커 태스크 종료."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self._worker_task = None
        if self._vlm_task is not None:
            self._vlm_task.cancel()
            try:
                await self._vlm_task
            except (asyncio.CancelledError, Exception):
                pass
            self._vlm_task = None
        # 세션 종료 시 DB 재시도 버퍼 flush
        retried = await defect_persistence.flush_retry_buffer()
        if retried:
            print(f"[StreamInfer] DB 재시도 {retried}건 저장 완료")
        # 세션 종료 시 hard example 잔여분 디스크 저장
        saved = hard_example_miner.flush_to_disk()
        if saved:
            print(f"[StreamInfer] Hard example {saved}건 디스크 저장 완료")
        print("[StreamInfer] 워커 종료")

    # ── 프레임 게이트 (디코드 이전) ─────────────────
    def will_enqueue(self) -> bool:
        """JPEG 디코드 *이전*에 호출 — 이 프레임을 추론 큐에 넣을지 판단.

        프레임 카운터를 증가시키고 FRAME_SKIP 게이트만 적용한다(프레임 데이터 불필요).
        False면 호출자는 디코드 자체를 건너뛰어 CPU 를 절약한다(스킵 프레임의 디코드 비용 제거).
        """
        self._submitted_count += 1
        return self._submitted_count % self._frame_skip == 0

    # ── 프레임 제출 ─────────────────────────────
    def submit(
        self,
        frame_bgr: np.ndarray,
        thermal_map: Optional[np.ndarray] = None,
        imu_data: Optional[dict] = None,
        thermal_frame_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """
        수신자(ws_stream.py)가 will_enqueue()==True 인 프레임만 디코드해 호출. 드롭 큐에 put.

        Args:
            frame_bgr: RGB 프레임
            thermal_map: 열화상 온도맵 float32 °C (M4 U-Net 단열 검출)
            imu_data: 드론 IMU 데이터 (기하학 검출)
            thermal_frame_bgr: 열화상 의사컬러 BGR (Thermal Anomaly PatchCore — Moisture/delam)

        Returns:
            True: 큐에 enqueue됨 (추론 예정)
            False: 스킵됐거나 드롭됨
        """
        # 주의: 프레임 카운트 증가 + FRAME_SKIP 게이트는 will_enqueue()에서 디코드 *이전*에
        # 이미 적용됐다(버릴 프레임을 디코드하지 않기 위함). 여기서는 카운트하지 않는다.

        # VLM 키프레임 태스크가 가져갈 최신 프레임 보관.
        # (enqueue 되는 프레임에서만 갱신 — 키프레임 주기가 초 단위라 충분히 신선)
        if settings.VLM_DETECTION_ENABLED:
            self._last_frame_bgr = frame_bgr
            self._last_frame_id = self._submitted_count  # 이 프레임의 정확한 id 캡처

        item = QueuedFrame(
            frame_bgr=frame_bgr,
            frame_id=self._submitted_count,
            submitted_at=time.time(),
            thermal_map=thermal_map,
            imu_data=imu_data,
            thermal_frame_bgr=thermal_frame_bgr,
        )
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            # 워커가 바쁨 — 그냥 드롭
            self._dropped_count += 1
            return False

    # ── 워커 루프 ───────────────────────────────
    async def _worker_loop(self) -> None:
        """큐에서 하나 꺼내 추론 → 브로드캐스트 반복."""
        while self._running:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._process(item)
                self._error_count = 0  # 성공 시 연속 실패 카운터 리셋
            except Exception as e:
                self._error_count += 1
                self._total_errors += 1
                print(f"[StreamInfer] 추론 오류 #{self._total_errors} (frame_id={item.frame_id}): {e}")
                if self._error_count >= 10:
                    print("[StreamInfer] ⚠ 연속 10회 추론 실패 — 모델 상태 점검 필요")
            finally:
                self._processed_count += 1

    async def _process(self, item: QueuedFrame) -> None:
        """단일 프레임 추론 + ByteTrack 추적 + 시간 필터 + 양방향 브로드캐스트."""
        # 20종 파이프라인 활성화 시 분기
        if settings.USE_20DEFECT_PIPELINE:
            await self._process_20(item)
            return

        if not pipeline.is_loaded:
            return

        # 블로킹 추론은 스레드 풀로
        result = await asyncio.to_thread(
            pipeline.detect, item.frame_bgr, None, False
        )

        # 프레임 캡처 시점의 드론 pose 스냅샷 → 3D 월드 좌표 계산
        # pose/LiDAR 없으면 좌표 None으로 graceful fallback
        lidar_xyz = self._compute_lidar_xyz()

        # ── ByteTrack 추적 (3-model 파이프라인) ──
        raw_dets = []
        for det in result.yolo_thermal:
            raw_dets.append({
                "class": det.class_, "conf": det.conf,
                "bbox_xyxy": list(det.bbox_xyxy),
                "defect_source": "yolo_thermal",
            })
        for det in result.yolo_delam:
            raw_dets.append({
                "class": det.class_, "conf": det.conf,
                "bbox_xyxy": list(det.bbox_xyxy),
                "defect_source": "yolo_delam",
            })

        if defect_tracker.is_available and raw_dets:
            tracked_dets = defect_tracker.update(raw_dets, frame_id=item.frame_id)
        else:
            tracked_dets = raw_dets

        # ── Temporal Filter (오탐 제거) ──
        lidar_pos = (
            {"x": lidar_xyz[0], "y": lidar_xyz[1], "z": lidar_xyz[2]}
            if lidar_xyz is not None else None
        )
        self._temporal_filter.update(
            tracked_dets, frame_id=item.frame_id, lidar_pos=lidar_pos,
        )

        now = time.time()
        payload = {
            "type": "detection",
            "timestamp": now,
            "frame_id": item.frame_id,
            "result": json.loads(result.model_dump_json()),
            "lidar_position": (
                {"x": lidar_xyz[0], "y": lidar_xyz[1], "z": lidar_xyz[2]}
                if lidar_xyz is not None else None
            ),
            "tracker_stats": {
                "active_tracks": defect_tracker.active_track_count,
                "confirmed_tracks": defect_tracker.confirmed_track_count,
            },
        }

        # 1) /ws/stream 구독자에게 전송 (stream 채널)
        await ws_manager.broadcast("stream", payload)

        # 2) 기존 /ws?channel=defects 구독자에게도 전송 (호환)
        #    레거시 포맷 최소 필드로 변환
        legacy_events = self._to_legacy_events(result, item, lidar_xyz)
        for ev in legacy_events:
            await ws_manager.broadcast("defects", ev)

    async def _process_20(self, item: QueuedFrame) -> None:
        """20종 파이프라인 추론 + ByteTrack 추적 + 시간 필터 + 브로드캐스트."""
        from app.services.inference_pipeline_20 import pipeline20

        if not pipeline20.is_loaded:
            return

        # 계층적 Tier 결정 (프레임 번호 기반)
        fid = item.frame_id
        if fid % settings.TIER3_FRAME_SKIP == 0:
            tier = 3
        elif fid % settings.TIER2_FRAME_SKIP == 0:
            tier = 2
        else:
            tier = 1

        result = await pipeline20.detect_async(
            item.frame_bgr,
            thermal_map=item.thermal_map,
            imu_data=item.imu_data,
            tier=tier,
            thermal_frame_bgr=item.thermal_frame_bgr,
        )

        # ── ByteTrack 객체 추적 ──
        # 검출 결과를 tracker에 통과시켜 track_id 부여 + 일시 미탐지 보완
        raw_dets = [
            {
                "class": d.class_,
                "conf": d.conf,
                "bbox_xyxy": list(d.bbox_xyxy),
                "defect_source": d.defect_source,
                "code": d.code,
                "class_display_en": d.class_display_en,
                "class_display_ko": d.class_display_ko,
                "severity": d.severity,
            }
            for d in result.detections
            if d.bbox_xyxy  # bbox가 있는 검출만 추적 대상
        ]

        # ── ByteTrack 추적 (실패 시 raw_dets fallback) ──
        try:
            if defect_tracker.is_available and raw_dets:
                tracked_dets = defect_tracker.update(raw_dets, frame_id=fid)
            else:
                tracked_dets = raw_dets
        except Exception as e:
            print(f"[StreamInfer] Tracker 오류 (frame={fid}): {e}")
            tracked_dets = raw_dets

        # ── Temporal Filter (실패 시 tracked_dets 그대로 통과) ──
        lidar_xyz = self._compute_lidar_xyz()
        lidar_pos = (
            {"x": lidar_xyz[0], "y": lidar_xyz[1], "z": lidar_xyz[2]}
            if lidar_xyz is not None else None
        )
        try:
            approved_dets = self._temporal_filter.update(
                tracked_dets, frame_id=fid, lidar_pos=lidar_pos,
            )
        except Exception as e:
            print(f"[StreamInfer] TemporalFilter 오류 (frame={fid}): {e}")
            approved_dets = tracked_dets

        # ── Hard Example Mining (실패해도 추론 흐름 중단 안 함) ──
        try:
            hard_example_miner.check_and_collect(
                frame_bgr=item.frame_bgr,
                detections=tracked_dets,
                frame_id=fid,
                anomaly_score=result.anomaly_score,
            )
        except Exception as e:
            print(f"[StreamInfer] HardExampleMiner 오류 (frame={fid}): {e}")

        now = time.time()
        payload = {
            "type": "detection_20",
            "timestamp": now,
            "frame_id": item.frame_id,
            "tier": tier,
            "result": json.loads(result.model_dump_json()),
            "tracker_stats": {
                "active_tracks": defect_tracker.active_track_count,
                "confirmed_tracks": defect_tracker.confirmed_track_count,
            },
        }

        await ws_manager.broadcast("stream", payload)

        # 기존 defects 채널 호환 이벤트 — 필터 통과한 검출만 보고
        for det in approved_dets:
            await ws_manager.broadcast("defects", {
                "type": "defect.new",
                "data": {
                    "area": None,
                    "category_code": det.get("code"),
                    "defect_type": det.get("class_display_ko"),
                    "severity": det.get("severity"),
                    "confidence": round(det.get("conf", 0), 3),
                    "accumulated_conf": det.get("accumulated_conf"),
                    "bbox": None,
                    "defect_source": det.get("defect_source"),
                    "defect_class": det.get("class"),
                    "defect_class_display_en": det.get("class_display_en"),
                    "defect_class_display_ko": det.get("class_display_ko"),
                    "frame_id": item.frame_id,
                    "track_id": det.get("track_id"),
                },
            })

        # ── DB 저장 (실시간 탐지 결과 영구 보존) ──
        if approved_dets:
            await defect_persistence.save_batch(
                detections=approved_dets,
                frame_id=fid,
                tier=tier,
                lidar_pos=lidar_pos,
            )

    # ── VLM 키프레임 오버레이 루프 ───────────────
    async def _vlm_keyframe_loop(self) -> None:
        """
        30fps ONNX 경로와 독립적으로, VLM_KEYFRAME_INTERVAL_SEC 주기마다
        최신 프레임 1장만 비전 LLM에 제출 → "stream" 채널에 vlm_detection 브로드캐스트 + DB 저장.
        VLM 호출은 vlm_detector 내부 세마포어/일일 상한으로 비용 통제.
        어떤 예외도 메인 추론 흐름을 막지 않도록 격리.
        """
        interval = max(1.0, float(settings.VLM_KEYFRAME_INTERVAL_SEC))
        print("[StreamInfer][VLM] 키프레임 루프 시작")
        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            frame = self._last_frame_bgr
            frame_id = self._last_frame_id  # 캡처 당시 id 고정 (이후 submit 으로 드리프트 방지)
            if frame is None:
                continue

            try:
                # JPEG 인코딩은 블로킹 → 스레드 풀
                ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                if not ok:
                    continue
                image_bytes = buf.tobytes()

                lidar_xyz = self._compute_lidar_xyz()
                lidar_pos = (
                    {"x": lidar_xyz[0], "y": lidar_xyz[1], "z": lidar_xyz[2]}
                    if lidar_xyz is not None else None
                )
                now = time.time()

                if settings.VLM_HYBRID_ENABLED:
                    # ── 상업용 경로: ONNX 제안 + VLM 판정 캐스케이드 ──
                    hr = await detect_hybrid_async(image_bytes)
                    self._vlm_count += 1
                    await ws_manager.broadcast("stream", {
                        "type": "hybrid_detection",
                        "timestamp": now,
                        "frame_id": frame_id,
                        "result": json.loads(hr.model_dump_json()),
                        "lidar_position": lidar_pos,
                    })
                    det_dicts = self._hybrid_to_dicts(hr)
                else:
                    # ── VLM 단독 경로 ──
                    vr = await detect_vlm_async(image_bytes)
                    self._vlm_count += 1
                    await ws_manager.broadcast("stream", {
                        "type": "vlm_detection",
                        "timestamp": now,
                        "frame_id": frame_id,
                        "result": json.loads(vr.model_dump_json()),
                        "lidar_position": lidar_pos,
                    })
                    det_dicts = self._vlm_to_dicts(vr)

                # defects 채널 호환 이벤트 + DB 저장
                for d in det_dicts:
                    await ws_manager.broadcast("defects", {
                        "type": "defect.new",
                        "data": {
                            "area": d.get("area"),
                            "category_code": d.get("code"),
                            "defect_type": d.get("class_display_ko"),
                            "severity": d.get("severity"),
                            "confidence": round(d.get("conf", 0), 3),
                            "grade": d.get("grade"),
                            "bbox": None,
                            "defect_source": d.get("defect_source"),
                            "defect_class": d.get("class"),
                            "defect_class_display_ko": d.get("class_display_ko"),
                            "frame_id": frame_id,
                            "localization": d.get("localization"),
                        },
                    })
                if det_dicts:
                    await defect_persistence.save_batch(
                        detections=det_dicts,
                        # 캡처 당시 frame_id 사용 (브로드캐스트와 일치). 과거엔 증가 중인
                        # self._submitted_count 를 넣어 DB 레코드가 broadcast 와 다른 frame_id 로
                        # 저장돼, 검출-프레임 대응(감사 추적)이 어긋났다.
                        frame_id=frame_id,
                        tier=0,  # VLM/하이브리드 경로 마커 (ONNX Tier 1~3과 구분)
                        lidar_pos=lidar_pos,
                    )
            except VLMQuotaExceeded as e:
                self._vlm_errors += 1
                print(f"[StreamInfer][VLM] 일일 상한 — 루프 대기: {e}")
                # 상한 도달 시 다음 날까지 굳이 매 주기 시도하지 않도록 길게 대기
                try:
                    await asyncio.sleep(300)
                except asyncio.CancelledError:
                    break
            except Exception as e:
                self._vlm_errors += 1
                print(f"[StreamInfer][VLM] 키프레임 처리 오류: {e}")

    @staticmethod
    def _vlm_to_dicts(result) -> list:
        """VLMDetectionResult → defect_persistence.save_batch 용 dict 리스트."""
        out = []
        for d in result.detections:
            out.append({
                "class": d.class_,
                "code": d.code,
                "area": d.area,
                "class_display_en": "",
                "class_display_ko": d.class_display_ko,
                "conf": d.conf,
                "severity": d.severity or "LOW",
                "bbox_xyxy": list(d.bbox_xyxy) if d.localization == "bbox" else [],
                "defect_source": "vlm",
                "localization": d.localization,
            })
        return out

    @staticmethod
    def _hybrid_to_dicts(result) -> list:
        """HybridDetectionResult → save_batch 용 dict. 기각(rejected)은 DB 미저장."""
        out = []
        for d in result.detections:
            if d.status == "rejected":
                continue  # VLM이 기각한 ONNX 오탐은 DB에 남기지 않음 (감사 응답엔 포함)
            out.append({
                "class": d.class_,
                "code": d.code,
                "area": d.area,
                "class_display_en": "",
                "class_display_ko": d.class_display_ko,
                "conf": d.conf,
                "severity": d.severity or "LOW",
                "bbox_xyxy": list(d.bbox_xyxy) if d.localization == "bbox" else [],
                "defect_source": d.source,
                "grade": d.grade,
                "localization": d.localization,
                "cross_model_boosted": d.agreement,
            })
        return out

    @staticmethod
    def _compute_lidar_xyz() -> Optional[tuple]:
        """
        최신 드론 pose + LiDAR 전방 거리 → 탐지 대상의 3D 월드 좌표.
        조건:
          - telemetry_cache에 fresh pose 있음 (roll/pitch/yaw 포함)
          - lidar_service 최신 거리 측정값 있음
        둘 중 하나라도 없으면 None 반환 (좌표 미기록).
        """
        pose = telemetry_cache.snapshot_fresh()
        if pose is None or not pose.has_attitude:
            return None
        if lidar_service.latest_distance_m is None:
            return None
        return lidar_service.compute_3d_position(pose.pos_x, pose.pos_y, pose.pos_z)

    @staticmethod
    def _to_legacy_events(result, item: QueuedFrame, lidar_xyz: Optional[tuple] = None) -> list:
        """신규 DetectionResult → 기존 'defect.new' 이벤트 리스트.

        lidar_xyz: (x, y, z) 월드 좌표. None이면 좌표 필드 생략.
        """
        events = []
        img_w = result.image_shape.width
        img_h = result.image_shape.height

        def _legacy_bbox(xyxy):
            cx, cy, bw, bh = xyxy_to_xywhn(xyxy, img_w, img_h)
            return {"x": cx, "y": cy, "w": bw, "h": bh}

        lidar_fields = {}
        if lidar_xyz is not None:
            lidar_fields = {
                "lidar_x": lidar_xyz[0],
                "lidar_y": lidar_xyz[1],
                "lidar_z": lidar_xyz[2],
            }

        for det in result.yolo_thermal:
            area, code, dtype = map_to_legacy("yolo_thermal", det.class_)
            events.append({
                "type": "defect.new",
                "data": {
                    "area": area,
                    "category_code": code,
                    "defect_type": dtype or det.class_display_ko,
                    "severity": "HIGH",
                    "confidence": round(det.conf, 3),
                    "bbox": _legacy_bbox(det.bbox_xyxy),
                    "defect_source": "yolo_thermal",
                    "defect_class": det.class_,
                    "defect_class_display_en": det.class_display_en,
                    "defect_class_display_ko": det.class_display_ko,
                    "frame_id": item.frame_id,
                    **lidar_fields,
                },
            })
        for det in result.yolo_delam:
            area, code, dtype = map_to_legacy("yolo_delam", det.class_)
            events.append({
                "type": "defect.new",
                "data": {
                    "area": area,
                    "category_code": code,
                    "defect_type": dtype or det.class_display_ko,
                    "severity": "HIGH",
                    "confidence": round(det.conf, 3),
                    "bbox": _legacy_bbox(det.bbox_xyxy),
                    "defect_source": "yolo_delam",
                    "defect_class": det.class_,
                    "defect_class_display_en": det.class_display_en,
                    "defect_class_display_ko": det.class_display_ko,
                    "frame_id": item.frame_id,
                    **lidar_fields,
                },
            })
        # 벽지는 신뢰 높을 때만 이벤트화
        if result.wallpaper_cls and result.wallpaper_cls.is_confident:
            wc = result.wallpaper_cls
            area, code, dtype = map_to_legacy("wallpaper", wc.top1_class)
            events.append({
                "type": "defect.new",
                "data": {
                    "area": area,
                    "category_code": code,
                    "defect_type": dtype or wc.top1_class_display_ko,
                    "severity": "MED" if result.severity == "MED" else (result.severity or "LOW"),
                    "confidence": round(wc.top1_conf, 3),
                    "bbox": None,
                    "defect_source": "wallpaper",
                    "defect_class": wc.top1_class,
                    "defect_class_display_en": wc.top1_class_display_en,
                    "defect_class_display_ko": wc.top1_class_display_ko,
                    "frame_id": item.frame_id,
                    **lidar_fields,
                },
            })
        return events


# ── 모듈 레벨 싱글톤 ─────────────────────────
stream_inference_worker = StreamInferenceWorker()


__all__ = ["StreamInferenceWorker", "stream_inference_worker"]
