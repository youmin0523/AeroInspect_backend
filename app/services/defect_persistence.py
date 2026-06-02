# =============================================
# app/services/defect_persistence.py
# 역할: 실시간 추론 결과를 DB에 비동기 저장
#       - stream_inference → approved_dets → DefectLog INSERT
#       - DB 쓰기 실패 시 메모리 버퍼에 보관 → 주기적 재시도
#       - 기존 REST API(defects.py create_defect)와 동일 포맷
#
# 사용: stream_inference.py의 _process_20()에서 호출
# =============================================

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from app.db.session import async_session_factory
from app.models.defect import DefectLog


class DefectPersistenceService:
    """
    실시간 탐지 결과를 DB에 비동기 저장하는 서비스.

    - 정상 시: 즉시 INSERT
    - DB 장애 시: 메모리 버퍼에 보관 (최대 max_buffer건)
    - 주기적 flush로 버퍼 소진 시도
    """

    def __init__(self, max_buffer: int = 1000):
        self._retry_buffer: Deque[dict] = deque(maxlen=max_buffer)
        self._total_saved = 0
        self._total_failed = 0
        self._last_flush_time = time.time()

    async def save_detection(
        self,
        det: dict,
        frame_id: int,
        tier: int,
        lidar_pos: Optional[dict] = None,
        site_id: Optional[str] = None,
    ) -> bool:
        """
        단일 탐지 결과를 DB에 저장.

        Returns:
            True: 저장 성공, False: 버퍼에 보관
        """
        record = self._build_record(det, frame_id, tier, lidar_pos, site_id)

        try:
            async with async_session_factory() as session:
                async with session.begin():
                    defect = DefectLog(**record)
                    session.add(defect)
            self._total_saved += 1
            return True
        except Exception as e:
            print(f"[DefectPersist] DB 저장 실패 (frame={frame_id}): {e}")
            self._retry_buffer.append(record)
            self._total_failed += 1
            return False

    async def save_batch(
        self,
        detections: List[dict],
        frame_id: int,
        tier: int,
        lidar_pos: Optional[dict] = None,
        site_id: Optional[str] = None,
    ) -> int:
        """
        여러 탐지 결과를 일괄 저장.

        Returns:
            저장 성공 건수
        """
        if not detections:
            return 0

        records = [
            self._build_record(d, frame_id, tier, lidar_pos, site_id)
            for d in detections
        ]

        try:
            async with async_session_factory() as session:
                async with session.begin():
                    for rec in records:
                        session.add(DefectLog(**rec))
            self._total_saved += len(records)
            return len(records)
        except Exception as e:
            print(f"[DefectPersist] 배치 저장 실패 ({len(records)}건, frame={frame_id}): {e}")
            for rec in records:
                self._retry_buffer.append(rec)
            self._total_failed += len(records)
            return 0

    async def flush_retry_buffer(self) -> int:
        """버퍼에 쌓인 실패 건 재시도."""
        if not self._retry_buffer:
            return 0

        saved = 0
        remaining: List[dict] = []

        while self._retry_buffer:
            record = self._retry_buffer.popleft()
            try:
                async with async_session_factory() as session:
                    async with session.begin():
                        session.add(DefectLog(**record))
                saved += 1
            except Exception:
                remaining.append(record)

        for rec in remaining:
            self._retry_buffer.append(rec)

        self._total_saved += saved
        self._last_flush_time = time.time()
        return saved

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_saved": self._total_saved,
            "total_failed": self._total_failed,
            "retry_buffer_size": len(self._retry_buffer),
        }

    @staticmethod
    def _build_record(
        det: dict,
        frame_id: int,
        tier: int,
        lidar_pos: Optional[dict] = None,
        site_id: Optional[str] = None,
    ) -> dict:
        """검출 dict → DefectLog 컬럼 매핑."""
        bbox = det.get("bbox_xyxy")
        # xyxy → normalized center (xywhn) 변환은 이미지 크기 필요
        # 여기서는 원본 pixel 좌표 저장 (정규화는 프론트엔드에서)

        return {
            "site_id": site_id,
            "area": None,  # taxonomy 매핑은 별도
            "category_code": det.get("code"),
            "defect_type": det.get("class_display_ko"),
            "defect_source": det.get("defect_source"),
            "defect_class": det.get("class"),
            "defect_class_display_en": det.get("class_display_en"),
            "defect_class_display_ko": det.get("class_display_ko"),
            "severity": det.get("severity", "LOW"),
            "confidence": det.get("conf", 0),
            # R-v1.1.17 TODO: grade(신뢰도 등급) DB 영속화 — DefectLog 모델 + alembic 추가 후 활성화
            # 현재는 API 응답/WS broadcast 경로에서만 grade 노출 (DB 미저장)
            "bbox_x": ((bbox[0] + bbox[2]) / 2) if bbox and len(bbox) == 4 else None,
            "bbox_y": ((bbox[1] + bbox[3]) / 2) if bbox and len(bbox) == 4 else None,
            "bbox_w": (bbox[2] - bbox[0]) if bbox and len(bbox) == 4 else None,
            "bbox_h": (bbox[3] - bbox[1]) if bbox and len(bbox) == 4 else None,
            "lidar_x": lidar_pos.get("x") if lidar_pos else None,
            "lidar_y": lidar_pos.get("y") if lidar_pos else None,
            "lidar_z": lidar_pos.get("z") if lidar_pos else None,
            "frame_id": frame_id,
            "track_id": det.get("track_id"),
            "accumulated_conf": det.get("accumulated_conf"),
            "tier_executed": tier,
            "ensemble_boosted": str(det.get("ensemble_boosted", False)).lower() if det.get("ensemble_boosted") else None,
            "raw_payload": det,
        }


# ── 모듈 레벨 싱글톤 ─────────────────────────
defect_persistence = DefectPersistenceService()


__all__ = ["DefectPersistenceService", "defect_persistence"]
