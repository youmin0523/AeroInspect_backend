# =============================================
# app/schemas/monitoring.py
# 역할: 운영 모니터링 엔드포인트용 Pydantic 스키마
#       - GET /api/v1/stream/stats  → StreamStatsResponse
#       - GET /api/v1/coverage/{id} → CoverageResponse
# 목적: OpenAPI/Swagger 문서화 + 필드 오타 런타임 방지
# =============================================

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── /api/v1/stream/stats ───────────────────────
class WorkerStats(BaseModel):
    """StreamInferenceWorker 내부 카운터."""
    running: bool = Field(..., description="워커 태스크 실행 중 여부")
    submitted: int = Field(..., description="수신된 총 프레임 수 (스킵 포함)")
    processed: int = Field(..., description="실제로 추론한 프레임 수")
    dropped: int = Field(..., description="큐 가득 차서 버린 프레임 수")
    queue_size: int = Field(..., description="현재 큐 적재량 (maxsize=1)")
    frame_skip: int = Field(..., description="N프레임 중 1프레임만 추론")


class TelemetryCacheStats(BaseModel):
    """telemetry_cache 싱글톤 상태."""
    ready: bool = Field(..., description="최소 1회 이상 텔레메트리 수신됐는지")
    age_sec: Optional[float] = Field(None, description="마지막 업데이트 후 경과 (초). None=미수신")


class LidarStats(BaseModel):
    """LiDAR 연결 및 최신 측정값."""
    connected: bool = Field(..., description="LiDAR 거리 측정값이 유효한지 (None 아님)")
    distance_m: Optional[float] = Field(None, description="최신 필터링 거리 (m)")


class StreamStatsResponse(BaseModel):
    """실시간 추론 워커 + 센서 연결 상태 스냅샷."""
    worker: WorkerStats
    telemetry_cache: TelemetryCacheStats
    lidar: LidarStats


# ── /api/v1/coverage/{site_id} ─────────────────
class CoverageResponse(BaseModel):
    """
    현장별 점검 커버리지 결과.
    텔레메트리(pos_x, pos_y) 샘플의 convex hull 면적 vs. site.total_area.
    """
    site_id: UUID
    covered_area_m2: float = Field(..., description="드론 경로 convex hull 면적 (m²)")
    supplied_area_m2: Optional[float] = Field(None, description="사전 입력된 공급 면적 (m²)")
    coverage_ratio: Optional[float] = Field(
        None,
        description="covered / supplied (0.0 ~ 1.0). supplied 없으면 None.",
    )
    uncovered_area_m2: Optional[float] = Field(
        None,
        description="미점검 면적. supplied - covered (음수 0으로 clamp).",
    )
    sample_count: int = Field(..., description="hull 계산에 쓰인 유효 텔레메트리 점 수")
    hull: List[List[float]] = Field(
        default_factory=list,
        description="외곽 폴리곤 꼭짓점 [[x, y], ...]. 3D 미니맵 음영용.",
    )
    note: Optional[str] = Field(None, description="샘플 부족 등 경고 메시지")
