# =============================================
# app/api/admin_gpu.py
# 역할: 관리자용 GCP GPU VM 원격 제어 엔드포인트
#       - GET  /admin/gpu/status — 인스턴스 상태 (RUNNING / TERMINATED 등)
#       - POST /admin/gpu/start  — 인스턴스 시작 (시간당 과금)
#       - POST /admin/gpu/stop   — 인스턴스 정지 (GPU 과금 중단)
#
# 인증: 슈퍼어드민 전용 (require_superadmin)
# 호출 흐름: 프론트(브라우저) → Fly.io 백엔드(이 라우터) → GCP Compute REST API
#   ↳ 로컬 bat 파일 의존 제거. 어떤 브라우저에서도 admin 권한이면 켜고 끌 수 있음.
# =============================================

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.dependencies import require_superadmin
from app.services.gcp_compute import GcpComputeError, gcp_compute
from app.services.gpu_usage import gpu_usage_tracker

router = APIRouter()


class UsageBlock(BaseModel):
    """이번 달 누적 사용량 (KST 기준 매월 1일 자동 롤오버, 사용자 리셋 가능)."""

    period_start: str          # 누적 기준 시점 ISO (KST)
    period_label: str          # "2026-05"
    completed_seconds: int     # 완료된 세션들의 합
    in_progress_seconds: int   # 현재 진행 중 세션 (RUNNING 상태일 때만 > 0)
    total_seconds: int         # completed + in_progress


class GpuStatusResponse(BaseModel):
    name: str | None = None
    status: str | None = None  # RUNNING / TERMINATED / STOPPING / PROVISIONING / STAGING
    zone: str | None = None
    machine_type: str | None = None
    last_start_at: str | None = None
    last_stop_at: str | None = None
    usage: UsageBlock | None = None  # 이번 달 누적 — 인메모리 한계로 머신 재배포 시 0


class GpuOperationResponse(BaseModel):
    operation: str | None = None
    status: str | None = None  # PENDING / RUNNING / DONE


class UsageResetResponse(BaseModel):
    usage: UsageBlock


def _raise_5xx(e: GcpComputeError) -> None:
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"GCP Compute 호출 실패: {e}",
    )


def _usage_block() -> UsageBlock:
    snap = gpu_usage_tracker.snapshot()
    return UsageBlock(
        period_start=snap.period_start,
        period_label=snap.period_label,
        completed_seconds=snap.completed_seconds,
        in_progress_seconds=snap.in_progress_seconds,
        total_seconds=snap.total_seconds,
    )


@router.get("/status", response_model=GpuStatusResponse)
async def get_gpu_status(_=Depends(require_superadmin)):
    """GPU VM 현재 상태 조회 + 이번 달 누적 사용량 (슈퍼어드민 전용)."""
    try:
        data = await gcp_compute.get_status()
        # GCP 실제 상태와 인메모리 추적 동기화 (외부 토글/race 보정)
        gpu_usage_tracker.reconcile(data.get("status"))
        return {**data, "usage": _usage_block().model_dump()}
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/start", response_model=GpuOperationResponse)
async def start_gpu(_=Depends(require_superadmin)):
    """GPU VM 시작 — 시간당 ~$0.71 과금 시작 (L4 GPU)."""
    try:
        result = await gcp_compute.start()
        gpu_usage_tracker.mark_start()
        return result
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/stop", response_model=GpuOperationResponse)
async def stop_gpu(_=Depends(require_superadmin)):
    """GPU VM 정지 — GPU 시간당 과금 중단 (디스크/IP 만 ~$13/월 유지)."""
    try:
        result = await gcp_compute.stop()
        gpu_usage_tracker.mark_stop()
        return result
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/usage/reset", response_model=UsageResetResponse)
async def reset_gpu_usage(_=Depends(require_superadmin)):
    """이번 달 누적 사용량 초기화. 진행 중 세션이 있으면 그 시점부터 다시 카운트."""
    gpu_usage_tracker.reset()
    return {"usage": _usage_block().model_dump()}
