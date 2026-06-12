# =============================================
# app/api/admin_gpu.py
# 역할: GCP GPU VM 원격 제어 엔드포인트
#       - GET  /admin/gpu/status — 인스턴스 상태 (RUNNING / TERMINATED 등)
#       - POST /admin/gpu/start  — 인스턴스 시작 (시간당 과금)
#       - POST /admin/gpu/stop   — 인스턴스 정지 (GPU 과금 중단)
#       - POST /admin/gpu/usage/reset — 누적 사용량 초기화
#
# 인증 (1차 배포):
#   - status / start / stop : 인증된 사용자 전체 (현장에서 직원이 직접 GPU 가동)
#   - usage/reset           : 슈퍼어드민 OR 조직 owner/admin (조직별 GPU 분리 운용 대비)
# =============================================

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.dependencies import get_current_user, require_admin_or_superadmin
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
    usage: UsageBlock | None = None  # 이번 달 누적 — DB 영속(Fly 재배포에도 보존)


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


async def _usage_block() -> UsageBlock:
    # DB 영속 — 조회 실패해도 GPU 제어를 막지 않도록 0 스냅샷 폴백.
    try:
        snap = await gpu_usage_tracker.snapshot()
        return UsageBlock(
            period_start=snap.period_start,
            period_label=snap.period_label,
            completed_seconds=snap.completed_seconds,
            in_progress_seconds=snap.in_progress_seconds,
            total_seconds=snap.total_seconds,
        )
    except Exception as e:  # pragma: no cover - 방어
        print(f"[gpu_usage] snapshot 실패: {e}")
        now_label = ""
        return UsageBlock(period_start="", period_label=now_label,
                          completed_seconds=0, in_progress_seconds=0, total_seconds=0)


@router.get("/status", response_model=GpuStatusResponse)
async def get_gpu_status(_=Depends(get_current_user)):
    """GPU VM 현재 상태 조회 + 이번 달 누적 사용량 (인증된 직원 전체)."""
    try:
        data = await gcp_compute.get_status()
        # GCP 실제 상태와 DB 추적 동기화 (외부 토글/race 보정)
        try:
            await gpu_usage_tracker.reconcile(data.get("status"))
        except Exception as e:
            print(f"[gpu_usage] reconcile 실패: {e}")
        return {**data, "usage": (await _usage_block()).model_dump()}
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/start", response_model=GpuOperationResponse)
async def start_gpu(_=Depends(get_current_user)):
    """GPU VM 시작 — 시간당 ~$0.71 과금 시작 (L4 GPU). 현장에서 직원이 직접 가동."""
    try:
        result = await gcp_compute.start()
        try:
            await gpu_usage_tracker.mark_start()
        except Exception as e:
            print(f"[gpu_usage] mark_start 실패: {e}")
        return result
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/stop", response_model=GpuOperationResponse)
async def stop_gpu(_=Depends(get_current_user)):
    """GPU VM 정지 — GPU 시간당 과금 중단 (디스크/IP 만 ~$13/월 유지)."""
    try:
        result = await gcp_compute.stop()
        try:
            await gpu_usage_tracker.mark_stop()
        except Exception as e:
            print(f"[gpu_usage] mark_stop 실패: {e}")
        return result
    except GcpComputeError as e:
        _raise_5xx(e)


@router.post("/usage/reset", response_model=UsageResetResponse)
async def reset_gpu_usage(_=Depends(require_admin_or_superadmin)):
    """이번 달 누적 사용량 초기화 — 슈퍼어드민 OR 조직 owner/admin 전용.

    진행 중 세션이 있으면 그 시점부터 다시 카운트.
    """
    await gpu_usage_tracker.reset()
    return {"usage": (await _usage_block()).model_dump()}
