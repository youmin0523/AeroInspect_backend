# =============================================
# app/services/gpu_usage.py
# 역할: GCP GPU VM 이번 달 누적 사용 시간 트래커 (인메모리 싱글톤)
#       - 매월 1일 KST 0시 자동 롤오버
#       - 사용자 명시 리셋 시 즉시 0으로 + 기준 시점을 now로
#       - start/stop 호출 시점에 누적 분 단위 갱신
#       - Fly.io 머신 재배포 시 인메모리라 휘발 — 운영자 참고용
# 사용처: app/api/admin_gpu.py
# =============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

KST = timezone(timedelta(hours=9))


@dataclass
class UsageSnapshot:
    """status 응답에 합쳐 반환되는 누적 스냅샷."""

    period_start: str          # 누적 기준 시점 (KST ISO)
    period_label: str          # 표시용 라벨 (예: "2026-05")
    completed_seconds: int     # 정지된 세션들의 누적 (이번 달)
    in_progress_seconds: int   # 현재 RUNNING 중이면 (now - last_start_at), 아니면 0
    total_seconds: int         # completed + in_progress


class GpuUsageTracker:
    """이번 달 누적 GPU 사용 시간 인메모리 트래커.

    설계:
        - `_completed_seconds`: stop 시점에 더해진 완료 세션의 합 (초)
        - `_running_since`: 현재 RUNNING 중이면 start 시각, 아니면 None
        - `_period_start`: 이번 누적 기간의 시작 시점 (월 1일 KST 또는 마지막 reset 시각)

    호출 진입점에서 매번 `_rollover_if_needed()`를 먼저 호출해
    KST 기준 월이 바뀌면 자동으로 누적을 0으로 초기화한다.
    """

    def __init__(self) -> None:
        now_kst = datetime.now(KST)
        self._period_start: datetime = self._month_start(now_kst)
        self._completed_seconds: int = 0
        self._running_since: Optional[datetime] = None

    # ── 헬퍼 ──────────────────────────────
    @staticmethod
    def _month_start(dt: datetime) -> datetime:
        """주어진 KST datetime의 해당 월 1일 0시 0분 0초 (KST)."""
        return datetime(dt.year, dt.month, 1, 0, 0, 0, tzinfo=KST)

    def _rollover_if_needed(self) -> None:
        """KST 기준 현재 월이 _period_start의 월과 다르면 누적 초기화."""
        now_kst = datetime.now(KST)
        cur_month_start = self._month_start(now_kst)
        if cur_month_start > self._period_start:
            # 진행 중 세션이 있으면 새 달의 시작 시점부터 다시 카운트되도록 보존
            self._completed_seconds = 0
            self._period_start = cur_month_start
            # _running_since는 그대로 두면 새 달 누적에 자연 포함됨

    # ── 이벤트 진입점 ──────────────────────
    def mark_start(self) -> None:
        """admin_gpu start 호출 시. 이미 RUNNING으로 보이면 무시(idempotent)."""
        self._rollover_if_needed()
        if self._running_since is None:
            self._running_since = datetime.now(KST)

    def mark_stop(self) -> None:
        """admin_gpu stop 호출 시. 진행 중 세션을 누적에 더하고 종료."""
        self._rollover_if_needed()
        if self._running_since is not None:
            elapsed = (datetime.now(KST) - self._running_since).total_seconds()
            # 음수/이상치 방어
            if elapsed > 0:
                self._completed_seconds += int(elapsed)
            self._running_since = None

    def reset(self) -> None:
        """사용자 리셋. 누적을 0으로, 기준 시점을 now로. 진행 중 세션도 절단."""
        now_kst = datetime.now(KST)
        self._completed_seconds = 0
        self._period_start = now_kst
        # 진행 중이면 reset 시점부터 다시 카운트되도록 _running_since를 갱신
        if self._running_since is not None:
            self._running_since = now_kst

    # ── 외부 동기화 (운영 안전망) ──────────
    def reconcile(self, gcp_status: Optional[str]) -> None:
        """GCP 실제 상태와 인메모리 추적이 어긋났을 때 보정.

        - GCP가 RUNNING인데 _running_since가 None: 누군가 우리 API 외부에서 켰음.
          → 지금부터라도 카운트 시작 (과거 시간은 영영 모름 — 인메모리 한계).
        - GCP가 RUNNING이 아닌데 _running_since가 있음: 외부 정지 또는 토큰 race.
          → mark_stop 처리.
        """
        self._rollover_if_needed()
        if gcp_status == "RUNNING" and self._running_since is None:
            self._running_since = datetime.now(KST)
        elif gcp_status not in ("RUNNING", "STAGING", "PROVISIONING") and self._running_since is not None:
            self.mark_stop()

    def snapshot(self) -> UsageSnapshot:
        """현재 누적 스냅샷."""
        self._rollover_if_needed()
        in_progress = 0
        if self._running_since is not None:
            in_progress = int((datetime.now(KST) - self._running_since).total_seconds())
            if in_progress < 0:
                in_progress = 0
        return UsageSnapshot(
            period_start=self._period_start.isoformat(),
            period_label=self._period_start.strftime("%Y-%m"),
            completed_seconds=self._completed_seconds,
            in_progress_seconds=in_progress,
            total_seconds=self._completed_seconds + in_progress,
        )


gpu_usage_tracker = GpuUsageTracker()
