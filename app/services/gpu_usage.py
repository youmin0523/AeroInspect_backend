# =============================================
# app/services/gpu_usage.py
# 역할: GCP GPU VM 이번 달 누적 사용 시간 트래커 (DB 영속)
#       - 매월 1일 KST 0시 자동 롤오버
#       - 사용자 명시 리셋 시 즉시 0으로 + 기준 시점을 now로
#       - start/stop 호출 시점에 누적 초 단위 갱신
#       - **DB(gpu_usage 단일 행)에 영속 → Fly.io 재배포/재시작에도 보존**
#         (과거: 인메모리라 Fly 재배포마다 휘발돼 누적이 0으로 리셋되는 문제)
#       - gpu_usage 테이블은 첫 사용 시 CREATE TABLE IF NOT EXISTS 로 자체 부트스트랩
#         (운영 create_all 스킵·마이그레이션 없이 동작)
# 사용처: app/api/admin_gpu.py
# =============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import text

from app.db.session import async_session_factory

KST = timezone(timedelta(hours=9))

_DDL = """
CREATE TABLE IF NOT EXISTS gpu_usage (
    id INTEGER PRIMARY KEY,
    period_start TIMESTAMPTZ NOT NULL,
    completed_seconds INTEGER NOT NULL DEFAULT 0,
    running_since TIMESTAMPTZ
)
"""


@dataclass
class UsageSnapshot:
    """status 응답에 합쳐 반환되는 누적 스냅샷."""

    period_start: str          # 누적 기준 시점 (KST ISO)
    period_label: str          # 표시용 라벨 (예: "2026-05")
    completed_seconds: int     # 정지된 세션들의 누적 (이번 달)
    in_progress_seconds: int   # 현재 RUNNING 중이면 (now - running_since), 아니면 0
    total_seconds: int         # completed + in_progress


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, 0, 0, 0, tzinfo=KST)


class GpuUsageTracker:
    """이번 달 누적 GPU 사용 시간 — DB 영속(단일 행 id=1).

    모든 진입점이 _load() 로 행을 읽으며 KST 월 변경 시 자동 롤오버한다.
    DB 오류가 나도 운영을 막지 않도록 각 메서드는 호출부(admin_gpu)에서 예외를 감싼다.
    """

    async def _ensure(self, s) -> None:
        await s.execute(text(_DDL))
        await s.execute(
            text("INSERT INTO gpu_usage (id, period_start, completed_seconds, running_since) "
                 "VALUES (1, :ps, 0, NULL) ON CONFLICT (id) DO NOTHING"),
            {"ps": _month_start(datetime.now(KST))},
        )

    async def _load(self, s) -> Tuple[datetime, int, Optional[datetime]]:
        """행 로드 + 월 롤오버. (period_start, completed_seconds, running_since) 반환."""
        await self._ensure(s)
        row = (await s.execute(
            text("SELECT period_start, completed_seconds, running_since FROM gpu_usage WHERE id=1")
        )).first()
        ps, comp, run = row[0], int(row[1]), row[2]
        cur_ms = _month_start(datetime.now(KST))
        if cur_ms > _month_start(ps.astimezone(KST)):
            comp = 0
            ps = cur_ms
            await s.execute(
                text("UPDATE gpu_usage SET period_start=:ps, completed_seconds=0 WHERE id=1"),
                {"ps": cur_ms},
            )
        return ps, comp, run

    async def mark_start(self) -> None:
        """admin_gpu start 호출 시. 이미 RUNNING이면 무시(idempotent)."""
        async with async_session_factory() as s:
            _, _, run = await self._load(s)
            if run is None:
                await s.execute(text("UPDATE gpu_usage SET running_since=:r WHERE id=1"),
                                {"r": datetime.now(KST)})
            await s.commit()

    async def mark_stop(self) -> None:
        """admin_gpu stop 호출 시. 진행 중 세션을 누적에 더하고 종료."""
        async with async_session_factory() as s:
            _, comp, run = await self._load(s)
            if run is not None:
                elapsed = (datetime.now(KST) - run.astimezone(KST)).total_seconds()
                if elapsed > 0:
                    comp += int(elapsed)
                await s.execute(
                    text("UPDATE gpu_usage SET completed_seconds=:c, running_since=NULL WHERE id=1"),
                    {"c": comp})
            await s.commit()

    async def reset(self) -> None:
        """사용자 리셋. 누적 0, 기준 시점 now. 진행 중이면 now부터 다시 카운트."""
        async with async_session_factory() as s:
            _, _, run = await self._load(s)
            now = datetime.now(KST)
            await s.execute(
                text("UPDATE gpu_usage SET completed_seconds=0, period_start=:p, running_since=:r WHERE id=1"),
                {"p": now, "r": (now if run is not None else None)})
            await s.commit()

    async def reconcile(self, gcp_status: Optional[str]) -> None:
        """GCP 실제 상태와 추적이 어긋났을 때 보정(외부 토글/race 안전망)."""
        async with async_session_factory() as s:
            _, comp, run = await self._load(s)
            if gcp_status == "RUNNING" and run is None:
                await s.execute(text("UPDATE gpu_usage SET running_since=:r WHERE id=1"),
                                {"r": datetime.now(KST)})
            elif gcp_status not in ("RUNNING", "STAGING", "PROVISIONING") and run is not None:
                elapsed = (datetime.now(KST) - run.astimezone(KST)).total_seconds()
                if elapsed > 0:
                    comp += int(elapsed)
                await s.execute(
                    text("UPDATE gpu_usage SET completed_seconds=:c, running_since=NULL WHERE id=1"),
                    {"c": comp})
            await s.commit()

    async def snapshot(self) -> UsageSnapshot:
        """현재 누적 스냅샷."""
        async with async_session_factory() as s:
            ps, comp, run = await self._load(s)
            await s.commit()
        in_progress = 0
        if run is not None:
            in_progress = int((datetime.now(KST) - run.astimezone(KST)).total_seconds())
            if in_progress < 0:
                in_progress = 0
        ps_kst = ps.astimezone(KST)
        return UsageSnapshot(
            period_start=ps_kst.isoformat(),
            period_label=ps_kst.strftime("%Y-%m"),
            completed_seconds=comp,
            in_progress_seconds=in_progress,
            total_seconds=comp + in_progress,
        )


gpu_usage_tracker = GpuUsageTracker()
