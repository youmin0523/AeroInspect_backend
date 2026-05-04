import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func
from app.db.session import async_session_factory
from app.models.site import Site
from app.models.defect import DefectLog
from app.models.report import Report
from app.models.inspection_schedule import InspectionSchedule
from app.models.organization import Organization


async def check():
    async with async_session_factory() as s:
        org = await s.scalar(select(Organization).where(Organization.biz_number == "0000000001"))
        print(f"Org: {org.name} (id={org.id})")
        sites = (await s.execute(select(Site).where(Site.organization_id == org.id))).scalars().all()
        print(f"Sites: {len(sites)}")
        for site in sites:
            print(f"  - [{site.status}] {site.name}")
        site_ids = [s.id for s in sites]
        defects = await s.scalar(select(func.count(DefectLog.id)).where(DefectLog.site_id.in_(site_ids)))
        high = await s.scalar(select(func.count(DefectLog.id)).where(DefectLog.site_id.in_(site_ids), DefectLog.severity == "HIGH"))
        print(f"Defects total: {defects} (HIGH: {high})")
        reports = await s.scalar(select(func.count(Report.id)).where(Report.site_id.in_(site_ids)))
        print(f"Reports: {reports}")
        scheds = (await s.execute(select(InspectionSchedule).where(InspectionSchedule.organization_id == org.id))).scalars().all()
        print(f"Schedules: {len(scheds)}")
        for sc in scheds:
            site = await s.scalar(select(Site).where(Site.id == sc.site_id))
            print(f"  {sc.scheduled_at.strftime('%H:%M UTC')} | {site.name} | status={sc.status}")


asyncio.run(check())
