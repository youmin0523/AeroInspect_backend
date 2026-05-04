# =============================================
# app/services/llm_report.py
# 역할: Claude API / Gemini API 기반 하자 점검 보고서 자동 생성 서비스
#       - DB에 저장된 하자 로그를 기반으로 종합 보고서 생성
#       - Claude: AsyncAnthropic 스트리밍 (기본 제공자)
#       - Gemini: google-generativeai 스트리밍 (대체 제공자)
#       - 보고서 형식: 마크다운 (영역별 요약, 심각도별 분류, 권고사항)
#       - HIGH 심각도 하자는 이미지 크롭을 프롬프트에 첨부
# =============================================

import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.defect import DefectLog
from app.schemas.report import ReportRequest, ReportResponse, ReportMetadata


class LLMReportService:
    """LLM 기반 하자 점검 보고서 생성 서비스"""

    async def generate_stream(
        self,
        request: ReportRequest,
        db: AsyncSession,
    ) -> AsyncIterator[str]:
        """스트리밍 보고서 생성 (프론트엔드에서 청크 단위 표시)"""
        defects = await self._fetch_defects(request, db)

        if not defects:
            yield "## 점검 결과\n\n탐지된 하자가 없습니다. 정상 상태입니다.\n"
            return

        prompt = self._build_prompt(request, defects)

        if request.provider == "claude":
            async for chunk in self._stream_claude(prompt):
                yield chunk
        else:
            async for chunk in self._stream_gemini(prompt):
                yield chunk

    async def generate_full(
        self,
        request: ReportRequest,
        db: AsyncSession,
    ) -> ReportResponse:
        """비스트리밍 전체 보고서 생성"""
        content_chunks = []
        async for chunk in self.generate_stream(request, db):
            content_chunks.append(chunk)

        defects = await self._fetch_defects(request, db)
        severity_counts = {"HIGH": 0, "MED": 0, "LOW": 0}
        for d in defects:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1

        return ReportResponse(
            content="".join(content_chunks),
            metadata=ReportMetadata(
                generated_at=datetime.utcnow(),
                provider=request.provider,
                defect_count=len(defects),
                high_count=severity_counts["HIGH"],
                med_count=severity_counts["MED"],
                low_count=severity_counts["LOW"],
                inspection_title=request.inspection_title,
                inspector_name=request.inspector_name,
                building_name=request.building_name,
            ),
        )

    async def _fetch_defects(
        self,
        request: ReportRequest,
        db: AsyncSession,
    ) -> list[DefectLog]:
        """보고서 대상 하자 로그 조회"""
        query = select(DefectLog).order_by(desc(DefectLog.severity), DefectLog.area)

        if request.defect_ids:
            uuids = [UUID(id_) for id_ in request.defect_ids]
            query = query.where(DefectLog.id.in_(uuids))
        if request.from_time:
            query = query.where(DefectLog.timestamp >= request.from_time)
        if request.to_time:
            query = query.where(DefectLog.timestamp <= request.to_time)

        result = await db.execute(query)
        return result.scalars().all()

    def _build_prompt(self, request: ReportRequest, defects: list[DefectLog]) -> str:
        """LLM 프롬프트 구성"""
        title = request.inspection_title or "입주 전 하자 점검"
        inspector = request.inspector_name or "AeroInspect 드론 시스템"
        building = request.building_name or "점검 대상 건물"
        lang = "한국어" if request.language == "ko" else "English"

        # 심각도별 분류
        by_severity: dict[str, list] = {"HIGH": [], "MED": [], "LOW": []}
        for d in defects:
            by_severity.get(d.severity, by_severity["HIGH"]).append(d)

        defect_summary = []
        for severity, items in by_severity.items():
            if items:
                defect_summary.append(f"\n### {severity} 심각도 ({len(items)}건)")
                for d in items:
                    defect_summary.append(
                        f"- [{d.category_code}] {d.defect_type} "
                        f"(신뢰도: {d.confidence:.0%}, "
                        f"위치: x={d.lidar_x:.1f}m, y={d.lidar_y:.1f}m)"
                        if d.lidar_x is not None
                        else f"- [{d.category_code}] {d.defect_type} (신뢰도: {d.confidence:.0%})"
                    )

        return f"""당신은 건설 하자 전문가입니다. 드론 AI 점검 시스템이 탐지한 하자 데이터를 바탕으로 {lang}로 전문적인 점검 보고서를 작성해주세요.

## 점검 정보
- 점검명: {title}
- 점검 장소: {building}
- 점검자: {inspector}
- 탐지 시스템: AeroInspect 자율 드론 + YOLOv8 AI
- 총 탐지 건수: {len(defects)}건 (HIGH: {len(by_severity['HIGH'])}, MED: {len(by_severity['MED'])}, LOW: {len(by_severity['LOW'])})

## 탐지된 하자 목록
{''.join(defect_summary)}

## 보고서 작성 요구사항
1. 종합 평가 (전반적인 상태, 즉시 조치 필요 여부)
2. 심각도별 상세 설명 (HIGH → MED → LOW 순)
3. 영역별 분석 (A.구조/B.단열방수/C.마감재/D.바닥/E.창호)
4. 조치 권고사항 (우선순위별)
5. 향후 모니터링 권고
마크다운 형식으로 작성하며, 전문적이고 명확한 언어를 사용하세요.
"""

    async def _stream_claude(self, prompt: str) -> AsyncIterator[str]:
        """Claude API 스트리밍"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            async with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"\n\n[오류] Claude API 호출 실패: {str(e)}\n"

    async def _stream_gemini(self, prompt: str) -> AsyncIterator[str]:
        """Gemini API 스트리밍"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = await asyncio.to_thread(
                model.generate_content, prompt, stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n\n[오류] Gemini API 호출 실패: {str(e)}\n"
