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
import logging
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.metrics import track_llm_call
from app.db.session import async_session_factory
from app.models.defect import DefectLog
from app.models.site import Site
from app.schemas.report import ReportRequest, ReportResponse, ReportMetadata

logger = logging.getLogger(__name__)

# 보고서 메타데이터 집계에 사용하는 심각도 키 (DefectLog.severity 와 일치)
_SEVERITY_KEYS = ("HIGH", "MED", "LOW")

# ── 외부 LLM 클라이언트 캐시 (호출마다 새 httpx/SDK 클라이언트 생성 방지) ──
_anthropic_client = None
_gemini_model = None


def capture_exception(exc: Exception) -> None:
    """Sentry 가 초기화돼 있으면 예외를 전송 (미설치/미초기화면 무해하게 무시)."""
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def _get_anthropic_client():
    """프로세스당 1개의 AsyncAnthropic 클라이언트 재사용 (TLS/커넥션 풀 재사용)."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    return _anthropic_client


def _get_gemini_model():
    """프로세스당 1개의 Gemini 모델 핸들 재사용. configure 는 1회만 호출."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        _gemini_model = genai.GenerativeModel(settings.REPORT_GEMINI_MODEL)
    return _gemini_model


class LLMReportService:
    """LLM 기반 하자 점검 보고서 생성 서비스"""

    async def generate_stream(
        self,
        request: ReportRequest,
        org_id: UUID,
    ) -> AsyncIterator[str]:
        """
        스트리밍 보고서 생성 (프론트엔드에서 청크 단위 표시).

        주의: StreamingResponse 의 제너레이터는 라우트가 반환된 *이후* 실행되므로
        요청 스코프의 Depends(get_db) 세션은 이미 commit/close 된 상태다.
        따라서 여기서 전용 세션을 직접 연다.
        """
        async with async_session_factory() as db:
            defects = await self._fetch_defects(request, db, org_id)

            if not defects:
                yield "## 점검 결과\n\n탐지된 하자가 없습니다. 정상 상태입니다.\n"
                return

            prompt = self._build_prompt(request, defects)

        # LLM 스트리밍은 DB 세션을 점유하지 않도록 세션 컨텍스트 밖에서 수행
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
        org_id: UUID,
    ) -> ReportResponse:
        """비스트리밍 전체 보고서 생성 (요청 스코프 세션 재사용 가능)."""
        defects = await self._fetch_defects(request, db, org_id)

        severity_counts = {k: 0 for k in _SEVERITY_KEYS}
        for d in defects:
            if d.severity in severity_counts:
                severity_counts[d.severity] += 1

        if not defects:
            content = "## 점검 결과\n\n탐지된 하자가 없습니다. 정상 상태입니다.\n"
        else:
            prompt = self._build_prompt(request, defects)
            chunks = []
            if request.provider == "claude":
                async for chunk in self._stream_claude(prompt):
                    chunks.append(chunk)
            else:
                async for chunk in self._stream_gemini(prompt):
                    chunks.append(chunk)
            content = "".join(chunks)

        return ReportResponse(
            content=content,
            metadata=ReportMetadata(
                generated_at=datetime.now(timezone.utc),
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
        org_id: UUID,
    ) -> list[DefectLog]:
        """보고서 대상 하자 로그 조회 (소속 조직의 현장에 속한 하자로 제한)."""
        # 테넌트 격리: 조직 소유 현장의 site_id 로만 한정
        org_sites = select(Site.id).where(Site.organization_id == org_id)
        query = (
            select(DefectLog)
            .where(DefectLog.site_id.in_(org_sites))
            .order_by(desc(DefectLog.severity), DefectLog.area)
        )

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
        """Claude API 스트리밍 (클라이언트 재사용 + 타임아웃 + 관측 가능한 오류)."""
        try:
            client = _get_anthropic_client()
            async with track_llm_call("claude", "report"):
                async with client.messages.stream(
                    model=settings.REPORT_CLAUDE_MODEL,
                    max_tokens=settings.REPORT_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
        except Exception as e:
            # 원시 예외 문자열(키/내부 디테일 노출)을 본문에 흘리지 않고,
            # 실제 원인은 로그/Sentry 로만 남긴다.
            logger.exception("Claude 보고서 스트리밍 실패")
            capture_exception(e)
            yield "\n\n[오류] 보고서 생성 중 LLM 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.\n"

    async def _stream_gemini(self, prompt: str) -> AsyncIterator[str]:
        """
        Gemini API 스트리밍.
        google-generativeai 의 stream=True 는 *동기* 제너레이터를 반환하므로
        각 청크 fetch 를 to_thread 로 오프로드해 이벤트 루프 블로킹을 방지한다.
        """
        try:
            model = _get_gemini_model()
            async with track_llm_call("gemini", "report"):
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    stream=True,
                    request_options={"timeout": settings.LLM_REQUEST_TIMEOUT},
                )
                # 동기 이터레이터를 루프 밖(스레드)에서 한 청크씩 끌어온다.
                it = iter(response)
                sentinel = object()
                while True:
                    chunk = await asyncio.to_thread(next, it, sentinel)
                    if chunk is sentinel:
                        break
                    text = getattr(chunk, "text", None)
                    if text:
                        yield text
        except Exception as e:
            logger.exception("Gemini 보고서 스트리밍 실패")
            capture_exception(e)
            yield "\n\n[오류] 보고서 생성 중 LLM 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.\n"
