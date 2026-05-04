# =============================================
# app/schemas/report.py
# 역할: LLM 하자 점검 보고서 요청/응답 Pydantic 스키마
#       - ReportRequest: 보고서 생성 요청 파라미터
#       - ReportResponse: 생성 완료된 보고서 응답
#       - 실제 보고서 내용은 StreamingResponse로 청크 단위 전송
# =============================================

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    """
    하자 점검 보고서 생성 요청.
    특정 세션 또는 시간 범위의 탐지 결과를 기반으로 보고서 생성.
    """
    # 보고서에 포함할 하자 ID 목록 (없으면 전체 탐지 결과 사용)
    defect_ids: Optional[List[str]] = None

    # 시간 범위 필터 (없으면 전체 기간)
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None

    # 보고서 언어 ("ko": 한국어, "en": 영어)
    language: str = Field(default="ko", pattern="^(ko|en)$")

    # LLM 제공자 선택
    provider: str = Field(default="claude", pattern="^(claude|gemini)$")

    # 보고서에 이미지 크롭 첨부 여부
    include_images: bool = True

    # 추가 맥락 정보 (건물명, 점검자명 등)
    inspection_title: Optional[str] = Field(None, max_length=200)
    inspector_name: Optional[str] = Field(None, max_length=100)
    building_name: Optional[str] = Field(None, max_length=200)


class ReportMetadata(BaseModel):
    """생성된 보고서 메타데이터"""
    generated_at: datetime
    provider: str
    defect_count: int
    high_count: int
    med_count: int
    low_count: int
    inspection_title: Optional[str]
    inspector_name: Optional[str]
    building_name: Optional[str]


class ReportResponse(BaseModel):
    """
    완성된 보고서 응답 (비스트리밍 방식).
    스트리밍 방식은 StreamingResponse로 직접 반환.
    """
    content: str            # 마크다운 형식 보고서 본문
    metadata: ReportMetadata


# ── 보고서 저장/조회 스키마 ──────────────────
class ReportSaveRequest(BaseModel):
    """생성된 보고서 DB 저장 요청"""
    title: Optional[str] = Field(None, max_length=200)
    building_name: Optional[str] = Field(None, max_length=200)
    inspector_name: Optional[str] = Field(None, max_length=100)
    provider: str = Field(default="claude", pattern="^(claude|gemini)$")
    content: str = Field(..., description="마크다운 보고서 본문")
    defect_count: int = 0
    high_count: int = 0
    med_count: int = 0
    low_count: int = 0


class ReportSavedResponse(BaseModel):
    """저장된 보고서 응답"""
    id: str
    title: Optional[str]
    building_name: Optional[str]
    inspector_name: Optional[str]
    provider: Optional[str]
    content: str
    defect_count: int
    high_count: int
    med_count: int
    low_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class ReportListResponse(BaseModel):
    """보고서 목록 응답"""
    items: List[ReportSavedResponse]
    total: int
