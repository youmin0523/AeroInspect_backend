# =============================================
# app/api/contact.py
# 역할: 랜딩 페이지 "도입 문의" 폼 수신 엔드포인트
#       - POST /contact → 문의 내용을 슈퍼어드민에게 알림으로 발송
#       - 비로그인 접근 허용 (랜딩 페이지에서 사용)
#       - Rate Limit 미들웨어가 IP 단위로 자연 제한
#
#   향후 보강:
#       - 별도 ContactInquiry 모델 + 마이그레이션 추가하여 DB 영구 저장
#       - SMTP 영업 메일 자동 발송
# =============================================

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.user import User
from app.services.notification_service import notification_service

router = APIRouter()


class ContactInquiryRequest(BaseModel):
    customer_type: str = Field(..., pattern="^(personal|business)$")
    biz_number: Optional[str] = Field(None, max_length=20)
    name: str = Field(..., min_length=1, max_length=50)
    phone: str = Field(..., min_length=1, max_length=20)
    message: str = Field(..., min_length=5, max_length=2000)


class ContactInquiryResponse(BaseModel):
    received: bool
    notified_admins: int


@router.post("", response_model=ContactInquiryResponse, status_code=201)
async def submit_contact_inquiry(
    payload: ContactInquiryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    랜딩 페이지 도입 문의 접수.
    슈퍼어드민 사용자에게 알림(notification.category="system") 으로 전달.
    """
    admin_rows = await db.scalars(
        select(User.id).where(User.is_superadmin.is_(True))
    )
    admin_ids = list(admin_rows.all())

    bizline = f" / 사업자번호 {payload.biz_number}" if payload.biz_number else ""
    title = f"신규 도입 문의 ({payload.customer_type})"
    body = (
        f"{payload.name} · {payload.phone}{bizline}\n"
        f"---\n{payload.message[:300]}"
    )

    if admin_ids:
        await notification_service.create_for_many(
            db=db,
            user_ids=admin_ids,
            category="system",
            title=title,
            message=body,
            metadata={
                "kind": "contact_inquiry",
                "customer_type": payload.customer_type,
                "biz_number": payload.biz_number,
                "name": payload.name,
                "phone": payload.phone,
            },
        )

    return ContactInquiryResponse(received=True, notified_admins=len(admin_ids))
