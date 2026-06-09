# =============================================
# app/services/email_service.py
# 역할: SMTP 이메일 발송 서비스
#       - 아이디 찾기 결과 발송
#       - 임시 비밀번호 발송
#       SMTP 미설정 시 콘솔 로그로 fallback (개발 편의)
# =============================================

import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import settings

logger = logging.getLogger(__name__)

# SMTP 미설정으로 발송을 건너뛴 경우를 "성공"과 구분하기 위한 센티넬.
# bool 컨텍스트에서는 truthy → 임시 비밀번호 회전 롤백을 유발하지 않으면서도,
# 호출자가 `is EMAIL_SKIPPED` 로 "조용히 건너뜀" 을 식별할 수 있다.
EMAIL_SKIPPED = "skipped"


def _smtp_configured() -> bool:
    """SMTP 자격 증명이 설정되었는지 확인."""
    return bool(settings.SMTP_USER and settings.SMTP_PASSWORD
                and settings.SMTP_USER != "your-email@gmail.com")


def _send_email_sync(to: str, subject: str, html_body: str) -> bool:
    """블로킹 SMTP 발송 본체 — asyncio.to_thread 로 워커 스레드에서 실행된다."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM}>"
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, to, msg.as_string())
        logger.info(f"[EMAIL] 발송 완료: {to} / {subject}")
        return True
    except Exception as e:
        logger.error(f"[EMAIL] 발송 실패: {to} — {e}")
        return False


async def _send_email(to: str, subject: str, html_body: str):
    """
    SMTP로 HTML 이메일을 전송한다 (블로킹 SMTP 는 스레드로 오프로드 → 이벤트 루프 비차단).
    SMTP 미설정 시 콘솔 출력으로 대체하고 EMAIL_SKIPPED 를 반환하여
    "발송됨" 과 "조용히 건너뜀" 을 호출자가 구분할 수 있게 한다.

    Returns:
        True  — 실제 발송 성공
        False — 발송 시도 실패
        EMAIL_SKIPPED — SMTP 미설정으로 발송 건너뜀 (truthy)
    """
    if not _smtp_configured():
        logger.warning(
            "[EMAIL-DEV] SMTP 미설정 — 콘솔 출력으로 대체\n"
            f"  To: {to}\n  Subject: {subject}\n  Body:\n{html_body}"
        )
        return EMAIL_SKIPPED  # 발송 안 됨 — 성공과 구분되는 센티넬(truthy)

    return await asyncio.to_thread(_send_email_sync, to, subject, html_body)


async def send_found_username_email(to: str, name: str, username: str):
    """아이디 찾기 결과 이메일 발송."""
    subject = "[DRONE INSPECT] 아이디 찾기 결과 안내"
    html = f"""
    <div style="font-family:'Malgun Gothic',sans-serif;max-width:480px;margin:0 auto;padding:24px;">
      <h2 style="color:#1e293b;">DRONE INSPECT 아이디 안내</h2>
      <p>{name}님, 요청하신 아이디 정보를 안내드립니다.</p>
      <div style="background:#f1f5f9;padding:16px 20px;border-radius:8px;margin:16px 0;">
        <p style="margin:0;font-size:14px;color:#64748b;">아이디</p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:bold;color:#1e40af;">{username}</p>
      </div>
      <p style="font-size:13px;color:#94a3b8;">
        본 메일은 발신전용이며, 로그인은 <a href="{settings.FRONTEND_BASE_URL}/login">여기</a>에서 가능합니다.
      </p>
    </div>
    """
    return await _send_email(to, subject, html)


async def send_temp_password_email(to: str, name: str, temp_password: str):
    """임시 비밀번호 이메일 발송."""
    subject = "[DRONE INSPECT] 임시 비밀번호 발급 안내"
    html = f"""
    <div style="font-family:'Malgun Gothic',sans-serif;max-width:480px;margin:0 auto;padding:24px;">
      <h2 style="color:#1e293b;">DRONE INSPECT 임시 비밀번호</h2>
      <p>{name}님, 요청하신 임시 비밀번호를 안내드립니다.</p>
      <div style="background:#fef3c7;padding:16px 20px;border-radius:8px;margin:16px 0;">
        <p style="margin:0;font-size:14px;color:#92400e;">임시 비밀번호</p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:bold;color:#b45309;letter-spacing:1px;">{temp_password}</p>
      </div>
      <p style="color:#dc2626;font-size:13px;font-weight:bold;">
        보안을 위해 로그인 후 반드시 비밀번호를 변경해주세요.
      </p>
      <p style="font-size:13px;color:#94a3b8;">
        본 메일은 발신전용이며, 로그인은 <a href="{settings.FRONTEND_BASE_URL}/login">여기</a>에서 가능합니다.
      </p>
    </div>
    """
    return await _send_email(to, subject, html)
