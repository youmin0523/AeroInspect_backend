# =============================================
# app/core/security.py
# 역할: 비밀번호 해싱 및 검증 유틸리티
#       - bcrypt 기반 단방향 해시
#       - bcrypt 4.x + passlib 1.7.4 호환성 문제 회피를 위해 bcrypt 직접 사용
# 사용: from app.core.security import hash_password, verify_password
# =============================================

import bcrypt


def hash_password(plain_password: str) -> str:
    """
    평문 비밀번호를 bcrypt 해시 문자열로 변환.
    결과 길이: 60자 ($2b$12$saltsalthash...)
    """
    password_bytes = plain_password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    사용자 입력 평문과 저장된 해시를 비교.
    bcrypt.checkpw 는 내부적으로 상수 시간 비교 수행.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """
    저장된 해시가 현재 정책 대비 구버전이면 True.
    bcrypt 직접 사용 시: rounds가 12 미만이면 재해싱 권장.
    """
    try:
        # $2b$12$... 에서 rounds 추출
        parts = password_hash.split("$")
        if len(parts) >= 3:
            rounds = int(parts[2])
            return rounds < 12
    except (ValueError, IndexError):
        pass
    return False
