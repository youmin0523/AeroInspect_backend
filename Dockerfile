# =============================================
# Dockerfile
# 역할: 백엔드 FastAPI 서버 컨테이너 빌드 설정
# 빌드: docker build -t aeroinspect-backend .
# 실행: docker run -p 8000:8000 --env-file .env aeroinspect-backend
# 주의: --workers 1 고정 (WebSocket 브로드캐스트 단일 프로세스 필요)
# =============================================

FROM python:3.11-slim

# OpenCV 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 저장 디렉토리 생성
RUN mkdir -p models_weights captured_frames

EXPOSE 8000

# workers=1: WebSocket ConnectionManager가 단일 프로세스 싱글톤이어야 함
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
