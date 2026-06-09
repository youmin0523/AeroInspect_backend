# =============================================
# app/models/defect.py
# 역할: 하자 탐지 로그 ORM 모델 정의
#       - 드론이 탐지한 모든 하자 정보를 저장하는 핵심 테이블
#       - 위치(LiDAR 3D 좌표), 이미지 크롭, 열화상 온도, 심각도 포함
#       - PostgreSQL JSONB 타입으로 원시 YOLO 탐지 결과 저장
# 테이블명: defect_logs
# =============================================

import uuid

from sqlalchemy import (
    Column, String, Float, Text, BigInteger,
    DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class DefectLog(Base):
    """
    하자 탐지 로그 테이블.
    드론 비행 중 AI가 탐지한 하자 1건 = 1 레코드.
    """
    __tablename__ = "defect_logs"

    # ── 기본 키 ──────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 현장 연결 (경향보고서 연계) ───────────
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=True, comment="연결 현장 ID")

    # ── 하자 분류 (레거시 A-E taxonomy) ───────
    # area: A(구조) / B(단열·방수) / C(마감재) / D(바닥) / E(창호)
    # 신규 3-모델 중 레거시 매핑 없는 케이스는 NULL 허용
    area = Column(String(1), nullable=True, comment="하자 영역 코드 (A-E, 매핑 없으면 NULL)")
    category_code = Column(String(10), nullable=True, comment="하자 카테고리 코드 (예: A-01)")
    defect_type = Column(String(100), nullable=True, comment="하자 유형명 (한글)")

    # ── 신규 3-모델 파이프라인 분류 ───────────
    # defect_source: 탐지한 모델 종류
    defect_source = Column(
        SAEnum("yolo_thermal", "yolo_delam", "wallpaper", name="defect_source_enum"),
        nullable=True,
        comment="탐지 모델 (yolo_thermal | yolo_delam | wallpaper)",
    )
    # defect_class: 모델 내부 클래스명 (예: 'Crack', 'good')
    # ⚠️ 'good'은 벽지 '터짐(Burst)'임 — "정상" 아님
    defect_class = Column(String(50), nullable=True, comment="모델 내부 클래스명")
    defect_class_display_en = Column(String(80), nullable=True, comment="영문 표시명 (예: 'Burst')")
    defect_class_display_ko = Column(String(80), nullable=True, comment="한글 표시명 (예: '터짐')")

    # ── 심각도 ───────────────────────────────
    # HIGH: 구조·안전·방수 직결 / MED: 기능 저하 / LOW: 마감 미관
    severity = Column(
        SAEnum("HIGH", "MED", "LOW", name="severity_enum"),
        nullable=False,
        comment="심각도 등급"
    )

    # ── AI 탐지 결과 ──────────────────────────
    confidence = Column(Float, nullable=False, comment="AI 탐지 신뢰도 (0.0~1.0)")
    # 바운딩 박스 (프레임 내 정규화 좌표 0.0~1.0)
    bbox_x = Column(Float, comment="바운딩 박스 중심 X (정규화)")
    bbox_y = Column(Float, comment="바운딩 박스 중심 Y (정규화)")
    bbox_w = Column(Float, comment="바운딩 박스 너비 (정규화)")
    bbox_h = Column(Float, comment="바운딩 박스 높이 (정규화)")

    # ── 3D 공간 좌표 (TF-Luna LiDAR) ──────────
    # 드론 TF(Transform) → 월드 좌표계(ENU) 변환 후 저장
    lidar_x = Column(Float, comment="월드 좌표 X (m)")
    lidar_y = Column(Float, comment="월드 좌표 Y (m)")
    lidar_z = Column(Float, comment="월드 좌표 Z / 고도 (m)")

    # ── 이미지 데이터 ─────────────────────────
    # image_crop (deprecated): Base64 인코딩된 JPEG. DB 용량 이슈로 파일 저장 방식으로 전환 중.
    # image_crop_path: 파일시스템 상대 경로 (예: "defects/2026-04-21/xxx.jpg"). /uploads/ StaticFiles로 서빙.
    # 신규 레코드는 image_crop_path만 채움. image_crop은 과거 데이터 호환용으로만 유지.
    image_crop = Column(Text, comment="[DEPRECATED] Base64 JPEG. 신규는 image_crop_path 사용.")
    image_crop_path = Column(String(255), comment="하자 크롭 이미지 상대 경로 (uploads/ 기준)")

    # ── 열화상 데이터 ─────────────────────────
    thermal_max = Column(Float, comment="하자 ROI 최고 온도 (°C)")
    thermal_min = Column(Float, comment="하자 ROI 최저 온도 (°C)")
    thermal_avg = Column(Float, comment="하자 ROI 평균 온도 (°C)")

    # ── 메타데이터 ────────────────────────────
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="탐지 시각 (UTC)"
    )
    frame_id = Column(BigInteger, comment="탐지된 비디오 프레임 번호")

    # 원시 YOLO 탐지 결과 전체 저장 (디버깅·재분석용)
    raw_payload = Column(JSONB, comment="YOLO 원시 탐지 결과 JSON")

    # ── 20종 파이프라인 확장 컬럼 ────────────
    # 기하학 하자 (A-01, A-04): 수직수평·직각도 편차
    deviation_degrees = Column(Float, nullable=True, comment="수직수평/직각도 편차 (도)")
    deviation_mm_per_m = Column(Float, nullable=True, comment="편차 mm/m 환산")

    # 단열 하자 (B-01, B-02, B-05, D-01): 온도 편차
    delta_temperature = Column(Float, nullable=True, comment="주변 대비 온도차 (°C)")

    # 앙상블 부스팅 여부
    ensemble_boosted = Column(String(5), nullable=True, default=None, comment="PatchCore 앙상블 승격 (true/false)")

    # ── 추적·시간 필터 확장 컬럼 ─────────────
    # ByteTrack 객체 추적 ID (동일 track_id = 동일 물리 하자)
    track_id = Column(BigInteger, nullable=True, comment="ByteTrack 추적 ID (프레임 간 동일 하자 식별)")
    # 시간 누적 신뢰도 (TemporalFilter Noisy-OR 결과)
    accumulated_conf = Column(Float, nullable=True, comment="시간 누적 신뢰도 (Noisy-OR)")
    # 실행 계층 (1=M1+M2, 2=+M3+M5, 3=+M4+M6)
    tier_executed = Column(BigInteger, nullable=True, comment="실행 Tier (1/2/3)")

    # ── 검수 메타데이터 (Audit) ─────────────────
    # 입주자 분쟁 시 책임 추적용. AI 탐지 결과를 사람이 승인/반려/오탐 플래그한 이력.
    # review_status: pending=미검수 / approved=정탐 확인 / rejected=오탐(취소) / flagged_false_positive=Active Learning 큐 적재
    review_status = Column(
        SAEnum("pending", "approved", "rejected", "flagged_false_positive", name="defect_review_status_enum"),
        nullable=False,
        server_default="pending",
        comment="검수 상태",
    )
    reviewed_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, comment="검수자 ID")
    reviewed_at = Column(DateTime(timezone=True), nullable=True, comment="검수 시각 (UTC)")
    review_note = Column(Text, nullable=True, comment="검수 사유/메모 (반려 시 필수)")

    # ── 탐지 모델 출처 (감사·디버깅용) ────────
    # 어떤 모델이 이 하자를 탐지했는지. M4 컨텍스트 mAP 0.59 등 약한 모델 결과 추적 가능.
    detection_model_id = Column(String(40), nullable=True, comment="탐지 모델 식별자 (M1_YOLO / M2_YOLO / M3_YOLO / M4_CONTEXT / M5_SEG / furniture_aware / wallpaper / patchcore)")

    # ── GPS WGS84 (현장 정확 위치) ────────────
    # LiDAR 월드 좌표(lidar_x/y/z)와 별개. 평면도 위 핀 표시, 외부 지도 연동, 다른 점검 현장과의 위치 식별용.
    gps_lat = Column(Float, nullable=True, comment="GPS 위도 (WGS84)")
    gps_lon = Column(Float, nullable=True, comment="GPS 경도 (WGS84)")
    gps_alt = Column(Float, nullable=True, comment="GPS 고도 (m, MSL)")

    # ── 인덱스 ───────────────────────────────
    # 필터링 쿼리 최적화: 심각도+시간 / 영역+시간 / 프레임 / 검수 워크플로우
    __table_args__ = (
        Index("idx_defect_severity_ts", "severity", timestamp.desc()),
        Index("idx_defect_area_ts", "area", timestamp.desc()),
        Index("idx_defect_frame", "frame_id"),
        Index("idx_defect_review_status", "review_status", timestamp.desc()),
        Index("idx_defect_reviewer", "reviewed_by_user_id", "reviewed_at"),
        # 조직 스코프 조회(site_id IN (...)) + 최신순 정렬을 한 번에 커버.
        # defect_logs 는 최대 볼륨 테이블이라 site_id 무인덱스 시 seq-scan 비용이 큼.
        Index("idx_defect_site_ts", "site_id", timestamp.desc()),
    )

    def __repr__(self):
        return (
            f"<DefectLog id={self.id} "
            f"code={self.category_code} "
            f"severity={self.severity} "
            f"ts={self.timestamp}>"
        )
