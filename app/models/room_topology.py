# =============================================
# app/models/room_topology.py
# 역할: 룸 토폴로지 그래프 ORM 모델
#       - 한 미션에서 occupancy 분리 결과로 얻은 거실/방1/방2 등의 노드와
#         도어웨이 엣지를 JSONB 두 컬럼에 저장
#       - mission_orchestrator의 ROOM_TRANSITION 단계에서 재참조
# 테이블명: room_topologies
# =============================================

import uuid

from sqlalchemy import (
    Column, DateTime, ForeignKey, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class RoomTopology(Base):
    """
    룸 토폴로지 1건 = 1 미션의 공간 그래프.

    nodes_json 형식 예:
      [{"idx": 0, "name": "room_0", "polygon": [[x,y], ...], "area": 12.4}, ...]

    edges_json 형식 예:
      [{"from": 0, "to": 1, "doorway_center": [x,y],
        "doorway_width": 0.85, "approach_yaw": 1.57}, ...]
    """

    __tablename__ = "room_topologies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── FK ──────────────────────────────────
    mission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("mission_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── 그래프 ──────────────────────────────
    nodes_json = Column(JSONB, nullable=False, default=list)
    edges_json = Column(JSONB, nullable=False, default=list)

    # ── 감사 ────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_room_topology_mission", "mission_id", unique=True),
    )

    def __repr__(self):
        n = len(self.nodes_json) if self.nodes_json else 0
        e = len(self.edges_json) if self.edges_json else 0
        return f"<RoomTopology mission={self.mission_id} nodes={n} edges={e}>"
