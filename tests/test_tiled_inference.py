# =============================================
# tests/test_tiled_inference.py
# SAHI 타일 분할 추론 단위 테스트
# - 타일 생성 좌표 검증
# - 저해상도 fallback
# - cross-tile NMS
# =============================================

from app.services.tiled_inference import generate_tiles, _cross_tile_nms


class TestGenerateTiles:
    """타일 좌표 생성 테스트."""

    def test_small_image_single_tile(self):
        """640x640 이하 이미지는 타일 1개."""
        tiles = generate_tiles(480, 640, tile_size=640)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 640, 480)

    def test_large_image_multiple_tiles(self):
        """1920x1080 이미지는 여러 타일."""
        tiles = generate_tiles(1080, 1920, tile_size=640, overlap_ratio=0.2)
        assert len(tiles) > 1
        # 모든 타일이 이미지 범위 내
        for x1, y1, x2, y2 in tiles:
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 1920 and y2 <= 1080
            assert x2 - x1 <= 640 and y2 - y1 <= 640

    def test_4k_tile_count(self):
        """4K(3840x2160) 이미지의 타일 수 검증."""
        tiles = generate_tiles(2160, 3840, tile_size=640, overlap_ratio=0.2)
        # 대략 (3840/512)*(2160/512) ≈ 7.5*4.2 ≈ 32 타일
        assert len(tiles) >= 15  # 최소 15개
        assert len(tiles) <= 50  # 최대 50개

    def test_overlap_coverage(self):
        """타일들이 전체 이미지를 빈틈 없이 커버."""
        tiles = generate_tiles(1000, 1000, tile_size=640, overlap_ratio=0.2)
        # 모든 픽셀이 최소 1개 타일에 포함되는지 확인 (간소화: 코너 확인)
        corners = [(0, 0), (999, 0), (0, 999), (999, 999), (500, 500)]
        for px, py in corners:
            covered = any(
                x1 <= px < x2 and y1 <= py < y2
                for x1, y1, x2, y2 in tiles
            )
            assert covered, f"Pixel ({px},{py}) not covered"


class TestCrossTileNMS:
    """cross-tile NMS 테스트."""

    def test_duplicate_removed(self):
        """동일 클래스 동일 위치 → 높은 conf 유지."""
        dets = [
            {"class": "crack", "conf": 0.8, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "crack", "conf": 0.6, "bbox_xyxy": [12, 12, 52, 52]},  # 겹침
        ]
        result = _cross_tile_nms(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["conf"] == 0.8

    def test_different_class_kept(self):
        """다른 클래스는 겹쳐도 둘 다 유지."""
        dets = [
            {"class": "crack", "conf": 0.8, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "moisture", "conf": 0.7, "bbox_xyxy": [10, 10, 50, 50]},
        ]
        result = _cross_tile_nms(dets, iou_threshold=0.3)
        assert len(result) == 2

    def test_no_overlap_kept(self):
        """겹치지 않는 검출은 모두 유지."""
        dets = [
            {"class": "crack", "conf": 0.8, "bbox_xyxy": [0, 0, 50, 50]},
            {"class": "crack", "conf": 0.7, "bbox_xyxy": [200, 200, 300, 300]},
        ]
        result = _cross_tile_nms(dets, iou_threshold=0.3)
        assert len(result) == 2

    def test_empty_input(self):
        assert _cross_tile_nms([], iou_threshold=0.5) == []

    def test_single_input(self):
        dets = [{"class": "crack", "conf": 0.5, "bbox_xyxy": [0, 0, 50, 50]}]
        assert len(_cross_tile_nms(dets)) == 1
