# 학습 데이터셋 출처 문서
# AeroInspect — 20종 건물 하자 검출 AI

> 최종 갱신: 2026-04-22

---

## 데이터셋 총괄

| 데이터셋 | 이미지 수 | 모델 | 하자 코드 |
|---------|----------|------|----------|
| structural | 22,504 | M1 YOLO | A-02, A-03, B-03, B-04 |
| structural_crops | 2,991 | M1 ResNet | A-02, A-03, B-03, B-04 |
| surface | 8,957 | M2 YOLO | C-01~C-05 |
| surface_crops | 3,404 | M2 ResNet | C-01~C-05 |
| floor_window | 8,646 | M3 YOLO | D-03, D-04, E-01, E-02 |
| frames | 7,418 | M5 YOLO-seg | A-01, A-04 |
| thermal | 2,742 | M4 U-Net | B-01, B-02, D-01 |
| thermal_yolo | 1,262 | 열화상 YOLO | B-01, B-02, B-05 |
| normal | 5,361 | M6 PatchCore | 이상탐지 |
| **합계** | **63,285** | | |

---

## 하자코드 ↔ 데이터셋 ↔ 클래스 매핑

| 코드 | 하자명 | 데이터셋 | 클래스 |
|------|--------|---------|--------|
| A-01 | 벽·천장 수직·수평도 불량 | frames | 0~4 (wall/ceiling/floor seg) + LiDAR |
| A-02 | 균열 (구조 균열) | structural | crack (0) |
| A-03 | 균열 (마감 균열) | structural + structural_crops | crack (0) + crack_indicator |
| A-04 | 문·창호 틀 직각도 불량 | frames | 0~4 (door/window seg) + LiDAR |
| B-01 | 창호 단열 불량 | thermal + thermal_yolo | insulation, air_infiltration |
| B-02 | 벽체 단열 공백·탈락 | thermal + thermal_yolo | delamination |
| B-03 | 코킹 누락·불량 | structural | caulking_defect (2) |
| B-04 | 방수층 들뜸 / 누수 흔적 | structural | waterproof_defect (1) |
| B-05 | 창호 기밀 불량 | thermal_yolo | Moisture (1) |
| C-01 | 도배 이음매 불량 | surface + surface_crops | surface_defect_wall (0) / wallpaper_seam |
| C-02 | 도배지 기포·들뜸 | surface + surface_crops | surface_defect_wall (0) / wallpaper_bubble |
| C-03 | 도색 얼룩·붓자국 | surface + surface_crops | surface_defect_wall (0) / paint_stain |
| C-04 | 찍힘·스크래치 | surface + surface_crops | surface_defect_wall (0) / scratch |
| C-05 | 걸레받이 오염·파손 | surface + surface_crops | baseboard_defect (1) / baseboard_damage |
| D-01 | 바닥 난방 불량 | thermal + thermal_yolo | Hotspot / floor_heating |
| D-02 | 바닥재 들뜸 | floor_window | floor_defect (0) - tile delamination |
| D-03 | 바닥 오염·스크래치 | floor_window | floor_defect (0) |
| D-04 | 줄눈 불량 | floor_window | floor_defect (0) |
| E-01 | 창호 유리 스크래치·파손 | floor_window | glass_defect (1) |
| E-02 | 창틀·문틀 도장 불량 | floor_window | frame_defect (2) |

---

## 원본 데이터 출처 (gdrive_raw/)

### A 구조·기하

| 폴더 | 출처 | 라이선스 | 이미지 수 |
|------|------|---------|----------|
| A01_A04_wall_ceiling_seg | Roboflow Universe - Detecting floor, wall and ceiling (Priyansh Sethi) | CC BY 4.0 | 87 |
| A01_A04_interior_seg_v2 | Roboflow Universe - Detecting floor, wall and ceiling (label-for-yolov8obb) | CC BY 4.0 | 87 |
| A02_A03_crack_structural_finishing | Roboflow Universe - Crack Instance Segmentation v5 | CC BY 4.0 | 2,966 |
| A02_A03_crack_classification | Roboflow Universe - Crack dataset v1 (Folder Classification) | CC BY 4.0 | 9,856 |
| A02_crack_detection_2k | Roboflow Universe - crack detection (crack-7rsjb) | CC BY 4.0 | 2,394 |
| A02_B03_B04_building_wall_defects | Roboflow Universe - Building defect on walls v4 (Builddef2) | CC BY 4.0 | 1,416 |
| A02_B04_building_defect_bd3 | GitHub - BD3: Building Defects Detection Dataset (Praveenkottari) | CC BY-NC | 47 |
| A02_B04_s2ds_building_seg | GitHub - S2DS Dataset (CUHK) | Public | 1,486 |
| A02_B04_aihub_building_crack | AI Hub - 건물 균열 탐지드론 개발을 위한 이미지 (미래아이티컨소시엄) | CC BY-NC | 413,934 라벨 |
| A02_thermal_crack_subset | 열화상 균열 서브셋 (Thermal_Imaging_in_Building_v1에서 추출) | CC BY 4.0 | 26 |

### B 단열·방수

| 폴더 | 출처 | 라이선스 | 이미지 수 |
|------|------|---------|----------|
| B01_B02_thermal_inspection | Roboflow Universe - Thermal Imaging in Building v1 + Building Thermal Inspection v11 | CC BY 4.0 | 504 |
| B01_B02_B05_thermal_building_6cls | Roboflow Universe - Thermal images in building inspection (IDT) | CC BY 4.0 | 86 |
| B01_B02_D01_crack900_thermal_rgb_seg | GitHub - Crack900 Dataset (열화상 RGB + IR + 온도맵 + 세그마스크) | Academic | 5,484 |
| B03_welding_caulking_defects | GitHub - Welding Defect Detection (szbela87) | GPL-3.0 | 288 |
| B03_caulking_team | 팀 자체 수집 — 코킹 하자 분류 | 내부 | 6,674 |
| B03_caulking_bluesky | Roboflow Universe - Train pack AI (Bluesky Caulking AI) | CC BY 4.0 | 130 |
| B03_scratch_joint | Roboflow Universe - Scratch (yolo-0avst) | CC BY 4.0 | 596 |
| B03_B04_facade_defect_5cls | Roboflow Universe - Defect (Defects in Facade Building) | CC BY 4.0 | 1,003 |
| B03_B04_concrete_defect_1k | Roboflow Universe - Concrete defect detection (SHM) | CC BY 4.0 | 1,680 |
| B03_concrete_defect_v2 | Roboflow Universe - Concrete defect detection (Defect detection) | CC BY 4.0 | 6,806 |

### C 마감·표면

| 폴더 | 출처 | 라이선스 | 이미지 수 |
|------|------|---------|----------|
| C01_C05_wallpaper_surface_defects | Roboflow Universe - Wallpaper v2 Folder Classification + Papering Problems Multiclass | CC BY 4.0 | 7,301 |
| C03_C04_surface_dirt_scratch | Roboflow Universe - dirtvisionpro v1 | CC BY 4.0 | 2,121 |

### D 바닥·난방

| 폴더 | 출처 | 라이선스 | 이미지 수 |
|------|------|---------|----------|
| D01_thermal_defects_v2 | Roboflow Universe - Thermal Defects v2 (Solveview) | CC BY 4.0 | 672 |
| D02_floor_defect | Roboflow Universe - YOLO Floor Detection (yolo-m8a4j) | CC BY 4.0 | 130 |
| D02_tile_delam_v2 | Roboflow Universe - sep (joe-i4soa) — tile delamination 8cls | Public Domain | 4,610 |
| D03_D04_ceramic_tile_defects | Roboflow Universe - ceramic-tile-defects v14 (DATN) | CC BY 4.0 | 941 |
| D03_D04_magnetic_tile_defects | GitHub - Magnetic Tile Defect Datasets (abin24) | MIT | 2,690 |

### E 창호·유리

| 폴더 | 출처 | 라이선스 | 이미지 수 |
|------|------|---------|----------|
| E01_glass_defect_1k | Roboflow Universe - Glass Defect Detection v3 (capjamesg) | CC BY 4.0 | 1,728 |
| E01_glass_defect_agdd | GitHub - AGDD: Aircraft Glass Defect Dataset (core128) | Academic | 439 |
| E01_scratch_additional | Roboflow Universe - Scratch (yolo-0avst) | CC BY 4.0 | 287 |

### 기타

| 폴더 | 출처 | 라이선스 | 비고 |
|------|------|---------|------|
| weights | Colab 학습 가중치 (이전 세션) | 내부 | 3 .pt 파일 |

---

## 라이선스 요약

| 라이선스 | 데이터셋 수 | 사용 조건 |
|---------|-----------|----------|
| CC BY 4.0 | 22개 | 출처 표기 시 자유 사용 |
| CC BY-NC | 2개 (BD3, AI Hub) | 비상업적 사용만 허용 |
| Public Domain | 1개 (sep) | 제한 없음 |
| MIT | 1개 (Magnetic Tile) | 출처 표기 시 자유 사용 |
| GPL-3.0 | 1개 (Welding) | 파생물 GPL 적용 |
| Academic | 2개 (Crack900, AGDD) | 학술 목적만 허용 |
| 내부 | 1개 (코킹 팀 수집) | 팀 내부 사용 |
