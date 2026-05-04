# ML 학습 가이드 — 20종 하자 검출 파이프라인

드론 기반 건축물 하자 검출을 위한 6-Model + Geometric 파이프라인 학습 가이드.

---

## 목차

1. [환경 셋업](#1-환경-셋업)
2. [데이터 준비](#2-데이터-준비)
3. [모델별 학습](#3-모델별-학습)
4. [ONNX 변환](#4-onnx-변환)
5. [평가 및 벤치마크](#5-평가-및-벤치마크)
6. [배포](#6-배포)
7. [파이프라인 아키텍처](#7-파이프라인-아키텍처)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 환경 셋업

### 필수 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| Python | 3.10+ | 3.11 |
| GPU | RTX 3060 (12GB) | RTX 4090 / A100 |
| RAM | 16GB | 32GB |
| 저장공간 | 50GB | 200GB |

### 설치

```bash
# 프로젝트 루트에서
cd backend

# 가상환경 (이미 있으면 활성화만)
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# 기본 의존성
pip install -r requirements.txt

# 학습 추가 의존성
pip install ultralytics>=8.3.0     # YOLOv8
pip install segmentation-models-pytorch  # U-Net
pip install anomalib               # PatchCore
pip install albumentations         # 데이터 증강
pip install onnx onnxruntime-gpu   # ONNX 변환/추론
pip install gdown                  # Google Drive 다운로드

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 2. 데이터 준비

### 2-1. Google Drive에서 다운로드

```bash
cd backend/training

# 방법 A: gdown 자동 다운로드
python download_and_organize.py --url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"

# 방법 B: 이미 로컬에 다운로드한 경우
python download_and_organize.py --local ./gdrive_raw
```

스크립트가 자동으로:
1. 이미지/라벨 파일을 분석
2. 20종 하자 카테고리에 자동 분류
3. train(70%) / val(15%) / test(15%) 분할
4. `datasets/` 폴더에 정리

### 2-2. 데이터셋 구조

학습 데이터는 아래 구조로 정리되어야 합니다:

```
training/datasets/
│
├── structural/              # M1: 구조·방수 (YOLO 포맷)
│   ├── images/
│   │   ├── train/  *.jpg
│   │   ├── val/    *.jpg
│   │   └── test/   *.jpg
│   └── labels/
│       ├── train/  *.txt    # YOLO: class x_center y_center width height
│       ├── val/    *.txt
│       └── test/   *.txt
│
├── structural_crops/        # M1-ResNet: 균열 분류 (ImageFolder)
│   ├── train/
│   │   ├── crack_structural/ *.jpg
│   │   └── crack_finishing/  *.jpg
│   ├── val/   ...
│   └── test/  ...
│
├── surface/                 # M2: 마감·표면 (YOLO 포맷)
│   ├── images/ ...
│   └── labels/ ...
│
├── surface_crops/           # M2-ResNet: 표면 분류 (ImageFolder)
│   ├── train/
│   │   ├── wallpaper_seam/
│   │   ├── wallpaper_bubble/
│   │   ├── paint_stain/
│   │   ├── scratch/
│   │   └── baseboard_damage/
│   ├── val/   ...
│   └── test/  ...
│
├── floor_window/            # M3: 바닥·창호 (YOLO 포맷)
├── floor_window_crops/      # M3-ResNet: 유형 분류 (ImageFolder)
│
├── thermal/                 # M4: 열화상 (커스텀)
│   ├── thermal_maps/ *.npy  # float32 온도맵
│   ├── masks/        *_mask.npy  # uint8 세그멘테이션
│   └── rgb/          *.jpg  # 페어 RGB 이미지
│
├── frames/                  # M5: 기하학 (YOLO-seg 포맷)
│   ├── images/ ...
│   └── labels/ ...          # polygon 세그멘테이션 라벨
│
└── normal/                  # M6: PatchCore (정상 이미지만)
    ├── good/       *.jpg    # 하자 없는 정상 표면
    └── defective/  *.jpg    # 검증용 하자 이미지 (선택)
```

### 2-3. YOLO 라벨 포맷

각 이미지에 대응하는 `.txt` 파일:
```
# class_id  x_center  y_center  width  height  (정규화 0~1)
0 0.453 0.621 0.120 0.085
1 0.712 0.334 0.056 0.044
```

### 2-4. 어노테이션 도구

라벨이 없는 이미지는 아래 도구로 어노테이션:
- **CVAT** (Self-hosted): bbox + polygon, 팀 협업 가능
- **Roboflow** (Web): 자동 증강, YOLO export 지원
- **Label Studio** (Self-hosted): 다양한 태스크 지원

---

## 3. 모델별 학습

### 학습 순서 (권장)

```
M1 (구조·방수) → M2 (마감·표면) → M3 (바닥·창호) → M4 (열화상) → M5 (기하학) → M6 (PatchCore)
```

M1~M3가 가장 직관적이고 데이터가 풍부하므로 먼저 진행.

### M1: 구조·방수 하자 (균열, 코킹, 방수)

```bash
cd backend/training

# Stage 1: YOLO 검출 학습
python train_m1_yolo_structural.py

# Stage 2: ResNet50 균열 분류 학습 (구조균열 vs 마감균열)
python train_m1_resnet_crack.py
```

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv8m + ResNet50 |
| 클래스 | crack, caulking_defect, waterproof_defect |
| 에폭 | 200 (YOLO) + 50 (ResNet) |
| 배치 | 16 |
| 소요시간 | ~10시간 (YOLO, RTX 4090 기준) |
| 출력 | `models_weights/m1_yolo_structural.onnx`, `m1_resnet_crack_classifier.onnx` |

### M2: 마감·표면 하자 (도배, 도색, 스크래치, 걸레받이)

```bash
python train_m2_yolo_surface.py
python train_m2_resnet_surface.py
```

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv8m + ResNet50 |
| 클래스 (YOLO) | surface_defect_wall, baseboard_defect |
| 클래스 (ResNet) | wallpaper_seam, wallpaper_bubble, paint_stain, scratch, baseboard_damage |
| 에폭 | 150 (YOLO) + 80 (ResNet) |
| 출력 | `m2_yolo_surface.onnx`, `m2_resnet_surface_classifier.onnx` |

### M3: 바닥·창호 하자

```bash
python train_m3_yolo_floor_window.py
python train_m3_resnet_floor_window.py
```

### M4: 열화상 단열 하자 (U-Net)

```bash
python train_m4_thermal_unet.py
```

| 항목 | 값 |
|------|-----|
| 모델 | U-Net (EfficientNet-B3 backbone) |
| 입력 | 열화상 온도맵 256×192 |
| 클래스 | background, window_insulation, wall_insulation, window_airtight, floor_heating |
| 에폭 | 100 |
| 출력 | `m4_unet_thermal_insulation.onnx` |

### M5: 기하학 프레임 세그멘테이션

```bash
python train_m5_frame_seg.py
```

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv8m-seg (인스턴스 세그멘테이션) |
| 클래스 | wall_edge, ceiling_edge, door_frame, window_frame |
| 에폭 | 200 |
| 출력 | `m5_yolo_seg_frames.onnx` |

### M6: PatchCore 비지도 학습

```bash
python train_m6_patchcore.py
```

| 항목 | 값 |
|------|-----|
| 모델 | PatchCore (WideResNet50 backbone) |
| 학습 데이터 | 정상 표면 이미지만 (라벨 불필요) |
| 소요시간 | ~2시간 |
| 출력 | `m6_patchcore_surface.onnx` |

---

## 4. ONNX 변환

각 학습 스크립트는 학습 완료 후 자동으로 ONNX 변환을 수행합니다.

이미 `.pt` 체크포인트가 있는 경우 일괄 변환:

```bash
# 전체 변환
python export_to_onnx.py

# 특정 모델만
python export_to_onnx.py --model m1
python export_to_onnx.py --model m2
```

변환된 파일은 `backend/models_weights/`에 저장됩니다:

```
models_weights/
├── m1_yolo_structural.onnx
├── m1_resnet_crack_classifier.onnx
├── m2_yolo_surface.onnx
├── m2_resnet_surface_classifier.onnx
├── m3_yolo_floor_window.onnx
├── m3_resnet_floor_window_classifier.onnx
├── m4_unet_thermal_insulation.onnx
├── m5_yolo_seg_frames.onnx
├── m6_patchcore_surface.onnx
└── thermal_rgb_homography.json
```

---

## 5. 평가 및 벤치마크

### 모델 성능 평가

```bash
# 전체 모델 평가
python eval/evaluate_all.py

# 특정 모델만
python eval/evaluate_all.py --model m1
```

결과는 `eval/evaluation_results.json`에 저장됩니다.

### 추론 속도 벤치마크

```bash
python eval/benchmark.py
```

### 목표 지표

| 모델 | 지표 | 목표 |
|------|------|------|
| M1 YOLO | Recall@IoU=0.5 | ≥ 0.95 |
| M1 ResNet | Accuracy | ≥ 0.90 |
| M2 YOLO | Recall@IoU=0.5 | ≥ 0.93 |
| M2 ResNet | Accuracy | ≥ 0.88 |
| M3 YOLO | Recall@IoU=0.5 | ≥ 0.93 |
| M4 U-Net | Mean Dice | ≥ 0.90 |
| M5+G1 | Angular MAE | < 0.05° |
| 전체 지연 | Tier1 (T4) | < 60ms |

---

## 6. 배포

### 6-1. 파이프라인 활성화

`backend/.env`에 추가:

```env
USE_20DEFECT_PIPELINE=true
```

### 6-2. 서버 시작

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버 시작 시 `models_weights/` 내 ONNX 파일을 자동 로드합니다.
없는 모델은 건너뛰고 가용한 모델만으로 동작합니다 (graceful degradation).

### 6-3. 헬스 체크

```
GET /health
```

응답에서 각 모델의 로드 상태를 확인할 수 있습니다.

### 6-4. 추론 API

```bash
# 단일 이미지 검출
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@test_image.jpg"

# WebSocket 실시간 스트리밍
# ws://localhost:8000/api/v1/ws/stream
# → 바이너리 JPEG 전송 → detection_20 결과 수신
```

---

## 7. 파이프라인 아키텍처

### 20종 하자 코드 체계 (A-01 ~ E-02)

| 코드 | 하자명 | 모델 | 센서 |
|------|--------|------|------|
| A-01 | 벽·천장 수직·수평도 불량 | M5+G1 | RGB+IMU |
| A-02 | 균열 (구조 균열) | M1 YOLO→ResNet | RGB |
| A-03 | 균열 (마감 균열) | M1 YOLO→ResNet | RGB |
| A-04 | 문·창호 틀 직각도 불량 | M5+G1 | RGB+IMU |
| B-01 | 창호 단열 불량 | M4 U-Net | Thermal+RGB |
| B-02 | 벽체 단열 공백·탈락 | M4 U-Net | Thermal+RGB |
| B-03 | 코킹 누락·불량 | M1 YOLO | RGB |
| B-04 | 방수층 들뜸 / 누수 흔적 | M1 YOLO | RGB |
| B-05 | 창호 기밀 불량 | M4 U-Net | Thermal |
| C-01 | 도배 이음매 불량 | M2 YOLO→ResNet | RGB |
| C-02 | 도배지 기포·들뜸 | M2 YOLO→ResNet | RGB |
| C-03 | 도색 얼룩·붓자국 | M2 YOLO→ResNet | RGB |
| C-04 | 찍힘·스크래치 | M2 YOLO→ResNet | RGB |
| C-05 | 걸레받이 오염·파손 | M2 YOLO→ResNet | RGB |
| D-01 | 바닥 난방 불량 | M4 U-Net | Thermal |
| D-02 | 바닥재 들뜸 | M4+M3 | Thermal+RGB |
| D-03 | 바닥 오염·스크래치 | M3 YOLO→ResNet | RGB |
| D-04 | 줄눈 불량 | M3 YOLO→ResNet | RGB |
| E-01 | 창호 유리 스크래치·파손 | M3 YOLO→ResNet | RGB |
| E-02 | 창틀·문틀 도장 불량 | M3 YOLO→ResNet | RGB |

### 2-Stage 구조 (YOLO → ResNet)

```
RGB Frame
  ↓
YOLO (Stage 1) → "어디에" 하자가 있는가 (bbox 검출)
  ↓ ROI 크롭
ResNet (Stage 2) → "무슨" 하자인가 (유형 정밀 분류)
  ↓
severity_mapper → 코드(A-01~E-02) + 심각도(HIGH/MED/LOW) 매핑
  ↓
DetectionResult20
```

### 계층적 실행 (Tiered Execution)

실시간 스트리밍에서 GPU 부하 분산:

```
Tier 1 (매 3프레임): M1 + M2      → ~50ms  (HIGH severity 즉시 검출)
Tier 2 (매 6프레임): M3 + M5+G1   → ~55ms  (MED/LOW + 기하학)
Tier 3 (매 9프레임): M4 + M6      → ~70ms  (열화상 + 앙상블)
```

---

## 8. 트러블슈팅

### CUDA 메모리 부족

```bash
# 배치 크기 줄이기
# train_m1_yolo_structural.py 내 BATCH = 16 → 8

# 또는 더 작은 모델 사용
# yolov8m.pt → yolov8s.pt (파라미터 절반)
```

### ONNX 변환 실패

```bash
# opset 버전 낮추기
model.export(format="onnx", opset=13)  # 17 → 13

# dynamic axes 비활성화
model.export(format="onnx", dynamic=False)
```

### gdown 다운로드 실패

```bash
# rate limit → 10~30분 대기 후 재시도

# 권한 문제 → Drive에서 "링크가 있는 모든 사용자" 설정

# 대안: 브라우저에서 ZIP 다운로드 후
python download_and_organize.py --local ./path/to/downloaded/folder
```

### 학습이 수렴하지 않는 경우

1. 데이터 품질 확인: 라벨이 정확한지, 이미지가 깨지지 않았는지
2. 학습률 조정: `lr0`을 1e-5로 낮춰보기
3. 데이터 증강 줄이기: `mosaic=0.5`, `mixup=0.0`
4. 클래스 불균형: `class_weights` 조정
