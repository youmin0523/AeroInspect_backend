# 로컬 GPU 파인튜닝 가이드 — 타일(바닥) 하자 오분류 해결

> 작성 2026-06-12. 대상: RTX 5070 Laptop 8GB 로컬 학습. **Gemini 최소 사용.**

---

## 1. 진단 (왜 타일이 틀리게 나오나) — GPU 실측 근거

사용자 실내 바닥타일 영상으로 모델 raw 출력 직접 측정:

```
t=0-1s  M4ctx=floor(0.4)  M3=floor_defect(0.35)   ← 초반 정상
t=2-5s  M4ctx=floor→window→[빈값]  M3=glass_defect(0.87) ← 타일을 '유리 결함'으로 오인
```

**원인 사슬:**
1. M3은 타일 결함을 **검출은 함**(박스 뜸) — 모델이 봄.
2. 그런데 **floor_defect ↔ glass_defect 혼동**(균열·줄눈 형태가 유리 균열과 동일) → "glass_defect"로 오라벨.
3. 게이트가 이걸 교정하게 돼 있음(M4 'floor' 컨텍스트 → glass를 floor로 `context_relabel`).
4. **그런데 M4 Context가 약함(mAP 0.527)** → 중반부터 floor→window→빈값으로 무너짐 → 게이트 교정/차단 불발 → 오라벨 통과 + M1/M2 오탐(방수·걸레받이)이 대신 노출.

**결론: 데이터 부족이 아니라 M4 Context가 약해 게이트가 작동 못 함.** floor_window 데이터에 타일 박리(D-02)·줄눈(D-04)은 이미 있음(`floor_defect`).

---

## 2. 고칠 우선순위 (레버리지 순)

| 순위 | 작업 | 효과 | 데이터 | Gemini |
|---|---|---|---|---|
| **1** | **M4 Context 재학습** | 게이트 권위 복구 → floor↔glass↔frame 혼동 + 엉뚱표면 오탐 **전반 해결** | datasets/m4_context (Roboflow+floor_window, ~10,500장) | 0 |
| 2 | M3(floor/glass/frame) 재학습 | floor↔glass 직접 분리 | floor_window 8,646장 | 0 |
| 3 | 신규 footage 추가 | in-domain 보강 | 추가 촬영 + 본 영상 | 0 (수작업 라벨) |

→ **M4부터.** 가장 싸고(기존 데이터) 효과 범위가 넓음.

---

## 3. 환경 (이미 거의 준비됨)

- 학습 스크립트가 **이미 RTX 5070 8GB용**: `train_m4_context_yolo.py`(yolov8m, batch=4, imgsz=960), `train_m3_yolo_floor_window.py`(BATCH=4 주석에 명시).
- **CUDA torch 필요**(현재 venv는 torch CPU). RTX 5070=Blackwell(sm_120) → CUDA 12.8 빌드:
  ```
  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
- ultralytics 8.4.61 설치됨.

## 4. 데이터 (gdrive에서 로컬로)

데이터셋은 Google Drive에 있고 로컬엔 없음. `download_and_organize.py`로 가져옴:
```
cd training
python download_and_organize.py --url "<gdrive 데이터셋 폴더 URL>"
# 또는 이미 받았으면: python download_and_organize.py --local ./gdrive_raw
```
→ `datasets/m4_context`, `datasets/floor_window` 등 구성됨(스크립트가 기대하는 경로).

## 5. 효율적 라벨링 (Gemini 최소)

- **M4·M3 bulk: 기존 라벨 그대로 재사용**(추가 라벨 0).
- **신규 footage(이 영상 등) 보강**:
  - 본 영상은 전부 바닥타일 → 결함=`floor_defect` 단일. 프레임 추출됨: `training/tile_finetune/frames/`.
  - 라벨링: ① 수작업 소량(타일 결함 육안 명확) **또는** ② VLM **Flash**(Pro의 1/10)로 seed 자동라벨 → 사람 검수 → 1차 학습 모델로 **pseudo-label 확장**(무료). **Pro 대량 라벨링 금지.**
  - ⚠️ 주의: M3 raw 박스를 그대로 floor_defect로 쓰면 M3의 오탐도 섞임 → 반드시 사람 검수.

## 6. 학습 → 내보내기 → 배포

```
cd training
# (1) M4 재학습 — 우선
python -u train_m4_context_yolo.py        # → models_weights/m4_yolo_context_elements.onnx (스크립트가 ONNX export 포함 시)
#     아니면: python export_to_onnx.py --model m4

# (2) M3 재학습(선택)
python -u train_m3_yolo_floor_window.py
python export_to_onnx.py --model m3

# (3) GCP GPU VM 배포: 새 .onnx 를 models_weights/ 에 반영 후
#     VM에서 git pull(또는 scp .onnx) → 컨테이너 재빌드/재시작
```

### 8GB VRAM 팁
- imgsz=960·batch=4가 한계선. OOM 나면 batch=2 또는 imgsz=768.
- `multi_scale` 켜져 있으면 메모리 변동 → OOM 시 끄기.

## 7. 검증 (재학습 후)
- 본 영상으로 `python -c "..."` raw 재측정: M4가 'floor' 안정적으로 주는지 + M3 glass 오라벨 사라지는지.
- 전 파이프라인: `pipeline20.detect(frame)` → floor_defect로 정상 라벨되는지.
- 비교 기준: VLM(gemini-3.1-pro)이 "줄눈 불량" 정확히 식별 → 그게 정답.

---

## 핵심 요약
- 문제 = **M4 약함**(데이터 부족 아님). M4 재학습이 1순위·최대 레버·Gemini 0원.
- 인프라(스크립트·GPU)는 준비됨. 막힌 건 ① CUDA torch 설치 ② gdrive 데이터 다운로드.
- 신규 라벨은 수작업/Flash seed로 최소화. Pro는 검수만.
