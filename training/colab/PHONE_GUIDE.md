# 핸드폰으로 학습 진행 가이드 (저녁 7시 이후)

## 📋 학습 종료 시점 (~17:00~18:00) 확인 후 작업

### Step 1: 현재 학습 결과 확인
각 코랩 노트북에서 **마지막 셀 (mAP 출력)** 확인:
- M5v2 v2 (계정 A): mAP50 결과
- M2 (계정 B): mAP50 결과
- M1-aggressive (계정 C): mAP50 결과

`mAP50 >= 0.9`이면 다음 단계 불필요 (그대로 사용).

### Step 2: 0.9 미달 모델만 재학습 노트북 실행

#### 우선순위 (시간 효율):
1. **M5v2 v2가 0.9 미달** → `m5v3_refine_retrain.ipynb` 실행 (계정 A)
2. **M2가 0.87 미달** → `m2v2_refine_retrain.ipynb` 실행 (계정 B)
3. **M4 Context가 0.9 미달** → `m4v2_refine_retrain.ipynb` 실행 (계정 B 또는 로컬)
4. **M1-aggressive가 0.9 미달** → `m1v3_refine_retrain.ipynb` 실행 (계정 C, A100 권장)

---

## 📱 핸드폰에서 실행 (간단):

### 코랩 앱 또는 모바일 브라우저:
1. https://colab.research.google.com 접속
2. 해당 노트북 열기
3. **셀 차례로 ▶ 클릭** (또는 메뉴 → 모두 실행)
4. **각 셀 1회 클릭만** 하면 자동 진행
5. 90분 inactive 끊김 방지: **30~60분에 한 번씩 코랩 탭 다시 보기 (셀 클릭 X, 화면만 보면 됨)**

---

## 🔧 주의사항:

### Drive 폴더 경로 확인 (셀 4):
- A 계정: `DRIVE_DIR = '/content/drive/MyDrive/drone_inspect'`
- C 계정: `DRIVE_DIR = '/content/drive/MyDrive/drone_inspect_A'`
- B 계정: 본인이 추가한 이름

### Drive에 미리 업로드 (PC에서 6시 전):
- `m4_context.zip` (~1.1GB) — m4v2 노트북용

---

## 🆘 끊겼을 때:

각 노트북에 **Drive autosave + Resume** 코드 있음:
- 끊긴 시점의 last.pt가 Drive에 5분마다 저장됨
- 다시 노트북 열고 **셀 1부터 차례로 실행**하면 자동 resume

---

## 📊 결과 확인 (학습 완료 후):

각 노트북 마지막 셀에서:
```
M5v3: mAP50=0.XXXX, 0.9? YES
```
또는
```
mAP50: 0.XXXX
0.9? NO
```

---

## 💾 결과 다운로드:
1. Drive에서 결과 폴더 확인:
   - `m5v3_results/` → m5_yolo_seg_frames.onnx
   - `m4v2_results/` → m4_yolo_context_elements.onnx
   - `m1v3_results/` → m1_yolo_structural.onnx
2. ONNX 파일 다운로드
3. PC에서 `backend/models_weights/`에 덮어쓰기

---

## ⏱️ 예상 시간 (재학습 노트북):

| 노트북 | T4 | A100 |
|---|---|---|
| m5v3 | 7~8h | 4~5h |
| m2v2 | 7~9h | 5~6h |
| m4v2 | 6~7h | 4~5h |
| m1v3 | 12~14h | 8~10h |

**오늘 18:00 시작 시:**
- T4: 익일 새벽 1~6시 종료
- A100: 23시 ~ 익일 4시 종료
