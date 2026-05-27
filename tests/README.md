# AeroInspect Backend — Tests

## ONNX 4-way 매핑 회귀 가드

**신규 ONNX 가중치를 추가하거나 클래스 수를 바꿀 때 반드시 다음 테스트를 먼저 통과시킬 것:**

```bash
pytest tests/test_onnx_class_mapping.py -v
```

검증 항목 (각 모델마다):
1. ONNX 출력 dim 으로 추정한 nc
2. 학습 데이터셋 `data.yaml` 의 `names`
3. `app/services/defect_taxonomy.py` 의 `EXPECTED_CLASS_NAMES`
4. `app/services/inference_pipeline_20.py` 의 로더 인자 (`_try_load_yolo` / `_try_load_resnet`)

위 4개가 길이/순서까지 정확히 같아야 통과. 하나라도 어긋나면 2026-05-07 의 거짓 라벨 5건 동시 사고가 재발합니다.

### 환경변수
- `ONNX_WEIGHTS_DIR` — ONNX 파일 디렉터리 override (기본: `../TEAM_PROJECT_2_Drone_project/backend/models_weights`)
- `DATASETS_DIR` — `data.yaml` 들이 있는 datasets 루트 override (기본: 통합 repo 내 `training/datasets`)

파일이 없으면 `pytest.skip` 으로 graceful 처리 — CI 에서도 안전합니다.
