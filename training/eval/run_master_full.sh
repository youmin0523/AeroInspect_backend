#!/bin/bash
# 확장 마스터 — 모든 카드 다 시도
set -e
cd "/c/Users/Codelab/Desktop/PROJECT/TEAM_PROJECT_2_Drone_project/backend/training"
PY="C:/Users/Codelab/Desktop/PROJECT/TEAM_PROJECT_2_Drone_project/backend/venv/Scripts/python.exe"
LOG_DIR="eval/results/master_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_DIR/master.log"
}

log "=== 확장 마스터 시작 ==="

# Step 1: extreme_boost 완료 대기
log "Step 1: extreme_boost 대기"
START_WAIT=$(date +%s)
while true; do
    f=$(ls -t eval/results/extreme_boost_*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ -n "$mtime" ] && [ "$mtime" -gt "$START_WAIT" ]; then
            break
        fi
    fi
    sleep 60
done
log "  extreme_boost 완료: $f"
cat "$f" >> "$LOG_DIR/extreme_boost_result.json"

# Step 2: WBF 3개 모델 full test set
for M in M3 M2 M5; do
    log "Step 2.$M: WBF $M full"
    "$PY" eval/evaluate_wbf.py --model $M --max-images 9999 > "$LOG_DIR/wbf_$M.log" 2>&1 || log "    FAIL"
    grep -E "WBF: mAP50|0.85 갭" "$LOG_DIR/wbf_$M.log" | tail -3 | tee -a "$LOG_DIR/master.log"
done

# Step 3: Multi-model voting (cross_model 진짜 효과)
for T in M3 M2 M5; do
    log "Step 3.$T: Multi-model voting (target=$T)"
    "$PY" eval/evaluate_multi_model_voting.py --target $T --max-images 200 > "$LOG_DIR/voting_$T.log" 2>&1 || log "    FAIL"
    grep -E "alone|cross_model" "$LOG_DIR/voting_$T.log" | tail -5 | tee -a "$LOG_DIR/master.log"
done

# Step 4: SAHI tiled (작은 객체)
for M in M5 M1 furniture; do
    log "Step 4.$M: SAHI tiled $M"
    "$PY" eval/evaluate_sahi_tiled.py --model $M --max-images 200 > "$LOG_DIR/sahi_$M.log" 2>&1 || log "    FAIL"
    grep -E "baseline|tiled|0.85" "$LOG_DIR/sahi_$M.log" | tail -3 | tee -a "$LOG_DIR/master.log"
done

# Step 5: 종합 리포트
log "Step 5: FINAL_REPORT 생성"
"$PY" eval/generate_final_report.py > "$LOG_DIR/report.log" 2>&1 || log "  report 실패"

log "=== 확장 마스터 완료 ==="
log "결과: $LOG_DIR"
