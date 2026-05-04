"""10분 간격 학습 모니터. 별도 터미널에서 실행: python monitor.py"""
import time, subprocess, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TASKS = os.path.expandvars(
    r"C:\Users\Codelab\AppData\Local\Temp\claude"
    r"\c--Users-Codelab-Desktop-PROJECT-TEAM-PROJECT-2-Drone-project"
    r"\958d11c0-2e1d-469a-afec-1ac2e25d0647\tasks"
)

M5 = os.path.join(TASKS, "bsmkz9p1p.output")
RESNET = os.path.join(TASKS, "bz20n7f0p.output")

def grep(path, pattern):
    try:
        with open(path, "r", errors="replace") as f:
            return [l.strip() for l in f if pattern in l]
    except: return []

while True:
    ts = time.strftime("%H:%M")
    print(f"\n{'='*50}")
    print(f"[{ts}] 학습 현황")
    print(f"{'='*50}")

    # M5
    epochs = [l for l in grep(M5, "/200") if "/200" in l]
    ep = epochs[-1].split("/200")[0].split()[-1] if epochs else "?"
    alls = grep(M5, "   all   ")
    last3 = alls[-3:] if alls else []
    print(f"\nM5 YOLO (GPU): epoch {ep}/200")
    for a in last3:
        parts = a.split()
        if len(parts) >= 8:
            print(f"  mAP50={parts[6]:>6s}  mAP50-95={parts[7]:>6s}  P={parts[4]:>6s}  R={parts[5]:>6s}")

    # ResNet
    rlines = grep(RESNET, "Epoch")
    if rlines:
        print(f"\nResNet (GPU):")
        for r in rlines[-3:]:
            print(f"  {r}")
    else:
        print(f"\nResNet (GPU): 데이터 로딩 중...")

    print(f"\n다음 체크: {int(time.strftime('%M'))//10*10+10:02d}분")
    time.sleep(600)
