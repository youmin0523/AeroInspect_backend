"""
tools/fetch_cubicasa5k.py
역할: CubiCasa5K (5K 장 라벨된 평면도) 데이터셋 다운로드.

CubiCasa5K
  - 출처: https://zenodo.org/record/2613548
  - 라이선스: Creative Commons Attribution 4.0 International (CC-BY 4.0)
  - 크기: 약 5GB (압축) / ~25GB (원본)
  - 라벨: 벽 + 방 + 가구 (45+ 클래스 SVG)

대안 (더 작음):
  - HouseExpo (35K 장, 점유 격자) — https://github.com/TeaganLi/HouseExpo
  - ROBIN (50 장, 빠른 검증용) — https://github.com/dchhibba/sym-dataset

용도:
  - 평면도 가구 검출 ML 모델 학습 (YOLO / Mask R-CNN)
  - 현재 도형 기반 검출의 ground truth 비교

실행:
  python tools/fetch_cubicasa5k.py --target ./datasets/cubicasa5k
  python tools/fetch_cubicasa5k.py --target ./datasets/cubicasa5k --sample-only  # 샘플 50장만

주의: 풀 다운로드는 시간/디스크 부담 큼. 자동 다운로드 시도 후 실패하면
       수동 다운로드 가이드 출력.
"""
from __future__ import annotations

import argparse
import sys
import ssl
import urllib.request
import zipfile
from pathlib import Path


# Zenodo direct download (large files — 약 4-5GB)
CUBICASA5K_URLS = [
    "https://zenodo.org/record/2613548/files/cubicasa5k.zip",
]

# 작은 sample (예시 — 실제 sample 페이지 없으면 풀 다운로드 안내)
HOUSEEXPO_URL = "https://github.com/TeaganLi/HouseExpo/archive/refs/heads/master.zip"


def _download(url: str, dest: Path, timeout: int = 600, chunk: int = 8192) -> bool:
    """대용량 파일 스트리밍 다운로드."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            print(f"  size: {total / 1024 / 1024:.1f} MB")
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total > 0 and downloaded % (chunk * 1024) == 0:
                        pct = downloaded * 100 / total
                        print(f"\r  progress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)",
                              end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def download_cubicasa5k(target: Path, sample_only: bool = False) -> bool:
    target.mkdir(parents=True, exist_ok=True)
    if sample_only:
        print("=== HouseExpo (샘플 대체 — CubiCasa 보다 가벼움) ===")
        zip_path = target / "houseexpo.zip"
        ok = _download(HOUSEEXPO_URL, zip_path, timeout=300)
        if ok and zip_path.exists():
            print(f"  압축 해제 → {target}/")
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target)
                zip_path.unlink()
                return True
            except Exception as e:
                print(f"  [FAIL] zip extract: {e}")
        return False

    print("=== CubiCasa5K 다운로드 (~5GB) ===")
    for url in CUBICASA5K_URLS:
        zip_path = target / "cubicasa5k.zip"
        print(f"  fetching {url}")
        ok = _download(url, zip_path, timeout=1800)
        if ok and zip_path.exists():
            print(f"  압축 해제 → {target}/")
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target)
                zip_path.unlink()
                return True
            except Exception as e:
                print(f"  [FAIL] zip extract: {e}")
    return False


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("datasets/cubicasa5k"))
    p.add_argument("--sample-only", action="store_true",
                   help="CubiCasa 대신 HouseExpo (가벼움) 다운로드")
    p.add_argument("--check-only", action="store_true",
                   help="다운로드 안 하고 환경/접근성만 확인")
    args = p.parse_args()

    if args.check_only:
        ctx = ssl.create_default_context()
        for url in CUBICASA5K_URLS + [HOUSEEXPO_URL]:
            try:
                req = urllib.request.Request(url, method="HEAD",
                                              headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                    size = r.headers.get("Content-Length", "?")
                    print(f"[OK] {url}  ({int(size)/1024/1024:.1f} MB)" if size != "?"
                          else f"[OK] {url}")
            except Exception as e:
                print(f"[FAIL] {url}  → {type(e).__name__}: {e}")
        return

    ok = download_cubicasa5k(args.target, sample_only=args.sample_only)
    if not ok:
        print()
        print("=" * 70)
        print(" 자동 다운로드 실패 — 수동 다운로드 가이드:")
        print("=" * 70)
        print("  1) https://zenodo.org/record/2613548 접속")
        print("  2) cubicasa5k.zip (~5GB) 다운로드")
        print(f"  3) 압축 해제: {args.target}/")
        print("  4) 검증: ls", args.target, "→ colorful/, high_quality/, high_quality_architectural/ 폴더 확인")
        print()
        print("  대안 (가벼움):")
        print(f"    python tools/fetch_cubicasa5k.py --target {args.target} --sample-only")
        sys.exit(1)
    else:
        print()
        print(f"=== 다운로드 완료 → {args.target} ===")
        print("  다음 단계: tools/train_floorplan_yolo.py 로 학습")


if __name__ == "__main__":
    main()
