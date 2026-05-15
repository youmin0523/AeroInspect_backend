"""
tools/fetch_real_floorplans.py
역할: 공개된 진짜 평면도 이미지를 다운로드해 정확도 측정용 데이터셋 구축.

소스 (저작권 안전 — Public Domain / CC BY-SA):
  - Wikimedia Commons: 공공 평면도 (역사 건축, 일반 주택 평면도)
  - 학술 데이터셋 샘플 (CubiCasa5K 일부)

저장 위치: ./datasets/real_floorplans/
실행: python tools/fetch_real_floorplans.py
"""
from __future__ import annotations

import sys
import ssl
import urllib.request
from pathlib import Path

DST = Path("datasets/real_floorplans")

# Wikimedia Commons 공개 평면도 — Special:FilePath redirect 사용
# (썸네일 직접 URL 은 차단되어 있음)
def _wiki(filename: str) -> str:
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width=1024"


SOURCES = [
    {
        "name": "wiki_one_room_school.jpg",
        "url": _wiki("One-room_schoolhouse_floor_plan.jpg"),
        "license": "Public Domain",
    },
    {
        "name": "wiki_house_plan.png",
        "url": _wiki("Floor_plan.png"),
        "license": "CC BY-SA",
    },
    {
        "name": "wiki_apartment_floorplan.png",
        "url": _wiki("Apartment_floorplan.png"),
        "license": "CC BY-SA",
    },
    {
        "name": "wiki_apartment_design.jpg",
        "url": _wiki("Apartment_design.jpg"),
        "license": "CC BY-SA",
    },
    {
        "name": "wiki_floor_plan_house.png",
        "url": _wiki("FloorPlanFinal2.png"),
        "license": "CC BY-SA",
    },
    {
        "name": "wiki_simple_floor_plan.png",
        "url": _wiki("Simple_floor_plan.png"),
        "license": "CC BY-SA",
    },
]


def _download(url: str, dest: Path, timeout: int = 20) -> bool:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {type(e).__name__}: {e}")
        return False


def fetch_all() -> list[Path]:
    DST.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for src in SOURCES:
        dest = DST / src["name"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] {dest.name} ({dest.stat().st_size // 1024}KB) — already exists")
            saved.append(dest)
            continue
        print(f"  fetching {src['name']} ...")
        ok = _download(src["url"], dest)
        if ok:
            print(f"  [OK]   {dest.name} ({dest.stat().st_size // 1024}KB) — {src['license']}")
            saved.append(dest)
    return saved


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print(" 공개 평면도 데이터 다운로드")
    print("=" * 60)
    paths = fetch_all()
    print()
    print(f"총 {len(paths)}/{len(SOURCES)} 다운로드 완료 → {DST}")
