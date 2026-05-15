"""
tools/fetch_dxf_samples.py
역할: 공개된 실 CAD/DXF 평면도 샘플 다운로드 (테스트 용도).

소스:
  - jscad/sample-files (GitHub) — 일반 사용 허용
  - ezdxf 공식 examples (GitHub mozman/ezdxf) — MIT 라이선스
  - 기타 GitHub 호스팅 architectural sample

다운로드된 .dxf 파일은 parse_dxf 로 처리해 walls + furniture 추출 정확도 측정.

실행: python tools/fetch_dxf_samples.py
"""
from __future__ import annotations

import sys
import ssl
import urllib.request
from pathlib import Path

DST = Path("datasets/dxf_samples")

# GitHub raw 직접 URL (저장소 / branch / 경로)
SOURCES = [
    {
        "name": "jscad_floorplan.dxf",
        "url": "https://raw.githubusercontent.com/jscad/sample-files/master/dxf/dxf-parser/floorplan.dxf",
        "license": "(jscad sample)",
    },
    {
        "name": "ezdxf_floorplan_01.dxf",
        "url": "https://raw.githubusercontent.com/mozman/ezdxf/master/examples/Architectural_Example.dxf",
        "license": "ezdxf MIT examples",
    },
    {
        "name": "ezdxf_house.dxf",
        "url": "https://raw.githubusercontent.com/mozman/ezdxf/master/examples_dxf/AutodeskSamples/floor_plan.dxf",
        "license": "Autodesk sample",
    },
    {
        "name": "ezdxf_simple.dxf",
        "url": "https://raw.githubusercontent.com/mozman/ezdxf/master/integration_tests/CADKitSamples/AutoCAD_Sample_Drawing.dxf",
        "license": "CADKit sample",
    },
    {
        "name": "afp_sample.dxf",
        "url": "https://raw.githubusercontent.com/cansik/architectural-floor-plan/master/test/data/sample.dxf",
        "license": "AFPlan test data",
    },
]


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
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


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DST.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(" 공개 DXF 샘플 다운로드")
    print("=" * 60)

    saved = []
    for src in SOURCES:
        dest = DST / src["name"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] {dest.name} ({dest.stat().st_size // 1024}KB)")
            saved.append(dest)
            continue
        print(f"  fetching {src['name']} ...")
        ok = _download(src["url"], dest)
        if ok:
            print(f"  [OK]   {dest.name} ({dest.stat().st_size // 1024}KB) — {src['license']}")
            saved.append(dest)

    print()
    print(f"총 {len(saved)}/{len(SOURCES)} 다운로드 완료 → {DST}")
    return saved


if __name__ == "__main__":
    main()
