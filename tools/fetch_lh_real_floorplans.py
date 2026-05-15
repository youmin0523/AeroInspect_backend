"""
tools/fetch_lh_real_floorplans.py
역할: 한국 LH 공사의 실제 분양 평면도 자료 (PDF) 를 다운로드하여 페이지별
      이미지로 변환. 테스트용 — 자동화된 가구 검출 정확도 측정 입력.

소스:
  - LH 일반분양주택 주력평면 매뉴얼 (drbuild 호스팅, 10MB PDF)
    : https://drbuild.co.kr/images/estimate/주력평면.pdf
  - LH 공공분양주택 주력평면 (수동 추가 가능)

사용:
  python tools/fetch_lh_real_floorplans.py
  python tools/fetch_lh_real_floorplans.py --pages 5-30  # 특정 페이지만
"""
from __future__ import annotations

import argparse
import sys
import ssl
import urllib.request
import urllib.parse
from pathlib import Path

DST = Path("datasets/lh_real_floorplans")
PDF_DIR = DST / "_pdf"
PAGE_DIR = DST / "pages"

# 한국어 파일명 인코딩 — drbuild 의 PDF 직접 URL
LH_SOURCES = [
    {
        "name": "lh_main_plans_2018.pdf",
        "url": "https://drbuild.co.kr/images/estimate/" + urllib.parse.quote("주력평면.pdf"),
        "license": "공공저작물 (LH 공식 매뉴얼) — 테스트 용도 사용",
        "page_range": None,  # 전체
    },
]


def _download(url: str, dest: Path, timeout: int = 300) -> bool:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            print(f"  size: {total / 1024 / 1024:.1f} MB")
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (8192 * 256) == 0:
                        pct = downloaded * 100 / total
                        print(f"\r  progress: {pct:.0f}%", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def _pdf_to_pages(pdf_path: Path, out_dir: Path, page_range: tuple[int, int] | None = None,
                   dpi: int = 200) -> list[Path]:
    """PDF → 페이지별 PNG 변환 (PyMuPDF 사용)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[ERR] PyMuPDF 미설치. pip install pymupdf")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    if page_range:
        start, end = page_range
    else:
        start, end = 1, n_pages

    saved: list[Path] = []
    zoom = dpi / 72  # PDF 기본 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for i in range(start - 1, min(end, n_pages)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix)
        out = out_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        pix.save(out)
        saved.append(out)
    doc.close()
    return saved


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--pages", type=str, default=None, help="페이지 범위 e.g. 5-30")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)

    page_range = None
    if args.pages:
        parts = args.pages.split('-')
        page_range = (int(parts[0]), int(parts[1]) if len(parts) > 1 else int(parts[0]))

    total_pages = 0
    for src in LH_SOURCES:
        pdf_path = PDF_DIR / src["name"]
        if not pdf_path.exists() or pdf_path.stat().st_size < 1024:
            print(f"=== Downloading {src['name']} ===")
            ok = _download(src["url"], pdf_path)
            if not ok:
                continue
        else:
            print(f"=== {src['name']} 이미 있음 ({pdf_path.stat().st_size // 1024 // 1024}MB) ===")

        print(f"  PDF → 페이지 PNG 변환 (DPI {args.dpi})")
        pages = _pdf_to_pages(pdf_path, PAGE_DIR, page_range or src["page_range"], args.dpi)
        total_pages += len(pages)
        print(f"  saved {len(pages)} pages → {PAGE_DIR}")

    print()
    print(f"=== 총 {total_pages} 페이지 이미지 생성 → {PAGE_DIR} ===")


if __name__ == "__main__":
    main()
