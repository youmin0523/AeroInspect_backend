"""
generate_mock_chat_files.py
역할: seed_mock_chats.py가 DB에 시드한 mock 메시지의 file_url 들이 가리키는
      실제 파일을 backend/uploads/chat/ 아래에 placeholder로 생성한다.

      - 이미지(.jpg): PIL로 라벨이 들어간 placeholder JPG
      - PDF(.pdf):    handcrafted 최소 유효 PDF (제목 텍스트만)
      - XLSX(.xlsx):  zipfile로 최소 유효 XLSX (한 셀에 라벨)

실행: cd backend && python -m scripts.generate_mock_chat_files
"""

import io
import os
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "uploads" / "chat"


# ── 이미지 placeholder ─────────────────────────────────
IMAGE_FILES = [
    ("mock_crack_detection.jpg",  "A동 외벽 균열 탐지 결과",       (180, 60, 70)),
    ("mock_crack_closeup.jpg",    "3층 균열 확대",                  (200, 80, 60)),
    ("mock_drone_overview.jpg",   "A동 전경 (드론 촬영)",           (60, 110, 180)),
    ("mock_thermal_image.jpg",    "북측 외벽 열화상",               (220, 90, 40)),
    ("mock_thermal_marked.jpg",   "북측 열화상 (마크업)",           (200, 60, 100)),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    """한글 가능한 시스템 폰트를 우선 시도, 실패 시 기본 폰트."""
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_image(path: Path, label: str, color: tuple) -> None:
    w, h = 800, 560
    img = Image.new("RGB", (w, h), color)
    draw = ImageDraw.Draw(img)

    # 격자 패턴으로 placeholder 느낌
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 80), width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 80), width=1)

    # 중앙 라벨
    title_font = _load_font(36)
    sub_font = _load_font(20)

    # 텍스트 박스
    box_pad = 24
    bbox = draw.textbbox((0, 0), label, font=title_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx0 = (w - tw) // 2 - box_pad
    by0 = (h - th) // 2 - box_pad
    bx1 = bx0 + tw + box_pad * 2
    by1 = by0 + th + box_pad * 2 + 36
    draw.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255))
    draw.text(((w - tw) // 2, (h - th) // 2 - 12), label, fill=(40, 40, 40), font=title_font)
    draw.text(((w - tw) // 2, (h - th) // 2 + th + 4), "Mock placeholder",
              fill=(120, 120, 120), font=sub_font)

    img.save(path, "JPEG", quality=88)


# ── PDF placeholder ─────────────────────────────────
def make_pdf(path: Path, title: str) -> None:
    """제목 텍스트 1줄만 들어간 최소 유효 PDF."""
    # WinAnsi로 표현 가능한 ASCII 라벨로 (한글은 폰트 임베드 필요해서 latin1만 사용)
    safe = title.encode("latin-1", errors="replace").decode("latin-1")

    objects = []

    def add(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    # 1: Catalog, 2: Pages, 3: Page, 4: Font, 5: Contents
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    stream_text = f"BT /F1 18 Tf 72 760 Td ({safe}) Tj ET\nBT /F1 12 Tf 72 730 Td (Mock placeholder PDF) Tj ET"
    stream_bytes = stream_text.encode("latin-1")
    contents = b"<< /Length " + str(len(stream_bytes)).encode() + b" >>\nstream\n" + stream_bytes + b"\nendstream"
    add(contents)

    # PDF 직렬화 (xref 포함)
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()

    path.write_bytes(bytes(out))


# ── XLSX placeholder (zipfile 기반 최소 유효 워크북) ──
def make_xlsx(path: Path, label: str) -> None:
    safe = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"""

    sheet = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c></row>
  </sheetData>
</worksheet>"""

    shared_strings = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
  <si><t>{safe}</t></si>
  <si><t>Mock placeholder XLSX</t></si>
</sst>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
        zf.writestr("xl/sharedStrings.xml", shared_strings)


# ── main ─────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    # 이미지
    for fname, label, color in IMAGE_FILES:
        p = OUT_DIR / fname
        make_image(p, label, color)
        made.append(str(p))

    # PDF
    for fname, title in [
        ("mock_report_march.pdf",       "March Inspection Report (mock)"),
        ("mock_flight_schedule.pdf",    "Drone Flight Schedule - April (mock)"),
        ("mock_safety_checklist.pdf",   "Safety Inspection Checklist v3 (mock)"),
    ]:
        p = OUT_DIR / fname
        make_pdf(p, title)
        made.append(str(p))

    # XLSX
    p = OUT_DIR / "mock_defect_data.xlsx"
    make_xlsx(p, "Defect Data Analysis v2 (mock)")
    made.append(str(p))

    print(f"[ok] generated {len(made)} placeholder files in {OUT_DIR}")
    for m in made:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
