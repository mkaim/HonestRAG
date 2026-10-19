"""Regenerates tests/fixtures/sample.docx and sample.pdf - small, hand-built
binary files for testing DocxFileLoader/PdfFileLoader without pulling in a
PDF/DOCX-authoring dependency just for tests. Re-run after editing this file
to refresh the committed fixtures:

    uv run python tests/fixtures/make_fixtures.py
"""

import zipfile
from pathlib import Path

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

# mammoth (markitdown's .docx backend) maps the standard Heading1/Heading2
# style ids to markdown #/##, even without the styles.xml definitions those
# ids normally reference - which is why this fixture can skip styles.xml.
_DOCX_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>
<w:p><w:r><w:t>Intro paragraph under chapter one.</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>A Subsection</w:t></w:r></w:p>
<w:p><w:r><w:t>Body text in the subsection.</w:t></w:r></w:p>
<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>ITEM</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>QTY</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Widget</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>3</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:body>
</w:document>"""


def make_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _DOCX_RELS)
        z.writestr("word/document.xml", _DOCX_DOCUMENT)


# A bold, larger-than-body-text line (/F2, Helvetica-Bold, 16pt) followed by
# two regular-weight body paragraphs (/F1, Helvetica, 10pt) - just enough for
# PdfFileLoader's bold+size heading heuristic to find exactly one Header.
_PDF_CONTENT_STREAM = b"""BT
/F2 16 Tf
72 700 Td
(Document Title) Tj
ET
BT
/F1 10 Tf
72 670 Td
(This is the first body paragraph of the document.) Tj
ET
BT
/F1 10 Tf
72 650 Td
(This is a second body paragraph with more text in it.) Tj
ET
"""


def make_pdf(path: Path) -> None:
    catalog_num, pages_num, page_num = 1, 2, 3
    font_regular_num, font_bold_num, content_num = 4, 5, 6

    objects: list[bytes] = [b""] * 6
    objects[catalog_num - 1] = f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()
    objects[pages_num - 1] = (
        f"<< /Type /Pages /Kids [{page_num} 0 R] /Count 1 >>".encode()
    )
    objects[page_num - 1] = (
        f"<< /Type /Page /Parent {pages_num} 0 R "
        f"/Resources << /Font << /F1 {font_regular_num} 0 R /F2 {font_bold_num} 0 R >> >> "
        f"/MediaBox [0 0 612 792] /Contents {content_num} 0 R >>"
    ).encode()
    objects[font_regular_num - 1] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    objects[font_bold_num - 1] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
    )
    objects[content_num - 1] = (
        f"<< /Length {len(_PDF_CONTENT_STREAM)} >>\nstream\n".encode()
        + _PDF_CONTENT_STREAM
        + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()

    path.write_bytes(bytes(out))


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    make_docx(fixtures_dir / "sample.docx")
    make_pdf(fixtures_dir / "sample.pdf")
    print(f"wrote {fixtures_dir / 'sample.docx'} and {fixtures_dir / 'sample.pdf'}")
