from collections import Counter

import pdfplumber

from models import Block, Header, Loader, Paragraph

# PDFs carry no semantic markup (no real <h1>/<p> tags), only positioned
# glyphs with a font name and size - so headings have to be inferred from
# typography. A line is a heading candidate when it's bold-styled and at
# least as large as the document's own body text (its most common
# non-bold font size); candidates are leveled by distinct size, largest
# first. Bold-at-body-size catches subsection headers that share the body
# font size but not its weight (common in papers/reports); bold-and-larger
# catches top-level section headers.
_BOLD_HINTS = ("bold", "medi", "black", "heavy", "semibold")
_MAX_HEADING_WORDS = 12


def _is_bold(fontname: str) -> bool:
    return any(hint in fontname.lower() for hint in _BOLD_HINTS)


def _extract_lines(pdf: pdfplumber.PDF) -> list[dict]:
    """Groups words into lines by y-position per page. x_tolerance=2 (below
    pdfplumber's default of 3) fixes word-gluing some PDFs trigger at the
    default gap-width threshold - e.g. arXiv's LaTeX-produced PDFs, whose
    actual inter-word gaps are narrower than pdfplumber assumes."""
    lines = []
    for page in pdf.pages:
        words = page.extract_words(extra_attrs=["size", "fontname"], x_tolerance=2)
        if not words:
            continue
        words.sort(key=lambda w: (round(w["top"], 0), w["x0"]))
        current_top: float | None = None
        buf: list[dict] = []
        for word in words:
            top = round(word["top"], 0)
            if current_top is None or abs(top - current_top) > 2:
                if buf:
                    lines.append(_line_from_words(buf))
                buf = [word]
                current_top = top
            else:
                buf.append(word)
        if buf:
            lines.append(_line_from_words(buf))
    return lines


def _line_from_words(words: list[dict]) -> dict:
    text = " ".join(w["text"] for w in words)
    sizes = Counter(round(w["size"], 1) for w in words)
    dominant_size = sizes.most_common(1)[0][0]
    # Bold only if every word on the line is bold-styled - a line that's
    # mostly body text with one bolded term isn't a heading.
    bold = all(_is_bold(w["fontname"]) for w in words)
    return {"text": text, "size": dominant_size, "bold": bold, "words": len(words)}


def _body_text_size(lines: list[dict]) -> float:
    sizes = Counter(line["size"] for line in lines if not line["bold"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _heading_levels(lines: list[dict], body_size: float) -> dict[float, int]:
    """Ranks distinct heading-candidate sizes into levels, largest first.
    Candidates are filtered to exclude noise that happens to be bold and
    large: lines with many words (likely bold body text, not a heading),
    single-character lines (drop caps), and lines repeated verbatim
    elsewhere in the document (running headers/footers, reused captions)."""
    text_counts = Counter(line["text"] for line in lines)
    sizes = {
        line["size"]
        for line in lines
        if line["bold"]
        and line["size"] >= body_size
        and line["words"] <= _MAX_HEADING_WORDS
        and len(line["text"]) > 1
        and text_counts[line["text"]] == 1
    }
    return {size: level + 1 for level, size in enumerate(sorted(sizes, reverse=True))}


def pdf_to_blocks(path: str) -> list[Block]:
    with pdfplumber.open(path) as pdf:
        lines = _extract_lines(pdf)
    body_size = _body_text_size(lines)
    levels = _heading_levels(lines, body_size)

    blocks: list[Block] = []
    for line in lines:
        level = levels.get(line["size"]) if line["bold"] else None
        if level is not None:
            blocks.append(Header(line["text"], level=level))
        else:
            blocks.append(Paragraph(line["text"]))
    return blocks


class PdfFileLoader(Loader):
    """Reads a PDF file, inferring headers from font weight/size (see
    pdf_to_blocks) and treating every other line as its own paragraph.
    `source` is a filesystem path."""

    def load(self, source: str) -> tuple[list[Block], dict]:
        return pdf_to_blocks(source), {}
