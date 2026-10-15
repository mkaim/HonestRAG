"""Fetch a GutenQA eval corpus: long-form prose with sparse chapter headers
(the opposite structural extreme from the D&D SRD example - few headers,
long sections), from public-domain Project Gutenberg novels, with gold Q&A
from the LumberChunker/GutenQA dataset on Hugging Face (MIT). Not vendored
here - see tests/eval/README.md for why.

    uv run python scripts/fetch_gutenqa.py tests/eval/data/gutenqa

GutenQA ships 100 books (30 questions each) as pre-chunked parquet, not
markdown - so this script fetches each book's *original* text straight from
Project Gutenberg instead. Gutenberg's own chapter wording is wildly
inconsistent across books ("CHAPTER I.", "Chapter I", "STAVE I", bare "I."
for a short-story collection, ...), so instead of guessing at text patterns
this parses Gutenberg's HTML edition, where every book consistently marks
each chapter as `<h2><a id="chapNN">...`, regardless of how the visible
chapter title itself reads - and converts that structure straight to
markdown (h2 -> #, h3 -> ##, p -> paragraph). By default fetches a fixed,
curated subset of 16 well-known, moderately-sized books (--books
overrides); the full 100 would take far longer to ingest for not much more
eval signal.

Writes <out_dir>/docs/*.md and <out_dir>/gold.json, in the same generic
format tests/eval/golden.py reads (see scripts/fetch_dnd_srd.py):

    [{"question": str, "answer": str,
      "passages": [{"content": str, "document": str}]}]

(No "tier" field - GutenQA's questions aren't tiered.)
"""

import argparse
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

QUESTIONS_URL = (
    "https://huggingface.co/datasets/LumberChunker/GutenQA/resolve/main/"
    "questions.parquet"
)

# Project Gutenberg ebook id per book, keyed by the exact "Book Name" string
# GutenQA uses - keep in sync with questions.parquet's "Book Name" column.
DEFAULT_BOOKS = {
    "A_Christmas_Carol_-_Charles_Dickens": 46,
    "Alice's_Adventures_in_Wonderland_-_Lewis_Carroll": 11,
    "The_Adventures_of_Sherlock_Holmes_-_Arthur_Conan_Doyle": 1661,
    "The_Call_of_the_Wild_-_Jack_London": 215,
    "The_Picture_of_Dorian_Gray_-_Oscar_Wilde": 174,
    "The_Strange_Case_of_Dr._Jekyll_and_Mr._Hyde_-_Robert_Louis_Stevenson": 43,
    "The_Time_Machine_-_H._G._Wells": 35,
    "The_War_of_the_Worlds_-_H._G._Wells": 36,
    "Treasure_Island_-_Robert_Louis_Stevenson": 120,
    "The_Wonderful_Wizard_of_Oz_-_L._Frank_Baum": 55,
    "Dracula_-_Bram_Stoker": 345,
    "Frankenstein_Or_The_Modern_Prometheus_-_Mary_Wollstonecraft_Shelley": 84,
    "Pride_and_Prejudice_-_Jane_Austen": 1342,
    "The_Adventures_of_Tom_Sawyer_-_Mark_Twain": 74,
    "Anne_of_Green_Gables_-_L._M._Montgomery": 45,
    "The_Hound_of_the_Baskervilles_-_Arthur_Conan_Doyle": 2852,
}

_START_RE = re.compile(r"START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_END_RE = re.compile(r"END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "div"}
_SKIP_TAGS = {"script", "style", "nav"}


class _BookHTMLParser(HTMLParser):
    """Walks Gutenberg's HTML edition (already sliced to the body region,
    between its start/end markers - see _to_markdown) and emits markdown.

    h2 is a chapter ONLY when it wraps an `<a id="...">` anchor - every
    sampled book marks real chapters this way regardless of its visible
    wording ("CHAPTER I.", "STAVE I", a bare "I.", ...) and regardless of
    the id's own naming scheme (seen both "chap01" and "link1"). An h2
    without such an anchor is front matter (title page, "Contents",
    "Illustrations", ...), used purely for typography - its text is
    dropped rather than treated as a false chapter break. h3 is a
    sub-section break within a chapter. Anything outside a heading or
    paragraph (nav links, images, inline formatting tags) is ignored;
    inline tag text still passes through."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._buffer: list[str] = []
        self._h2_has_anchor = False

    def _flush(self, prefix: str = "", *, drop: bool = False) -> None:
        text = "".join(self._buffer).strip()
        text = re.sub(r"\s+", " ", text)
        self._buffer.clear()
        if text and not drop:
            self.lines.append(f"{prefix}{text}" if prefix else text)

    def _open_tag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
            # A line break within a heading/paragraph is a word separator,
            # not a reason to flush - e.g. Dracula's <h2>CHAPTER I<br>
            # JONATHAN HARKER'S JOURNAL</h2> is one heading, two lines.
            self._buffer.append(" ")
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag == "h2":
                self._h2_has_anchor = False
        if tag == "a" and self._tag_stack and self._tag_stack[-1] == "h2":
            if any(name == "id" for name, _ in attrs):
                self._h2_has_anchor = True
        self._tag_stack.append(tag)

    def handle_starttag(self, tag, attrs):
        self._open_tag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag == "h1":
            self._flush("# ")
        elif tag == "h2":
            self._flush("# ", drop=not self._h2_has_anchor)
        elif tag == "h3":
            self._flush("## ")
        elif tag in ("p", "div"):
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        self._buffer.append(data)

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._buffer.append(" ")


def _fetch_html(book_id: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images.html"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8", errors="replace")


def _to_markdown(html: str) -> str:
    start = _START_RE.search(html)
    end = _END_RE.search(html)
    body = html[start.end() if start else 0 : end.start() if end else len(html)]

    parser = _BookHTMLParser()
    parser.feed(body)
    return "\n\n".join(parser.lines)


def _fetch_questions() -> list[dict]:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "fetch_gutenqa.py needs pandas+pyarrow to read GutenQA's parquet "
            "questions file (not project dependencies - this script only):\n"
            "    uv run --with pandas --with pyarrow python scripts/fetch_gutenqa.py ..."
        ) from e

    with urllib.request.urlopen(QUESTIONS_URL) as response:
        tmp = Path("/tmp/gutenqa_questions.parquet")
        tmp.write_bytes(response.read())
    return pd.read_parquet(tmp).to_dict("records")


def fetch(out_dir: Path, books: dict[str, int]) -> None:
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for name, book_id in books.items():
        html = _fetch_html(book_id)
        markdown = _to_markdown(html)
        (docs_dir / f"{name}.md").write_text(markdown)
    print(f"Wrote {len(books)} document(s) to {docs_dir}")

    questions = _fetch_questions()
    gold = [
        {
            "question": q["Question"],
            "answer": q["Answer"],
            "passages": [
                {"content": q["Chunk Must Contain"], "document": f"{q['Book Name']}.md"}
            ],
        }
        for q in questions
        if q["Book Name"] in books
    ]
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=2))
    print(f"Wrote {len(gold)} question(s) to {out_dir / 'gold.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="directory to write docs/ and gold.json into")
    parser.add_argument(
        "--books",
        help="comma-separated 'Book Name:gutenberg_id' pairs to override the "
        "default 16-book selection, e.g. 'Dracula_-_Bram_Stoker:345'",
    )
    args = parser.parse_args()

    books = DEFAULT_BOOKS
    if args.books:
        books = dict(pair.split(":") for pair in args.books.split(","))
        books = {name: int(book_id) for name, book_id in books.items()}

    fetch(args.out_dir, books)


if __name__ == "__main__":
    main()
