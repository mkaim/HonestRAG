from markitdown import MarkItDown

from loaders.markdown import markdown_to_blocks
from models import Block, Loader


class DocxFileLoader(Loader):
    """Reads a .docx file by converting it to Markdown (via markitdown's
    mammoth-based converter, which maps Word heading styles to #/##/... and
    simple tables to pipe syntax) and reusing markdown_to_blocks on the
    result. `source` is a filesystem path."""

    def __init__(self):
        self._converter = MarkItDown()

    def load(self, source: str) -> tuple[list[Block], dict]:
        text = self._converter.convert(source).text_content
        return markdown_to_blocks(text), {}
