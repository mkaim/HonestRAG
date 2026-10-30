from loaders.docx import DocxFileLoader
from loaders.markdown import MarkdownFileLoader, markdown_to_blocks
from loaders.pdf import PdfFileLoader, pdf_to_blocks
from loaders.text import TextFileLoader, text_to_blocks

# Loader by file extension - the one place ingest.py's CLI and server.py's
# /api/documents endpoint both look up how to parse a given file, so adding
# a new format only means updating this dict.
LOADERS = {
    ".txt": TextFileLoader(),
    ".md": MarkdownFileLoader(),
    ".pdf": PdfFileLoader(),
    ".docx": DocxFileLoader(),
}

__all__ = [
    "LOADERS",
    "DocxFileLoader",
    "MarkdownFileLoader",
    "PdfFileLoader",
    "TextFileLoader",
    "markdown_to_blocks",
    "pdf_to_blocks",
    "text_to_blocks",
]
