from pathlib import Path

from models import Loader


class TextFileLoader(Loader):
    """Reads a plain text/markdown file as-is. `source` is a filesystem path."""

    def load(self, source: str) -> tuple[str, dict]:
        return Path(source).read_text(), {}
