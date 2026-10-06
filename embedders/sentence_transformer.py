import asyncio

from sentence_transformers import SentenceTransformer

from models import Embedder


class HfTokenizer:
    """Adapter over a HuggingFace tokenizer exposing token counting and
    offset-based splitting into exact original-text slices."""

    def __init__(self, tokenizer, max_tokens: int):
        self._tokenizer = tokenizer
        self.max_tokens = max_tokens

    def _encode(self, text: str):
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            # Long texts are encoded only to be counted or split, never fed to
            # the model whole, so skip the "longer than max length" warning.
            verbose=False,
        )
        return enc["input_ids"], enc["offset_mapping"]

    def count(self, text: str) -> int:
        ids, _ = self._encode(text)
        return len(ids)

    def split(self, text: str, limit: int) -> list[str]:
        """Cut `text` into consecutive slices of at most `limit` tokens. Cuts
        land on token starts (token end offsets can overlap the next token),
        and a slice that re-tokenizes to more than `limit` tokens, e.g. a word
        cut in half, is shortened by a token."""
        ids, offsets = self._encode(text)
        cuts = [start for start, _ in offsets] + [len(text)]
        pieces = []
        i = 0
        while i < len(ids):
            j = min(i + limit, len(ids))
            while j > i + 1 and self.count(text[cuts[i] : cuts[j]]) > limit:
                j -= 1
            if piece := text[cuts[i] : cuts[j]].strip():
                pieces.append(piece)
            i = j
        return pieces

    def tail(self, text: str, n: int) -> str:
        if n <= 0 or not text:
            return ""
        ids, offsets = self._encode(text)
        if len(ids) <= n:
            return text
        for start, _ in offsets[len(ids) - n :]:
            if self.count(text[start:]) <= n:
                return text[start:]
        return ""


class SenTranEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
    ):
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.transformer = SentenceTransformer(model)
        # get_sentence_embedding_dimension was renamed to get_embedding_dimension;
        # fall back to the old name for sentence-transformers versions that
        # predate the rename.
        get_dim = (
            getattr(self.transformer, "get_embedding_dimension", None)
            or self.transformer.get_sentence_embedding_dimension
        )
        self.dims = get_dim()
        # max_seq_length includes the special tokens the model adds; HfTokenizer
        # counts without them.
        hf_tokenizer = self.transformer.tokenizer
        self.max_input_tokens = (
            self.transformer.max_seq_length - hf_tokenizer.num_special_tokens_to_add()
        )
        # The tokenizer's limit is what's left for a chunk's text once
        # document_prefix is prepended to it.
        self.tokenizer = HfTokenizer(hf_tokenizer, self.max_input_tokens)
        self.tokenizer.max_tokens -= self.tokenizer.count(document_prefix)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            count = self.tokenizer.count(text)
            if count > self.max_input_tokens:
                raise ValueError(
                    f"text exceeds embedder token limit "
                    f"({count} > {self.max_input_tokens})"
                )
        embeddings = await asyncio.to_thread(
            self.transformer.encode,
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()
