"""
tokenizer.py  —  BellHart Tokenizer Interface
===============================================
Wraps the Hugging Face Tokenizers backend for fast Rust-based BPE encoding
and decoding with a standardized 32,768 vocabulary.
"""

from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer wrapper for BellHart.
    
    Vocabulary: 32,768 tokens.
    Special Tokens:
      - End-of-Text (EOT): Token ID 0 (delimiters between documents and turns).
    """

    def __init__(self, path: str | None = None):
        if path is None:
            path = str(Path(__file__).resolve().parent / "tokenizer.json")
        self._tok = HFTokenizer.from_file(path)
        self._vocab_size = self._tok.get_vocab_size()
        self._eot_id = 0

    def encode(self, text: str) -> list[int]:
        """Encodes raw text into a list of integer token IDs."""
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of raw strings into a list of token ID lists."""
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into human-readable text."""
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (32,768)."""
        return self._vocab_size

    @property
    def eot_token(self) -> int:
        """End-of-Text token ID (0)."""
        return self._eot_id
