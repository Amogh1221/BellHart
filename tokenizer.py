from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, path: str | None = None):
        if path is None:
            path = str(
                Path(__file__).resolve().parent / "tokenizer.json"
            )
        self._tok = HFTokenizer.from_file(path)
        self._vocab_size = self._tok.get_vocab_size()
        self._eot_id = 0

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eot_token(self) -> int:
        return self._eot_id
