"""Embedding backends for HASY memory.

The resolver and episode recall depend only on the `Embedder` protocol, so the
real model can be swapped without touching resolution logic — and tests can run
with no model download and no API key.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

DEFAULT_DIM = 384


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch. Must be deterministic for a given input."""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class HashEmbedder:
    """Deterministic character-n-gram hashing embedder. No dependencies.

    Cheap and offline, which makes it the right default for tests and for a
    first run before a real model is wired in. It captures *lexical* overlap
    only — 'Phani' and 'Phani Meduri' land near each other, but 'the platform
    architect' will not. Semantic matches are the adjudicator's job, and that
    division is deliberate: it keeps the expensive model call rare.
    """

    def __init__(self, dim: int = DEFAULT_DIM, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def _grams(self, text: str) -> list[str]:
        t = f"  {text.lower().strip()} "
        if len(t) <= self.ngram:
            return [t]
        return [t[i : i + self.ngram] for i in range(len(t) - self.ngram + 1)]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for gram in self._grams(text):
                h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out
