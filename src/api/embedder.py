"""
Embedder — wraps sentence-transformers with caching and batching.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger("hvs.embedder")


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 4096):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded — dim={self.dim}")
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        emb = self._model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        if len(self._cache) < self._cache_size:
            self._cache[key] = emb
        return emb

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Returns (N, dim) float32 array."""
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)
