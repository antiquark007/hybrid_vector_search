"""
IndexManager — thin Python wrapper around the C++ hvs_core.HNSW.
Handles load/save and falls back to hnswlib if hvs_core is not compiled.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("hvs.index")


def _load_backend(dim: int, M: int, ef_construction: int, max_elements: int):
    """Try hvs_core (custom C++), fall back to hnswlib."""
    try:
        import hvs_core
        idx = hvs_core.HNSW(M=M, ef_construction=ef_construction,
                             max_elements=max_elements, dim=dim)
        logger.info("Using custom C++ HNSW backend (hvs_core)")
        return idx, "hvs_core"
    except ImportError:
        pass
    try:
        import hnswlib
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
        idx.set_ef(50)
        logger.warning("hvs_core not found — falling back to hnswlib")
        return idx, "hnswlib"
    except ImportError:
        raise RuntimeError(
            "No HNSW backend found. Build hvs_core (cmake) or `pip install hnswlib`."
        )


class IndexManager:
    def __init__(
        self,
        index_path: str = "./data/hnsw.bin",
        dim: int = 384,
        M: int = 16,
        ef_construction: int = 200,
        max_elements: int = 1_000_000,
    ):
        self.index_path = Path(index_path)
        self.dim = dim
        self._idx, self._backend = _load_backend(dim, M, ef_construction, max_elements)
        self._size = 0
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self._load()

    def _load(self):
        try:
            if self._backend == "hvs_core":
                self._idx.load(str(self.index_path))
                self._size = len(self._idx)
            else:  # hnswlib
                self._idx.load_index(str(self.index_path))
                self._size = self._idx.get_current_count()
            logger.info(f"Index loaded from {self.index_path} ({self._size} items)")
        except Exception as e:
            logger.warning(f"Could not load index: {e} — starting fresh")

    def add(self, doc_id: int, embedding: np.ndarray):
        emb = embedding.astype(np.float32)
        if self._backend == "hvs_core":
            self._idx.insert(doc_id, emb)
        else:
            self._idx.add_items(emb.reshape(1, -1), [doc_id])
        self._size += 1

    def add_batch(self, ids: np.ndarray, embeddings: np.ndarray):
        if self._backend == "hvs_core":
            self._idx.batch_insert(ids.astype(np.int32), embeddings.astype(np.float32))
        else:
            self._idx.add_items(embeddings, ids.tolist())
        self._size += len(ids)

    def search(self, query: np.ndarray, k: int = 10, ef: int = 50):
        q = query.astype(np.float32)
        if self._backend == "hvs_core":
            return self._idx.search(q, k=k, ef=ef)
        else:
            self._idx.set_ef(max(ef, k))
            labels, dists = self._idx.knn_query(q.reshape(1, -1), k=k)
            # wrap in namedtuple-like objects
            from types import SimpleNamespace
            return [SimpleNamespace(id=int(l), distance=float(d))
                    for l, d in zip(labels[0], dists[0])]

    def save(self):
        if self._backend == "hvs_core":
            self._idx.save(str(self.index_path))
        else:
            self._idx.save_index(str(self.index_path))
        logger.info(f"Index saved to {self.index_path}")

    async def save_async(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save)

    def __len__(self) -> int:
        return self._size
