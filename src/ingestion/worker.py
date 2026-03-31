"""
Celery worker — async batch ingestion from Redis queue.
Start with: celery -A src.ingestion.worker worker --loglevel=info --concurrency=4
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from celery import Celery
from celery.utils.log import get_task_logger

from ..api.embedder import Embedder
from ..api.store import DocumentStore
from ..api.index_manager import IndexManager

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DB_URL    = os.getenv("DATABASE_URL", "sqlite:///./hvs.db")
INDEX_PATH = os.getenv("INDEX_PATH", "./data/hnsw.bin")

app = Celery("hvs_worker", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    worker_prefetch_multiplier=1,   # fair dispatch
)

logger = get_task_logger(__name__)

# Module-level singletons (loaded once per worker process)
_embedder: Embedder | None = None
_store:    DocumentStore | None = None
_index:    IndexManager  | None = None


def _get_resources():
    global _embedder, _store, _index
    if _embedder is None:
        _embedder = Embedder(model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
        _store    = DocumentStore(db_url=DB_URL)
        _index    = IndexManager(index_path=INDEX_PATH, dim=_embedder.dim)
    return _embedder, _store, _index


# ─── Tasks ────────────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=5)
def ingest_document(self, doc_id: int | None, text: str, metadata: dict[str, Any]):
    """Embed and index a single document."""
    try:
        embedder, store, index = _get_resources()
        emb = embedder.encode(text)
        assigned_id = store.upsert(doc_id, text, metadata, emb)
        index.add(assigned_id, emb)
        logger.info(f"Ingested doc_id={assigned_id}")
        return {"doc_id": assigned_id, "status": "ok"}
    except Exception as exc:
        logger.exception(f"Ingest failed, retrying: {exc}")
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=2, default_retry_delay=10)
def ingest_batch(self, documents: list[dict]):
    """
    Bulk embed + index a list of documents.
    Each document: {"id": int|None, "text": str, "metadata": dict}
    """
    try:
        embedder, store, index = _get_resources()
        texts   = [d["text"] for d in documents]
        embs    = embedder.encode_batch(texts)          # (N, dim) float32
        results = []
        for i, doc in enumerate(documents):
            aid = store.upsert(doc.get("id"), doc["text"], doc.get("metadata", {}), embs[i])
            index.add(aid, embs[i])
            results.append(aid)
        index.save()
        logger.info(f"Batch ingested {len(results)} documents")
        return {"indexed": len(results), "ids": results}
    except Exception as exc:
        logger.exception(f"Batch ingest failed: {exc}")
        raise self.retry(exc=exc)


@app.task
def rebuild_index():
    """Full index rebuild from the database (use after bulk deletes)."""
    embedder, store, index_mgr = _get_resources()
    from ..api.index_manager import IndexManager
    fresh = IndexManager(
        index_path=INDEX_PATH,
        dim=embedder.dim,
    )
    total = 0
    for doc_id, emb in store.iter_all(batch_size=512):
        fresh.add(doc_id, emb)
        total += 1
    fresh.save()
    logger.info(f"Index rebuilt with {total} documents")
    return {"rebuilt": total}
