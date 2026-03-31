"""
Hybrid Vector Search Engine — FastAPI Service
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .embedder import Embedder
from .store import DocumentStore
from .index_manager import IndexManager

logger = logging.getLogger("hvs.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hybrid Vector Search Engine...")
    app.state.embedder = Embedder(model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    app.state.store    = DocumentStore(db_url=os.getenv("DATABASE_URL", "sqlite:///./hvs.db"))
    app.state.index    = IndexManager(
        index_path=os.getenv("INDEX_PATH", "./data/hnsw.bin"),
        dim=app.state.embedder.dim,
        M=int(os.getenv("HNSW_M", "16")),
        ef_construction=int(os.getenv("HNSW_EF_CONSTRUCTION", "200")),
    )
    logger.info(f"Engine ready — index size: {len(app.state.index)}")
    yield
    logger.info("Saving index...")
    app.state.index.save()

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid Vector Search Engine",
    version="1.0.0",
    description="C++ HNSW-backed semantic search over text documents",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    id: Optional[int]           = Field(None,  description="Optional document ID (auto-assigned if omitted)")
    text: str                   = Field(...,   min_length=1, max_length=50000)
    metadata: dict[str, Any]    = Field(default_factory=dict)

class IngestResponse(BaseModel):
    id: int
    indexed: bool
    latency_ms: float

class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(..., min_length=1, max_length=512)

class BatchIngestResponse(BaseModel):
    indexed: int
    failed: int
    latency_ms: float

class SearchRequest(BaseModel):
    query: str  = Field(..., min_length=1, max_length=2000)
    k: int      = Field(10, ge=1, le=100)
    ef: int     = Field(50, ge=10, le=500)

class SearchHit(BaseModel):
    id: int
    score: float
    text: str
    metadata: dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float

class StatsResponse(BaseModel):
    index_size: int
    documents: int
    model: str
    dim: int

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "index_size": len(app.state.index)}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    return StatsResponse(
        index_size=len(app.state.index),
        documents=app.state.store.count(),
        model=app.state.embedder.model_name,
        dim=app.state.embedder.dim,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    t0 = time.perf_counter()
    try:
        doc_id = await asyncio.get_event_loop().run_in_executor(
            None, _ingest_sync, app.state, req
        )
        return IngestResponse(
            id=doc_id,
            indexed=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_ingest", response_model=BatchIngestResponse)
async def batch_ingest(req: BatchIngestRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    indexed, failed = 0, 0
    for doc in req.documents:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, _ingest_sync, app.state, doc
            )
            indexed += 1
        except Exception:
            logger.exception(f"Failed to ingest doc id={doc.id}")
            failed += 1
    background_tasks.add_task(app.state.index.save_async)
    return BatchIngestResponse(
        indexed=indexed,
        failed=failed,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.perf_counter()
    try:
        hits = await asyncio.get_event_loop().run_in_executor(
            None, _search_sync, app.state, req
        )
        return SearchResponse(
            query=req.query,
            hits=hits,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: int):
    """Soft-delete: removes from DB. Index rebuild required for full removal."""
    removed = app.state.store.delete(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@app.post("/index/save")
async def save_index():
    app.state.index.save()
    return {"saved": True, "size": len(app.state.index)}


# ─── Sync helpers (run in thread pool) ───────────────────────────────────────

def _ingest_sync(state, req: IngestRequest) -> int:
    emb = state.embedder.encode(req.text)                  # np.ndarray
    doc_id = state.store.upsert(req.id, req.text, req.metadata, emb)
    state.index.add(doc_id, emb)
    return doc_id


def _search_sync(state, req: SearchRequest) -> list[SearchHit]:
    emb = state.embedder.encode(req.query)
    results = state.index.search(emb, k=req.k, ef=req.ef)
    hits = []
    for r in results:
        doc = state.store.get(r.id)
        if doc is None:
            continue
        hits.append(SearchHit(
            id=r.id,
            score=1.0 - float(r.distance),   # convert cosine distance → similarity
            text=doc["text"],
            metadata=doc["metadata"],
        ))
    return hits
