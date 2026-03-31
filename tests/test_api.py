"""
tests/test_api.py — integration tests for the FastAPI endpoints.
Run: pytest tests/test_api.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Lightweight mocks so tests run without GPU / C++ module ──────────────────
import sys
from unittest.mock import MagicMock, patch

# Mock hvs_core so tests work without the compiled module
hvs_core_mock = MagicMock()
sys.modules.setdefault("hvs_core", hvs_core_mock)


class _FakeResult:
    def __init__(self, id_, dist): self.id = id_; self.distance = dist

class _FakeIndex:
    def __init__(self, *a, **kw): self._data = {}
    def insert(self, id_, emb): self._data[id_] = emb
    def search(self, query, k=10, ef=50):
        return [_FakeResult(i, 0.1 * j) for j, i in enumerate(list(self._data)[:k])]
    def save(self, path): pass
    def load(self, path): pass
    def __len__(self): return len(self._data)
    def __contains__(self, id_): return id_ in self._data

hvs_core_mock.HNSW.side_effect = lambda **kw: _FakeIndex()

class _FakeEmbedder:
    model_name = "test-model"
    dim = 8
    def encode(self, text): return np.random.rand(self.dim).astype(np.float32)

class _FakeStore:
    def __init__(self, *a, **kw): self._db = {}; self._ctr = 0
    def upsert(self, id_, text, meta, emb):
        if id_ is None: self._ctr += 1; id_ = self._ctr
        self._db[id_] = {"id": id_, "text": text, "metadata": meta}
        return id_
    def get(self, id_): return self._db.get(id_)
    def delete(self, id_):
        if id_ in self._db: del self._db[id_]; return True
        return False
    def count(self): return len(self._db)

class _FakeIndexManager:
    def __init__(self, *a, **kw): self._idx = _FakeIndex()
    def add(self, id_, emb): self._idx.insert(id_, emb)
    def search(self, emb, k=10, ef=50): return self._idx.search(emb, k=k, ef=ef)
    def save(self): pass
    async def save_async(self): pass
    def __len__(self): return 0

# ── Build app with mocks ─────────────────────────────────────────────────────
with patch("src.api.main.Embedder", return_value=_FakeEmbedder()), \
     patch("src.api.main.DocumentStore", return_value=_FakeStore()), \
     patch("src.api.main.IndexManager", return_value=_FakeIndexManager()):
    from src.api.main import app

client = TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "index_size" in data
    assert "dim" in data


def test_ingest_and_search():
    # Ingest
    r = client.post("/ingest", json={"text": "The quick brown fox", "metadata": {"src": "test"}})
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] is True
    assert isinstance(data["id"], int)
    doc_id = data["id"]

    # Search
    r2 = client.post("/search", json={"query": "quick fox", "k": 5})
    assert r2.status_code == 200
    resp = r2.json()
    assert "hits" in resp
    assert isinstance(resp["hits"], list)
    assert resp["latency_ms"] > 0


def test_batch_ingest():
    r = client.post("/batch_ingest", json={
        "documents": [
            {"text": "Document one", "metadata": {}},
            {"text": "Document two", "metadata": {"tag": "test"}},
        ]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] == 2
    assert data["failed"] == 0


def test_delete_document():
    r = client.post("/ingest", json={"text": "Delete me", "metadata": {}})
    doc_id = r.json()["id"]
    r2 = client.delete(f"/document/{doc_id}")
    assert r2.status_code == 200
    assert r2.json()["deleted"] == doc_id


def test_delete_missing():
    r = client.delete("/document/999999")
    assert r.status_code == 404


def test_search_empty_query_rejected():
    r = client.post("/search", json={"query": "", "k": 5})
    assert r.status_code == 422


def test_ingest_empty_text_rejected():
    r = client.post("/ingest", json={"text": "", "metadata": {}})
    assert r.status_code == 422
