"""
DocumentStore — SQLAlchemy-backed persistence for full text + metadata.
Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
from sqlalchemy import (
    Column, Integer, String, Text, LargeBinary, create_engine, text
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

logger = logging.getLogger("hvs.store")
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id         = Column(Integer, primary_key=True, index=True)
    text       = Column(Text,    nullable=False)
    metadata_  = Column("metadata", Text, default="{}")
    embedding  = Column(LargeBinary)          # float32 bytes
    deleted    = Column(Integer, default=0)   # soft-delete flag


class DocumentStore:
    def __init__(self, db_url: str = "sqlite:///./hvs.db"):
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args, echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info(f"DocumentStore ready — {db_url}")

    # ── Sequence counter ──────────────────────────────────────────────────────
    def _next_id(self, session: Session) -> int:
        row = session.execute(
            text("SELECT COALESCE(MAX(id), 0) + 1 FROM documents")
        ).scalar()
        return int(row)

    def upsert(
        self,
        doc_id: Optional[int],
        text_: str,
        metadata: dict[str, Any],
        embedding: np.ndarray,
    ) -> int:
        with self._Session() as session:
            if doc_id is None:
                doc_id = self._next_id(session)
            existing = session.get(Document, doc_id)
            emb_bytes = embedding.astype(np.float32).tobytes()
            if existing:
                existing.text       = text_
                existing.metadata_  = json.dumps(metadata)
                existing.embedding  = emb_bytes
                existing.deleted    = 0
            else:
                session.add(Document(
                    id=doc_id,
                    text=text_,
                    metadata_=json.dumps(metadata),
                    embedding=emb_bytes,
                ))
            session.commit()
        return doc_id

    def get(self, doc_id: int) -> Optional[dict[str, Any]]:
        with self._Session() as session:
            doc = session.get(Document, doc_id)
            if doc is None or doc.deleted:
                return None
            return {
                "id":       doc.id,
                "text":     doc.text,
                "metadata": json.loads(doc.metadata_ or "{}"),
            }

    def get_embedding(self, doc_id: int) -> Optional[np.ndarray]:
        with self._Session() as session:
            doc = session.get(Document, doc_id)
            if doc is None or doc.embedding is None:
                return None
            return np.frombuffer(doc.embedding, dtype=np.float32)

    def delete(self, doc_id: int) -> bool:
        with self._Session() as session:
            doc = session.get(Document, doc_id)
            if doc is None:
                return False
            doc.deleted = 1
            session.commit()
            return True

    def count(self) -> int:
        with self._Session() as session:
            return session.execute(
                text("SELECT COUNT(*) FROM documents WHERE deleted=0")
            ).scalar()

    def iter_all(self, batch_size: int = 1000):
        """Yield (id, embedding) tuples for index rebuild."""
        with self._Session() as session:
            offset = 0
            while True:
                rows = session.execute(
                    text("SELECT id, embedding FROM documents WHERE deleted=0 LIMIT :lim OFFSET :off"),
                    {"lim": batch_size, "off": offset}
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    emb = np.frombuffer(row[1], dtype=np.float32) if row[1] else None
                    if emb is not None:
                        yield int(row[0]), emb
                offset += batch_size
