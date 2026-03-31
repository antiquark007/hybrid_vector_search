#!/usr/bin/env python3
"""
scripts/load_msmarco.py — Download and index MS MARCO 100k passages.

Usage:
    python scripts/load_msmarco.py --limit 100000 --batch 256 --api http://localhost:8000
    python scripts/load_msmarco.py --celery --limit 100000   # via Redis queue
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("load_msmarco")

MSMARCO_URLS = (
    # Legacy endpoint kept for backward compatibility where still accessible.
    "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
    # Public static mirror used by the MS MARCO team.
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
)
DATA_DIR = Path("./data")


def download_msmarco(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tsv_path = dest / "collection.tsv"
    if tsv_path.exists():
        logger.info(f"Already downloaded: {tsv_path}")
        return tsv_path
    gz_path = dest / "collection.tar.gz"
    errors = []
    for url in MSMARCO_URLS:
        logger.info(f"Downloading MS MARCO collection from {url} to {gz_path}...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            break
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")
            logger.warning(f"Download failed from {url}: {exc}")
    else:
        raise RuntimeError(
            "Unable to download MS MARCO collection from any known source. "
            + " | ".join(errors)
        )

    import tarfile
    with tarfile.open(gz_path, "r:gz") as tar:
        tar.extractall(dest)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Expected collection file missing after extract: {tsv_path}")
    logger.info("Download complete.")
    return tsv_path


def iter_passages(tsv_path: Path, limit: int) -> Iterator[dict]:
    with open(tsv_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts
            yield {"id": int(pid), "text": text.strip(), "metadata": {"source": "msmarco"}}


def load_via_api(passages, api_url: str, batch_size: int):
    endpoint = f"{api_url}/batch_ingest"
    total, failed = 0, 0
    batch = []

    def flush():
        nonlocal total, failed
        if not batch:
            return
        try:
            r = requests.post(endpoint, json={"documents": batch}, timeout=60)
            r.raise_for_status()
            data = r.json()
            total  += data["indexed"]
            failed += data["failed"]
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            failed += len(batch)
        batch.clear()

    t0 = time.time()
    for doc in passages:
        batch.append(doc)
        if len(batch) >= batch_size:
            flush()
            elapsed = time.time() - t0
            logger.info(f"Indexed {total} | Failed {failed} | {total/elapsed:.0f} doc/s")
    flush()
    logger.info(f"Done — indexed {total}, failed {failed}, elapsed {time.time()-t0:.1f}s")


def load_via_celery(passages, batch_size: int):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.ingestion.worker import ingest_batch
    batch = []

    def flush():
        if not batch:
            return
        ingest_batch.delay([dict(b) for b in batch])
        batch.clear()

    total = 0
    for doc in passages:
        batch.append(doc)
        total += 1
        if len(batch) >= batch_size:
            flush()
            logger.info(f"Queued {total} documents...")
    flush()
    logger.info(f"All {total} documents queued to Celery.")


def main():
    parser = argparse.ArgumentParser(description="Load MS MARCO into HVS")
    parser.add_argument("--limit",    type=int, default=100_000)
    parser.add_argument("--batch",    type=int, default=256)
    parser.add_argument("--api",      type=str, default=None,
                        help="FastAPI base URL, e.g. http://localhost:8000")
    parser.add_argument("--celery",   action="store_true",
                        help="Send tasks to Celery queue instead of direct API")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tsv_path = download_msmarco(data_dir)
    passages = iter_passages(tsv_path, args.limit)

    if args.celery:
        load_via_celery(passages, args.batch)
    elif args.api:
        load_via_api(passages, args.api, args.batch)
    else:
        logger.error("Specify --api <url> or --celery")
        sys.exit(1)


if __name__ == "__main__":
    main()
