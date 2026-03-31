#!/usr/bin/env python3
"""
src/evaluation/evaluate.py

Evaluate the search engine against MS MARCO queries.

Usage:
    # Generate sample data first (recommended)
    python scripts/generate_sample_eval_data.py
    
    # Then run evaluation
    python -m src.evaluation.evaluate \
        --api http://localhost:8000 \
        --queries data/queries.dev.small.tsv \
        --qrels  data/qrels.dev.small.tsv \
        --k 10 --n 1000
    
    # Or with shorthand from Makefile
    make generate-eval-data
    make eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hvs.eval")

QRELS_URL   = "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv"
QUERIES_URL = "https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv"


def download_if_missing(url: str, dest: Path) -> Path:
    if not dest.exists():
        logger.info(f"Downloading {url} → {dest}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            logger.error(f"Failed to download from {url}: {e}")
            logger.error(f"Please manually download and place at {dest}")
            logger.error("Or provide --queries and --qrels arguments with local file paths")
            raise
    return dest


def load_qrels(path: Path) -> dict[int, set[int]]:
    """Returns {query_id: set of relevant doc_ids}"""
    qrels: dict[int, set[int]] = defaultdict(set)
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            qid, _, pid, rel = row
            if int(rel) > 0:
                qrels[int(qid)].add(int(pid))
    return dict(qrels)


def load_queries(path: Path) -> dict[int, str]:
    queries: dict[int, str] = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                queries[int(row[0])] = row[1]
    return queries


def recall_at_k(result_ids: list[int], relevant: set[int], k: int) -> float:
    top_k = set(result_ids[:k])
    return len(top_k & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(result_ids: list[int], relevant: set[int]) -> float:
    for rank, rid in enumerate(result_ids, 1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def run_evaluation(
    api_url: str,
    queries: dict[int, str],
    qrels: dict[int, set[int]],
    k: int = 10,
    ef: int = 50,
    n: int | None = None,
) -> dict:
    endpoint = f"{api_url}/search"
    recalls, rrs, latencies = [], [], []
    errors = 0

    query_items = list(queries.items())
    if n:
        query_items = query_items[:n]
    # Only evaluate queries that have relevance judgements
    query_items = [(qid, q) for qid, q in query_items if qid in qrels]

    logger.info(f"Evaluating {len(query_items)} queries (k={k}, ef={ef})")

    for qid, query_text in query_items:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                endpoint,
                json={"query": query_text, "k": k, "ef": ef},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Query {qid} failed: {e}")
            errors += 1
            continue

        latency_ms = (time.perf_counter() - t0) * 1000
        result_ids = [h["id"] for h in data.get("hits", [])]
        relevant   = qrels.get(qid, set())

        recalls.append(recall_at_k(result_ids, relevant, k))
        rrs.append(reciprocal_rank(result_ids, relevant))
        latencies.append(latency_ms)

    if not recalls:
        logger.error("No successful queries!")
        return {}

    results = {
        "n_queries":       len(recalls),
        "n_errors":        errors,
        f"Recall@{k}":     round(statistics.mean(recalls), 4),
        "MRR":             round(statistics.mean(rrs), 4),
        "latency_p50_ms":  round(statistics.median(latencies), 2),
        "latency_p95_ms":  round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "latency_p99_ms":  round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "QPS":             round(len(latencies) / (sum(latencies) / 1000), 1),
    }

    print("\n" + "=" * 50)
    print("  Hybrid Vector Search — Evaluation Results")
    print("=" * 50)
    for k_, v in results.items():
        print(f"  {k_:<22} {v}")
    print("=" * 50 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",     default="http://localhost:8000")
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--qrels",   type=str, default=None)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--ef",      type=int, default=50)
    parser.add_argument("--n",       type=int, default=None, help="Max queries to eval")
    parser.add_argument("--out",     type=str, default=None,  help="Save JSON results here")
    args = parser.parse_args()

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    qrels_path   = Path(args.qrels)   if args.qrels   else download_if_missing(QRELS_URL,   data_dir / "qrels.dev.small.tsv")
    queries_path = Path(args.queries) if args.queries  else download_if_missing(QUERIES_URL, data_dir / "queries.dev.small.tsv")

    qrels   = load_qrels(qrels_path)
    queries = load_queries(queries_path)

    results = run_evaluation(args.api, queries, qrels, k=args.k, ef=args.ef, n=args.n)

    if args.out and results:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
