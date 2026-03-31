#!/usr/bin/env python3
"""
scripts/benchmark.py — end-to-end latency & throughput benchmark.

Usage:
    python scripts/benchmark.py --api http://localhost:8000 --n 500
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import string
import time

import requests


def random_query(length=6):
    words = [
        "machine learning", "neural network", "transformer", "attention",
        "retrieval", "embedding", "vector", "search", "BERT", "GPT",
        "semantic", "passage", "query", "document", "ranking", "score",
        "similarity", "cosine", "index", "approximate",
    ]
    return " ".join(random.choices(words, k=length))


def single_query(api_url: str, k: int = 10) -> float:
    t0 = time.perf_counter()
    requests.post(
        f"{api_url}/search",
        json={"query": random_query(), "k": k},
        timeout=10,
    )
    return (time.perf_counter() - t0) * 1000


def run_benchmark(api_url: str, n: int, k: int, concurrency: int):
    print(f"\nHybrid Vector Search — Benchmark")
    print(f"API: {api_url}  N={n}  k={k}  concurrency={concurrency}\n")

    # Warmup
    print("Warming up (10 queries)...")
    for _ in range(10):
        single_query(api_url, k)

    latencies = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(single_query, api_url, k) for _ in range(n)]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                latencies.append(fut.result())
            except Exception as e:
                print(f"  Query {i} failed: {e}")

    elapsed = time.perf_counter() - t_start
    qps = len(latencies) / elapsed
    sorted_lat = sorted(latencies)

    results = {
        "n_queries":       len(latencies),
        "QPS":             round(qps, 1),
        "latency_p50_ms":  round(statistics.median(latencies), 2),
        "latency_p95_ms":  round(sorted_lat[int(len(latencies) * 0.95)], 2),
        "latency_p99_ms":  round(sorted_lat[int(len(latencies) * 0.99)], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_min_ms":  round(min(latencies), 2),
        "latency_max_ms":  round(max(latencies), 2),
        "elapsed_s":       round(elapsed, 2),
    }

    print("=" * 44)
    for k_, v in results.items():
        print(f"  {k_:<22} {v}")
    print("=" * 44)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",         default="http://localhost:8000")
    parser.add_argument("--n",           type=int, default=200, help="Number of queries")
    parser.add_argument("--k",           type=int, default=10,  help="Top-K per query")
    parser.add_argument("--concurrency", type=int, default=8,   help="Parallel workers")
    parser.add_argument("--out",         type=str, default=None)
    args = parser.parse_args()

    results = run_benchmark(args.api, args.n, args.k, args.concurrency)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {args.out}")


if __name__ == "__main__":
    main()
