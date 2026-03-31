#!/usr/bin/env python3
"""
Generate sample MS MARCO-like evaluation data for testing.
Creates synthetic queries and qrels files.

Usage:
    python scripts/generate_sample_eval_data.py
"""

import csv
import random
from pathlib import Path

# Sample query templates
QUERIES = [
    "what is machine learning",
    "graph-based nearest neighbor search",
    "how do embeddings work",
    "vector database comparison",
    "approximate nearest neighbor algorithms",
    "HNSW indexing tutorial",
    "cosine similarity calculation",
    "sentence transformers models",
    "semantic search explained",
    "high-dimensional vector search",
    "information retrieval ranking",
    "document similarity scoring",
    "efficient similarity search",
    "embedding dimension reduction",
    "parallel search optimization",
]

# Sample relevant documents (synthetic IDs)
RELEVANT_DOCS = {
    0: [1, 5, 12, 23, 45, 67, 89, 102, 150, 201],
    1: [2, 8, 15, 34, 56, 78, 92, 110, 156, 203],
    2: [3, 9, 18, 42, 63, 84, 99, 120, 167, 210],
    3: [4, 11, 22, 50, 71, 88, 105, 135, 178, 220],
    4: [6, 14, 28, 58, 75, 91, 112, 145, 189, 230],
    5: [7, 17, 35, 65, 82, 98, 122, 155, 199, 240],
    6: [10, 20, 40, 70, 87, 104, 130, 165, 209, 250],
    7: [13, 25, 48, 77, 93, 108, 140, 175, 215, 260],
    8: [16, 30, 55, 85, 100, 115, 150, 185, 225, 270],
    9: [19, 38, 62, 95, 116, 128, 160, 195, 235, 280],
}

def generate_queries(output_path: Path, n_queries: int = 100):
    """Generate queries file in MS MARCO format: qid \t query"""
    with open(output_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid in range(n_queries):
            query = random.choice(QUERIES)
            writer.writerow([qid, query])
    print(f"✅ Generated {n_queries} queries → {output_path}")

def generate_qrels(output_path: Path, n_queries: int = 100):
    """Generate qrels file in MS MARCO format: qid \t 0 \t pid \t relevance"""
    with open(output_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid in range(n_queries):
            # Get relevant docs for this query
            relevant_docs = RELEVANT_DOCS.get(qid % len(RELEVANT_DOCS), [qid * 10, qid * 10 + 1])
            
            # Write relev ance judgements
            for pid in relevant_docs:
                # Binary relevance: 1 or 0
                rel = random.choice([1, 1, 1, 0])  # 75% relevant
                writer.writerow([qid, 0, pid, rel])
    
    print(f"✅ Generated qrels → {output_path}")

def main():
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    queries_path = data_dir / "queries.dev.small.tsv"
    qrels_path = data_dir / "qrels.dev.small.tsv"
    
    # Generate 100 sample queries and qrels
    generate_queries(queries_path, n_queries=100)
    generate_qrels(qrels_path, n_queries=100)
    
    print("\n✅ Sample evaluation data generated!")
    print(f"   Queries: {queries_path}")
    print(f"   Qrels:   {qrels_path}")
    print("\nNow you can run: make eval")

if __name__ == "__main__":
    main()
