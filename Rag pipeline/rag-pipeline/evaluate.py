"""
evaluate.py
Runs a labeled eval set of (question, expected_source_doc_id) pairs
against the pipeline and reports retrieval precision@k and the
confidence-gate refusal rate. This is what turns the resume claims
("Y% retrieval precision@k") into numbers you actually measured.

Eval set format (JSON): a list of objects:
[
  {"question": "...", "expected_doc_id": "policy.pdf"},
  ...
]

Usage:
  python evaluate.py --source data/sample_docs --eval-set data/eval_set.json
"""

from __future__ import annotations

import argparse
import json
import statistics

from src.pipeline import RAGPipeline


def precision_at_k(retrieved_doc_ids: list[str], expected_doc_id: str) -> float:
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d == expected_doc_id)
    return hits / len(retrieved_doc_ids)


def run_eval(source_dir: str, eval_set_path: str, top_k: int = 5) -> dict:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    pipeline = RAGPipeline()
    pipeline.build_index(source_dir)

    precisions, latencies, refusals = [], [], 0

    for item in eval_set:
        result = pipeline.query(item["question"], top_k=top_k)
        retrieved_doc_ids = [
            c["chunk_id"].split("::chunk-")[0] for c in result["retrieved_chunks"]
        ]
        precisions.append(precision_at_k(retrieved_doc_ids, item["expected_doc_id"]))
        latencies.append(result["latency_seconds"])
        if not result["grounded"]:
            refusals += 1

    return {
        "n_queries": len(eval_set),
        "mean_precision_at_k": round(statistics.mean(precisions), 4) if precisions else 0.0,
        "mean_latency_seconds": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "refusal_rate": round(refusals / len(eval_set), 4) if eval_set else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline retrieval quality")
    parser.add_argument("--source", default="data/sample_docs")
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    metrics = run_eval(args.source, args.eval_set, args.top_k)
    print(json.dumps(metrics, indent=2))
