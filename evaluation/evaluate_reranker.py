#!/usr/bin/env python3
"""Benchmark comparativo de Cross-Encoder Reranker contra Baseline FAISS."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer


def load_corpus_metadata(corpus_dir: Path, indexed_aliases: set[str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    by_command: dict[str, set[str]] = defaultdict(set)
    alias_intents: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload.get("category", ""))
        for row in payload.get("documents", []):
            alias = f"{category}:{row['id']}"
            command = str(row.get("command", "")).split(maxsplit=1)[0]
            if alias in indexed_aliases and command:
                by_command[command].add(alias)
                alias_intents[alias] = str(row.get("intent", ""))
    return by_command, alias_intents


def dcg(relevance: list[bool]) -> float:
    return sum(1 / math.log2(position + 2) for position, hit in enumerate(relevance) if hit)


def compute_metrics(hits: list[bool], relevant_count: int, k_values: tuple[int, ...]) -> dict[str, float]:
    first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    metrics: dict[str, float] = {}
    for k in k_values:
        at_k = hits[:k]
        metrics[f"command_hit@{k}"] = float(any(at_k))
        ideal_dcg = dcg([True] * min(k, relevant_count))
        metrics[f"ndcg@{k}"] = (dcg(at_k) / ideal_dcg) if ideal_dcg > 0 else 0.0
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0
    return metrics


def summarize(records: list[dict], k_values: tuple[int, ...]) -> dict[str, float]:
    n = len(records)
    summary: dict[str, float] = {"queries": n}
    for k in k_values:
        summary[f"command_hit@{k}"] = round(sum(row[f"command_hit@{k}"] for row in records) / n, 4)
        summary[f"ndcg@{k}"] = round(sum(row[f"ndcg@{k}"] for row in records) / n, 4)
    summary["mrr"] = round(sum(row["mrr"] for row in records) / n, 4)
    return summary


def run_benchmark(
    queries: list[dict],
    index,
    aliases: np.ndarray,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    command_aliases: dict[str, set[str]],
    alias_intents: dict[str, str],
    initial_top_k: int = 50,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    t0 = time.perf_counter()
    query_texts = [q["query"] for q in queries]
    query_vectors = embedder.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    faiss_scores, faiss_positions = index.search(query_vectors, initial_top_k)
    retrieval_latency = (time.perf_counter() - t0) / len(queries)

    baseline_records = []
    reranked_records = []
    reranker_times = []

    for item, pos_row, score_row in zip(queries, faiss_positions, faiss_scores, strict=True):
        query_text = item["query"]
        relevant = set().union(*(command_aliases[cmd] for cmd in item["relevant_commands"]))
        ranked_aliases = [str(aliases[pos]) for pos in pos_row]

        # 1. Baseline metrics (orden directo de FAISS)
        baseline_hits = [alias in relevant for alias in ranked_aliases]
        baseline_m = compute_metrics(baseline_hits, len(relevant), k_values)
        baseline_records.append({"id": item["id"], "query": query_text, **baseline_m})

        # 2. Re-ranking con Cross-Encoder
        t_r0 = time.perf_counter()
        pairs = [(query_text, alias_intents.get(alias, "")) for alias in ranked_aliases]
        cross_scores = reranker.predict(pairs)
        reranker_times.append(time.perf_counter() - t_r0)

        # Ordenar candidatos por score de cross-encoder descendente
        reranked_pairs = sorted(zip(ranked_aliases, cross_scores), key=lambda x: float(x[1]), reverse=True)
        reranked_aliases = [alias for alias, _ in reranked_pairs]

        reranked_hits = [alias in relevant for alias in reranked_aliases]
        reranked_m = compute_metrics(reranked_hits, len(relevant), k_values)
        reranked_records.append({
            "id": item["id"],
            "query": query_text,
            "relevant_commands": item["relevant_commands"],
            "top_5_baseline": ranked_aliases[:5],
            "top_5_reranked": reranked_aliases[:5],
            **reranked_m,
        })

    avg_rerank_latency = sum(reranker_times) / len(reranker_times)

    return {
        "initial_candidates": initial_top_k,
        "latency_ms": {
            "retrieval_per_query": round(retrieval_latency * 1000, 2),
            "rerank_per_query": round(avg_rerank_latency * 1000, 2),
            "total_per_query": round((retrieval_latency + avg_rerank_latency) * 1000, 2),
        },
        "baseline_summary": summarize(baseline_records, k_values),
        "reranked_summary": summarize(reranked_records, k_values),
        "queries_detail": reranked_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evalua Cross-Encoder Reranker contra Baseline.")
    parser.add_argument("--queries", type=Path, default=Path("evaluation/retrieval_queries.json"))
    parser.add_argument("--corpus", type=Path, default=Path("dataset_builder/corpus"))
    parser.add_argument("--index", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"))
    parser.add_argument("--aliases", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/reranker_benchmark_results.json"))
    parser.add_argument("--initial-top-k", type=int, default=50, help="Cantidad de candidatos iniciales de FAISS para reordenar")
    parser.add_argument("--cross-encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    index = faiss.read_index(str(args.index))
    aliases = np.load(args.aliases, allow_pickle=False).astype(str)
    command_aliases, alias_intents = load_corpus_metadata(args.corpus, set(aliases))

    print(f"Cargando Embedder: sentence-transformers/all-mpnet-base-v2...")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print(f"Cargando Reranker: {args.cross_encoder}...")
    reranker = CrossEncoder(args.cross_encoder)

    print(f"Ejecutando benchmark comparativo (Top-{args.initial_top_k} inicial)...")
    results = run_benchmark(
        queries, index, aliases, embedder, reranker, command_aliases, alias_intents, initial_top_k=args.initial_top_k
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Formatear tabla comparativa en consola
    base = results["baseline_summary"]
    rerk = results["reranked_summary"]
    print("\n" + "=" * 65)
    print(" RESULTADOS COMPARATIVOS: BASELINE (FAISS) vs RERANKER (CROSS-ENCODER)")
    print("=" * 65)
    print(f"{'Métrica':<18} | {'Baseline v1.0':<15} | {'Con Reranker':<15} | {'Delta':<10}")
    print("-" * 65)
    for m in ["command_hit@1", "command_hit@3", "command_hit@5", "command_hit@10", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "mrr"]:
        b_val = base[m]
        r_val = rerk[m]
        diff = r_val - b_val
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        if "hit" in m:
            print(f"{m:<18} | {b_val*100:>13.2f}% | {r_val*100:>13.2f}% | {diff*100:>+8.2f}%")
        else:
            print(f"{m:<18} | {b_val:>15.4f} | {r_val:>15.4f} | {diff_str:>10}")
    print("-" * 65)
    lat = results["latency_ms"]
    print(f"Latencia promedio: Retrieval: {lat['retrieval_per_query']} ms | Rerank: {lat['rerank_per_query']} ms | Total: {lat['total_per_query']} ms")
    print("=" * 65)
    print(f"Resultados exportados a: {args.output}\n")


if __name__ == "__main__":
    main()
