#!/usr/bin/env python3
"""Evalua retrieval FAISS con consultas humanas y relevancia por comando."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def load_command_aliases(corpus_dir: Path, indexed_aliases: set[str]) -> dict[str, set[str]]:
    by_command: dict[str, set[str]] = defaultdict(set)
    for path in sorted(corpus_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload["category"])
        for row in payload.get("documents", []):
            alias = f"{category}:{row['id']}"
            command = str(row.get("command", "")).split(maxsplit=1)[0]
            if alias in indexed_aliases and command:
                by_command[command].add(alias)
    return by_command


def dcg(relevance: list[bool]) -> float:
    return sum(1 / math.log2(position + 2) for position, hit in enumerate(relevance) if hit)


def evaluate(queries: list[dict], index, aliases: np.ndarray, encoder, command_aliases, k_values: tuple[int, ...]):
    scores, positions = index.search(encoder.encode(
        [item["query"] for item in queries], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32), max(k_values))
    records = []
    for item, row_scores, row_positions in zip(queries, scores, positions, strict=True):
        relevant = set().union(*(command_aliases[command] for command in item["relevant_commands"]))
        ranked = [str(aliases[position]) for position in row_positions]
        hits = [alias in relevant for alias in ranked]
        first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
        metrics = {}
        for k in k_values:
            at_k = hits[:k]
            metrics[f"command_hit@{k}"] = float(any(at_k))
            metrics[f"ndcg@{k}"] = dcg(at_k) / dcg([True] * min(k, len(relevant)))
        metrics["mrr"] = 1 / first_rank if first_rank else 0.0
        records.append({
            "id": item["id"], "language": item["language"], "query": item["query"],
            "relevant_commands": item["relevant_commands"], "relevant_aliases": len(relevant),
            "first_relevant_rank": first_rank, "top_5": ranked[:5], "top_5_scores": [round(float(score), 4) for score in row_scores[:5]],
            **metrics,
        })
    return records


def summarize(records: list[dict], k_values: tuple[int, ...]) -> dict:
    summary = {"queries": len(records)}
    for k in k_values:
        summary[f"command_hit@{k}"] = round(sum(row[f"command_hit@{k}"] for row in records) / len(records), 4)
        summary[f"ndcg@{k}"] = round(sum(row[f"ndcg@{k}"] for row in records) / len(records), 4)
    summary["mrr"] = round(sum(row["mrr"] for row in records) / len(records), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("evaluation/retrieval_queries.json"))
    parser.add_argument("--corpus", type=Path, default=Path("dataset_builder/corpus"))
    parser.add_argument("--index", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"))
    parser.add_argument("--aliases", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/retrieval_results.json"))
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    index = faiss.read_index(str(args.index))
    aliases = np.load(args.aliases, allow_pickle=False).astype(str)
    if index.ntotal != len(aliases):
        raise ValueError("El indice y aliases no estan alineados")
    command_aliases = load_command_aliases(args.corpus, set(aliases))
    for item in queries:
        missing = [command for command in item["relevant_commands"] if not command_aliases[command]]
        if missing:
            raise ValueError(f"{item['id']}: comandos no indexados: {', '.join(missing)}")

    k_values = (1, 3, 5, 10)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    records = evaluate(queries, index, aliases, model, command_aliases, k_values)
    report = {"index_vectors": int(index.ntotal), "metrics": summarize(records, k_values), "queries": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
