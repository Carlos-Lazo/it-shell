#!/usr/bin/env python3
"""Evalua retrieval FAISS con consultas humanas y relevancia por comando."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(
    manifest_path: Path,
    queries_path: Path,
    corpus_dir: Path,
    index_path: Path,
    aliases_path: Path,
) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_artifacts = manifest["artifacts"]
    expected_corpus = manifest["corpus"]["directory"]
    if corpus_dir.resolve().relative_to(ROOT).as_posix() != expected_corpus:
        raise ValueError(f"El corpus no coincide con el snapshot: esperado {expected_corpus}")
    expected_queries = manifest["evaluation"]["queries_file"]
    if queries_path.resolve().relative_to(ROOT).as_posix() != expected_queries:
        raise ValueError(f"El conjunto de consultas no coincide con el snapshot: esperado {expected_queries}")

    input_files = [
        queries_path,
        index_path,
        aliases_path,
        *sorted(corpus_dir.glob("*.yaml")),
    ]
    for path in input_files:
        relative_path = path.resolve().relative_to(ROOT).as_posix()
        expected_hash = expected_artifacts.get(relative_path)
        if expected_hash is None or sha256_file(path) != expected_hash:
            raise ValueError(f"El artefacto no coincide con el snapshot: {relative_path}")
    return sha256_file(manifest_path)


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


def evaluate(
    queries: list[dict],
    index,
    aliases: np.ndarray,
    encoder,
    command_aliases,
    k_values: tuple[int, ...],
    candidate_top_k: int,
):
    scores, positions = index.search(encoder.encode(
        [item["query"] for item in queries], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32), candidate_top_k)
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
            "id": item["id"], "language": item.get("language", "en"), "query": item["query"],
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
    parser.add_argument("--queries", type=Path, default=Path("evaluation/evaluation_v2_queries.json"))
    parser.add_argument("--corpus", type=Path, default=Path("dataset_builder/corpus"))
    parser.add_argument("--index", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"))
    parser.add_argument("--aliases", type=Path, default=Path("dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"))
    parser.add_argument("--candidate-top-k", type=int, default=15)
    parser.add_argument("--snapshot-manifest", type=Path, default=Path("evaluation/retrieval_snapshot_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/retrieval_benchmark_v2.json"))
    args = parser.parse_args()

    k_values = (1, 3, 5, 10)
    if args.candidate_top_k < max(k_values):
        parser.error(f"--candidate-top-k debe ser al menos {max(k_values)}")
    snapshot_sha256 = validate_snapshot(
        args.snapshot_manifest,
        args.queries,
        args.corpus,
        args.index,
        args.aliases,
    )
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

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    started_at = time.perf_counter()
    records = evaluate(queries, index, aliases, model, command_aliases, k_values, args.candidate_top_k)
    elapsed_per_query = (time.perf_counter() - started_at) * 1000 / len(queries)
    report = {
        "index_vectors": int(index.ntotal),
        "candidate_top_k": args.candidate_top_k,
        "metrics": summarize(records, k_values),
        "latency_ms": {"retrieval_per_query": round(elapsed_per_query, 2)},
        "provenance": {
            "snapshot_manifest_sha256": snapshot_sha256,
            "queries_sha256": sha256_file(args.queries),
            "faiss_index_sha256": sha256_file(args.index),
            "aliases_sha256": sha256_file(args.aliases),
        },
        "queries": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
