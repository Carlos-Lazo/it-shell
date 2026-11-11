#!/usr/bin/env python3
"""Compare MPNet and BGE-M3 on the reviewed capacity queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.evaluate_capacity_retrieval import (
    K_VALUES,
    load_capacity_queries,
    score_rankings,
    sha256_file,
    summarize_method,
)
from evaluation.evaluate_retrieval import load_command_aliases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITIES = ROOT / "evaluation/capacity/plain_source.yaml"
DEFAULT_CORPUS = ROOT / "dataset_builder/corpus"
DEFAULT_OUTPUT = ROOT / "evaluation/capacity/embedding_model_comparison.json"
DEFAULT_SNAPSHOT = ROOT / "evaluation/retrieval_snapshot_v2.json"
DEFAULT_TOP_K = 10
MODELS = {
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "index": ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss",
        "aliases": ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "index": ROOT / "dataset_builder/indexer/artifacts/bge-m3/index.faiss",
        "aliases": ROOT / "dataset_builder/indexer/artifacts/bge-m3/aliases.npy",
    },
}


def load_index(index_path: Path, aliases_path: Path):
    index = faiss.read_index(str(index_path))
    aliases = np.load(aliases_path, allow_pickle=False).astype(str)
    if index.ntotal != len(aliases):
        raise ValueError(f"FAISS index and aliases are not aligned: {index_path}")
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"Aliases are not unique: {aliases_path}")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(f"Expected inner-product FAISS index: {index_path}")
    return index, aliases


def corpus_matches_snapshot(
    corpus_hashes: dict[str, str],
    snapshot_path: Path,
) -> bool:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_corpus_hashes = {
        path: digest
        for path, digest in snapshot.get("artifacts", {}).items()
        if path.startswith("dataset_builder/corpus/") and path.endswith(".yaml")
    }
    return bool(snapshot_corpus_hashes) and corpus_hashes == snapshot_corpus_hashes


def rank_in_alias_universe(
    index,
    vectors: np.ndarray,
    aliases: np.ndarray,
    allowed_aliases: set[str],
    top_k: int,
) -> dict[int, list[str]]:
    if top_k > len(allowed_aliases):
        raise ValueError("top_k exceeds the shared alias universe")
    candidate_k = min(index.ntotal, max(top_k, 2 * top_k))
    while True:
        _, positions = index.search(vectors, candidate_k)
        rankings = {}
        for query_position, row in enumerate(positions):
            ranking = [
                str(aliases[position])
                for position in row
                if position >= 0 and str(aliases[position]) in allowed_aliases
            ][:top_k]
            rankings[query_position] = ranking
        if all(len(ranking) == top_k for ranking in rankings.values()):
            return rankings
        if candidate_k == index.ntotal:
            raise ValueError("Index search returned too few shared aliases")
        candidate_k = min(index.ntotal, candidate_k * 2)


def evaluate_model(
    model_key: str,
    model_config: dict[str, Any],
    queries: list[dict[str, Any]],
    aliases: np.ndarray,
    index,
    common_aliases: set[str],
    command_aliases: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    model_started = time.perf_counter()
    model = SentenceTransformer(model_config["name"])
    model_load_seconds = time.perf_counter() - model_started

    encode_started = time.perf_counter()
    vectors = model.encode(
        [query["query"] for query in queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    encode_seconds = time.perf_counter() - encode_started
    if vectors.ndim != 2 or vectors.shape != (len(queries), index.d):
        raise ValueError(
            f"{model_key} produced {vectors.shape} embeddings for an index "
            f"with dimension {index.d}"
        )

    search_started = time.perf_counter()
    rankings_by_position = rank_in_alias_universe(
        index, vectors, aliases, common_aliases, top_k
    )
    search_ms_per_query = (
        (time.perf_counter() - search_started) * 1000 / len(queries)
    )
    rankings = {
        query["id"]: rankings_by_position[position]
        for position, query in enumerate(queries)
    }
    records = score_rankings(queries, rankings, command_aliases, model_key)
    by_partition = {}
    for partition in ("development", "test"):
        partition_records = [
            record for record in records if record["partition"] == partition
        ]
        if not partition_records:
            raise ValueError(f"No queries found in the {partition} partition")
        summary = summarize_method(partition_records)
        by_partition[partition] = {
            "metrics": summary["metrics"],
            "language_metrics": summary["language_metrics"],
            "category_partition_metrics": summary["category_partition_metrics"],
            "queries": summary["queries"],
        }

    return {
        "model": model_config["name"],
        "embedding_dimension": int(vectors.shape[1]),
        "device": str(model.device),
        "model_load_seconds": round(model_load_seconds, 3),
        "query_encoding_seconds": round(encode_seconds, 3),
        "query_encoding_ms_per_query": round(encode_seconds * 1000 / len(queries), 3),
        "shared_index_search_and_filter_ms_per_query": round(search_ms_per_query, 3),
        "partitions": by_partition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=Path, default=DEFAULT_CAPACITIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k < max(K_VALUES):
        parser.error(f"--top-k must be at least {max(K_VALUES)}")

    indexes = {}
    alias_sets = {}
    for key, config in MODELS.items():
        indexes[key] = load_index(config["index"], config["aliases"])
        alias_sets[key] = set(indexes[key][1])

    common_aliases = set.intersection(*alias_sets.values())
    if not common_aliases:
        raise ValueError("The model indexes have no aliases in common")

    queries = load_capacity_queries(args.capacities, args.corpus, common_aliases)
    command_aliases = load_command_aliases(args.corpus, common_aliases)
    for query in queries:
        absent_commands = [
            command for command in query["relevant_commands"]
            if not command_aliases.get(command)
        ]
        if absent_commands:
            raise ValueError(
                f"{query['id']}: relevant commands absent from the shared index: "
                f"{', '.join(absent_commands)}"
            )

    model_results = {}
    for key, config in MODELS.items():
        index, aliases = indexes[key]
        model_results[key] = evaluate_model(
            key,
            config,
            queries,
            aliases,
            index,
            common_aliases,
            command_aliases,
            args.top_k,
        )

    corpus_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(args.corpus.glob("*.yaml"))
    }
    mpnet_snapshot_matches = corpus_matches_snapshot(
        corpus_hashes, DEFAULT_SNAPSHOT
    )
    alignment_note = (
        "Current corpus YAML hashes match the MPNet snapshot manifest. "
        if mpnet_snapshot_matches
        else "Current corpus YAML hashes do not match the MPNet snapshot manifest. "
    ) + (
        "BGE aliases align to the shared alias universe, but its vector artifact "
        "does not record source corpus hashes, so exact source text revision for "
        "BGE cannot be independently verified."
    )
    common_alias_hash = hashlib.sha256(
        "\n".join(sorted(common_aliases)).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "kind": "capacity_embedding_model_comparison",
        "relevance_unit": (
            "command: every alias in the shared index whose command matches a "
            "capacity source alias is relevant"
        ),
        "top_k": args.top_k,
        "query_counts": {
            partition: sum(query["partition"] == partition for query in queries)
            for partition in ("development", "test")
        } | {"total": len(queries)},
        "language_query_counts": {
            language: sum(query["language"] == language for query in queries)
            for language in ("en", "es")
        },
        "shared_alias_universe": {
            "aliases": len(common_aliases),
            "sha256_sorted_aliases": common_alias_hash,
            "excluded_aliases_by_model": {
                key: len(alias_sets[key] - common_aliases) for key in MODELS
            },
        },
        "test_partition_used_for_selection": False,
        "models": model_results,
        "provenance": {
            "capacities_sha256": sha256_file(args.capacities),
            "corpus_sha256": corpus_hashes,
            "mpnet_snapshot_sha256": sha256_file(DEFAULT_SNAPSHOT),
            "current_corpus_matches_mpnet_snapshot": mpnet_snapshot_matches,
            "indexes": {
                key: {
                    "index_sha256": sha256_file(config["index"]),
                    "aliases_sha256": sha256_file(config["aliases"]),
                    "dimension": int(indexes[key][0].d),
                }
                for key, config in MODELS.items()
            },
            "bge_vectors_sha256": sha256_file(
                ROOT / "dataset_builder/embedder/artifacts/bge_m3_vectors.npz"
            ),
            "bge_source_corpus_hashes_recorded": False,
            "corpus_alignment_note": alignment_note,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "query_counts": report["query_counts"],
        "language_query_counts": report["language_query_counts"],
        "shared_aliases": report["shared_alias_universe"]["aliases"],
        "development": {
            key: value["partitions"]["development"]["metrics"]
            for key, value in model_results.items()
        },
        "test": {
            key: value["partitions"]["test"]["metrics"]
            for key, value in model_results.items()
        },
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
