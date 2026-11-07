#!/usr/bin/env python3
"""Compare FAISS, BM25, and hybrid retrieval on reviewed capacity queries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from pyserini.search.lucene import LuceneSearcher

from evaluation.evaluate_retrieval import (
    dcg,
    load_command_aliases,
    summarize,
)
from semantic_search_engine.retrieval_service.bm25_search import HybridBM25Retriever
from semantic_search_engine.retrieval_service.retriever import SentenceTransformerEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITIES = ROOT / "evaluation/capacity/plain_source.yaml"
DEFAULT_CORPUS = ROOT / "dataset_builder/corpus"
DEFAULT_INDEX = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"
DEFAULT_ALIASES = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"
DEFAULT_BM25_INDEX = ROOT / "dataset_builder/indexer/artifacts/bm25_indexs_v2"
DEFAULT_OUTPUT = ROOT / "evaluation/capacity/retrieval_comparison.json"
EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"
K_VALUES = (1, 3, 5, 10)
HYBRID_ALPHA_VALUES = (0.25, 0.5, 0.75)
SELECTION_METRICS = ("command_hit@10", "mrr", "ndcg@10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_commands(corpus_dir: Path) -> dict[str, str]:
    source_commands: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload["category"])
        for document in payload.get("documents", []):
            alias = f"{category}:{document['id']}"
            command = str(document.get("command", "")).split(maxsplit=1)
            if not command:
                continue
            if alias in source_commands:
                raise ValueError(f"Corpus contains duplicate alias {alias}")
            source_commands[alias] = command[0]
    return source_commands


def load_capacity_queries(
    capacities_path: Path,
    corpus_dir: Path,
    indexed_aliases: set[str],
) -> list[dict[str, Any]]:
    payload = yaml.safe_load(capacities_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Capacity file must be an object with schema_version 1")
    if payload.get("metadata", {}).get("query_review_status") != "reviewed":
        raise ValueError("Capacity queries must be marked reviewed before evaluation")

    source_commands = load_source_commands(corpus_dir)
    records: list[dict[str, Any]] = []
    capability_ids: set[str] = set()
    query_ids: set[str] = set()
    query_texts: set[str] = set()

    for group in payload.get("categories", []):
        category = group.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("Every category requires a non-empty name")
        for capability in group.get("capacities", []):
            capability_id = capability.get("capability_id")
            if not isinstance(capability_id, str) or not capability_id:
                raise ValueError("Every capacity requires a non-empty capability_id")
            if capability_id in capability_ids:
                raise ValueError(f"Duplicate capability_id: {capability_id}")
            capability_ids.add(capability_id)

            partition = capability.get("partition")
            if partition not in {"development", "test"}:
                raise ValueError(f"{capability_id}: invalid partition {partition!r}")

            aliases = capability.get("source_aliases")
            if not isinstance(aliases, list) or not aliases:
                raise ValueError(f"{capability_id}: source_aliases must not be empty")
            relevant_commands = set()
            for alias in aliases:
                if alias not in source_commands:
                    raise ValueError(f"{capability_id}: unknown corpus alias {alias!r}")
                if alias not in indexed_aliases:
                    raise ValueError(f"{capability_id}: alias {alias!r} is not in the FAISS index")
                relevant_commands.add(source_commands[alias])

            queries = capability.get("queries")
            if not isinstance(queries, list) or not queries:
                raise ValueError(f"{capability_id}: queries must not be empty")
            for position, query in enumerate(queries, start=1):
                if query.get("status") != "reviewed":
                    raise ValueError(f"{capability_id}: query {position} is not marked reviewed")
                text = query.get("text")
                language = query.get("language")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{capability_id}: query {position} has no text")
                if language not in {"es", "en"}:
                    raise ValueError(f"{capability_id}: query {position} has invalid language")

                query_id = f"{capability_id}::{position}"
                normalized_text = text.strip().casefold()
                if query_id in query_ids:
                    raise ValueError(f"Duplicate query id: {query_id}")
                if normalized_text in query_texts:
                    raise ValueError(f"Duplicate query text: {text}")
                query_ids.add(query_id)
                query_texts.add(normalized_text)
                records.append({
                    "id": query_id,
                    "query": text.strip(),
                    "language": language,
                    "partition": partition,
                    "category": category,
                    "capability_id": capability_id,
                    "relevant_commands": sorted(relevant_commands),
                })

    if not records:
        raise ValueError("Capacity file contains no queries")
    return records


def group_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return {
        name: summarize(group_records, K_VALUES)
        for name, group_records in sorted(grouped.items())
    }


def index_hashes(index_dir: Path) -> dict[str, str]:
    files = sorted(
        path for path in index_dir.rglob("*")
        if path.is_file() and path.name != "write.lock"
    )
    if not files:
        raise ValueError(f"No BM25 index files found in {index_dir}")
    return {
        path.relative_to(index_dir).as_posix(): sha256_file(path)
        for path in files
    }


def score_rankings(
    queries: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    command_aliases: dict[str, set[str]],
    method: str,
) -> list[dict[str, Any]]:
    records = []
    for query in queries:
        relevant = set().union(
            *(command_aliases[command] for command in query["relevant_commands"])
        )
        ranked_aliases = rankings[query["id"]]
        hits = [alias in relevant for alias in ranked_aliases]
        first_relevant_rank = next(
            (rank for rank, hit in enumerate(hits, start=1) if hit),
            None,
        )
        metrics = {}
        for k in K_VALUES:
            at_k = hits[:k]
            ideal = dcg([True] * min(k, len(relevant)))
            metrics[f"command_hit@{k}"] = float(any(at_k))
            metrics[f"ndcg@{k}"] = dcg(at_k) / ideal if ideal else 0.0
        metrics["mrr"] = 1 / first_relevant_rank if first_relevant_rank else 0.0
        records.append({
            **query,
            "method": method,
            "top_5": ranked_aliases[:5],
            "first_relevant_rank": first_relevant_rank,
            **metrics,
        })
    return records


def summarize_method(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({record["category"] for record in records})
    return {
        "metrics": summarize(records, K_VALUES),
        "partition_metrics": group_metrics(records, "partition"),
        "language_metrics": group_metrics(records, "language"),
        "category_partition_metrics": {
            category: group_metrics(
                [record for record in records if record["category"] == category],
                "partition",
            )
            for category in categories
        },
        "queries": records,
    }


def rank_faiss(
    queries: list[dict[str, Any]],
    index,
    aliases: np.ndarray,
    encoder: SentenceTransformerEncoder,
    top_k: int,
) -> tuple[dict[str, list[str]], float]:
    started_at = time.perf_counter()
    vectors = encoder.encode([query["query"] for query in queries])
    _, positions = index.search(vectors, top_k)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    rankings = {
        query["id"]: [
            str(aliases[position])
            for position in row
            if position >= 0
        ]
        for query, row in zip(queries, positions, strict=True)
    }
    return rankings, elapsed_ms / len(queries)


def rank_bm25(
    queries: list[dict[str, Any]],
    searcher: LuceneSearcher,
    top_k: int,
) -> tuple[dict[str, list[str]], float]:
    rankings: dict[str, list[str]] = {}
    started_at = time.perf_counter()
    for query in queries:
        hits = searcher.search(query["query"], k=top_k)
        ranked_aliases = []
        for hit in hits:
            try:
                alias = json.loads(hit.raw)["id"]
            except (TypeError, json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"BM25 result for {query['id']} has no JSON raw id"
                ) from error
            ranked_aliases.append(str(alias))
        rankings[query["id"]] = ranked_aliases
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return rankings, elapsed_ms / len(queries)


def rank_hybrid(
    queries: list[dict[str, Any]],
    retriever: HybridBM25Retriever,
    top_k: int,
) -> tuple[dict[str, list[str]], float]:
    rankings: dict[str, list[str]] = {}
    started_at = time.perf_counter()
    for query in queries:
        rankings[query["id"]] = [
            alias for alias, _, _ in retriever.search(query["query"], top_k)
        ]
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return rankings, elapsed_ms / len(queries)


def select_hybrid_alpha(
    alpha_summaries: dict[float, dict[str, Any]],
) -> float:
    if not alpha_summaries:
        raise ValueError("No development alpha results are available")
    return max(
        alpha_summaries,
        key=lambda alpha: (
            *(alpha_summaries[alpha][metric] for metric in SELECTION_METRICS),
            -abs(alpha - 0.5),
            -alpha,
        ),
    )


def run_method(
    queries: list[dict[str, Any]],
    command_aliases: dict[str, set[str]],
    ranker,
    method: str,
) -> tuple[dict[str, Any], float]:
    rankings, latency_ms = ranker(queries)
    records = score_rankings(queries, rankings, command_aliases, method)
    return summarize_method(records), latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=Path, default=DEFAULT_CAPACITIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--candidate-top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.candidate_top_k < max(K_VALUES):
        parser.error(f"--candidate-top-k must be at least {max(K_VALUES)}")

    index = faiss.read_index(str(args.index))
    aliases = np.load(args.aliases, allow_pickle=False).astype(str)
    if index.ntotal != len(aliases):
        raise ValueError("FAISS index and aliases are not aligned")

    queries = load_capacity_queries(args.capacities, args.corpus, set(aliases))
    command_aliases = load_command_aliases(args.corpus, set(aliases))
    for item in queries:
        missing = [
            command for command in item["relevant_commands"]
            if not command_aliases.get(command)
        ]
        if missing:
            raise ValueError(
                f"{item['id']}: relevant commands are absent from FAISS: {', '.join(missing)}"
            )

    development = [query for query in queries if query["partition"] == "development"]
    test = [query for query in queries if query["partition"] == "test"]
    if not development or not test:
        raise ValueError("Both development and test partitions must contain queries")

    searcher = LuceneSearcher(str(args.bm25_index))
    expected_bm25_docs = len(load_source_commands(args.corpus))
    if searcher.num_docs != expected_bm25_docs:
        raise ValueError(
            f"BM25 index has {searcher.num_docs} docs but corpus has "
            f"{expected_bm25_docs}; rebuild or select the matching index"
        )
    encoder = SentenceTransformerEncoder(EMBEDDER_NAME)
    hybrid_retrievers = {
        alpha: HybridBM25Retriever(
            searcher,
            index,
            aliases,
            encoder,
            args.candidate_top_k,
            alpha,
        )
        for alpha in HYBRID_ALPHA_VALUES
    }

    def evaluate_methods(
        selected_queries: list[dict[str, Any]],
        include_alpha_sweep: bool,
    ) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
        methods: dict[str, Any] = {}
        latencies: dict[str, float] = {}
        faiss_result, latencies["faiss"] = run_method(
            selected_queries,
            command_aliases,
            lambda items: rank_faiss(
                items, index, aliases, encoder, args.candidate_top_k
            ),
            "faiss",
        )
        methods["faiss"] = faiss_result
        bm25_result, latencies["bm25"] = run_method(
            selected_queries,
            command_aliases,
            lambda items: rank_bm25(items, searcher, args.candidate_top_k),
            "bm25",
        )
        methods["bm25"] = bm25_result

        hybrid_results: dict[float, dict[str, Any]] = {}
        alphas = HYBRID_ALPHA_VALUES if include_alpha_sweep else (selected_alpha,)
        for alpha in alphas:
            result, latencies[f"hybrid_alpha_{alpha:g}"] = run_method(
                selected_queries,
                command_aliases,
                lambda items, current_alpha=alpha: rank_hybrid(
                    items,
                    hybrid_retrievers[current_alpha],
                    args.candidate_top_k,
                ),
                f"hybrid_alpha_{alpha:g}",
            )
            hybrid_results[alpha] = result
        return {
            "methods": methods,
            "hybrid_alpha_sweep": {
                f"{alpha:g}": value for alpha, value in hybrid_results.items()
            },
            "hybrid_alpha_sweep_performed": include_alpha_sweep,
            "latency_ms_per_query": latencies,
        }, hybrid_results

    development_results, development_hybrids = evaluate_methods(
        development,
        include_alpha_sweep=True,
    )
    alpha_summaries = {
        alpha: result["metrics"]
        for alpha, result in development_hybrids.items()
    }
    selected_alpha = select_hybrid_alpha(alpha_summaries)
    development_results["hybrid_selection"] = {
        "selected_alpha": selected_alpha,
        "selection_rule": (
            "maximize development command_hit@10, then MRR, then ndcg@10; "
            "ties prefer alpha closest to 0.5, then the lower alpha"
        ),
        "selection_metrics": alpha_summaries,
        "test_partition_used_for_selection": False,
    }

    test_results, _ = evaluate_methods(test, include_alpha_sweep=False)
    test_results["selected_hybrid_alpha"] = selected_alpha
    report_queries = {
        "development": len(development),
        "test": len(test),
        "total": len(queries),
    }

    corpus_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(args.corpus.glob("*.yaml"))
    }
    report = {
        "schema_version": 1,
        "kind": "capacity_retrieval_comparison",
        "relevance_unit": (
            "command: every indexed corpus alias whose command matches a capacity "
            "source alias is relevant"
        ),
        "candidate_top_k": args.candidate_top_k,
        "query_counts": report_queries,
        "development": development_results,
        "test": test_results,
        "provenance": {
            "capacities_sha256": sha256_file(args.capacities),
            "faiss_index_sha256": sha256_file(args.index),
            "aliases_sha256": sha256_file(args.aliases),
            "corpus_sha256": corpus_hashes,
            "embedding_model": EMBEDDER_NAME,
            "bm25_index_directory": args.bm25_index.resolve().relative_to(ROOT).as_posix(),
            "bm25_index_representation": (
                "legacy command-only representation (bm25_indexs_v2)"
            ),
            "bm25_index_sha256": index_hashes(args.bm25_index),
            "pyserini_version": importlib.metadata.version("pyserini"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "query_counts": report["query_counts"],
        "selected_hybrid_alpha": selected_alpha,
        "development": {
            "faiss": development_results["methods"]["faiss"]["metrics"],
            "bm25": development_results["methods"]["bm25"]["metrics"],
            "hybrid_alpha_sweep": {
                alpha: result["metrics"]
                for alpha, result in development_results["hybrid_alpha_sweep"].items()
            },
        },
        "test": {
            "faiss": test_results["methods"]["faiss"]["metrics"],
            "bm25": test_results["methods"]["bm25"]["metrics"],
            f"hybrid_alpha_{selected_alpha:g}": test_results["hybrid_alpha_sweep"][
                f"{selected_alpha:g}"
            ]["metrics"],
        },
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
