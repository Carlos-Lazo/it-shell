#!/usr/bin/env python3
"""Compare Debian BM25 indexes with and without intent paraphrases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from pyserini.search.lucene import LuceneSearcher

from semantic_search_engine.retrieval_service.bm25_search import HybridBM25Retriever
from semantic_search_engine.retrieval_service.retriever import SentenceTransformerEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "evaluation" / "evaluation_v2_queries.json"
DEFAULT_BASE_INDEX = ROOT / "dataset_builder" / "indexer" / "artifacts" / "bm25_debian12_enriched_base_lucene9"
DEFAULT_PARAPHRASE_INDEX = ROOT / "dataset_builder" / "indexer" / "artifacts" / "bm25_debian12_enriched_paraphrases_lucene9"
DEFAULT_FAISS_INDEX = ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "index.faiss"
DEFAULT_ALIASES = ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "aliases.npy"
DEFAULT_OUTPUT = ROOT / "evaluation" / "bm25_paraphrase_comparison_debian12.json"
EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"
K_VALUES = (1, 3, 5, 10)
TOP_K = 15
TARGET_CATEGORIES = ("filesystem", "networking")
DEBIAN_CATEGORIES = (
    "c programming",
    "cryptography",
    "filesystem",
    "hardwaremanagement",
    "keyboard",
    "misc",
    "networking",
    "processmanagement",
    "security",
    "shellprogramming",
    "textprocessing",
    "usergroup",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_index(path: Path) -> dict[str, str]:
    files = sorted(
        file for file in path.rglob("*")
        if file.is_file() and file.name != "write.lock"
    )
    if not files:
        raise ValueError(f"No se encontraron archivos Lucene en {path}")
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in files
    }


def load_relevance(corpus_dir: Path) -> dict[str, set[str]]:
    aliases_by_command: dict[str, set[str]] = defaultdict(set)
    seen_aliases: set[str] = set()
    for path in sorted(corpus_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload.get("category", ""))
        for row in payload.get("documents", []):
            alias = f"{category}:{row['id']}"
            if alias in seen_aliases:
                raise ValueError(f"Alias duplicado en corpus: {alias}")
            seen_aliases.add(alias)
            command = str(row.get("command", "")).split(maxsplit=1)
            if command:
                aliases_by_command[command[0]].add(alias)
    if not seen_aliases:
        raise ValueError(f"No se encontraron documentos YAML en {corpus_dir}")
    return aliases_by_command


def parse_hit_alias(hit: Any) -> str:
    try:
        alias = json.loads(hit.raw)["id"]
    except (TypeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("Un resultado BM25 no contiene un JSON raw con 'id'.") from error
    return str(alias)


def dcg(relevance: list[bool]) -> float:
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, relevant in enumerate(relevance)
        if relevant
    )


def metrics(ranked_aliases: list[str], relevant_aliases: set[str]) -> dict[str, float]:
    hits = [alias in relevant_aliases for alias in ranked_aliases[:TOP_K]]
    first_rank = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    result: dict[str, float] = {}
    for k in K_VALUES:
        ideal_dcg = dcg([True] * min(k, len(relevant_aliases)))
        result[f"hit@{k}"] = float(any(hits[:k]))
        result[f"ndcg@{k}"] = dcg(hits[:k]) / ideal_dcg if ideal_dcg else 0.0
    result["mrr"] = 1.0 / first_rank if first_rank else 0.0
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("No hay consultas que resumir.")
    summary: dict[str, float | int] = {"queries": len(records)}
    for metric in ("hit@1", "hit@3", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr"):
        summary[metric] = round(
            statistics.mean(record[metric] for record in records), 4
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Directory containing exactly the Debian 12 YAML corpus categories.",
    )
    parser.add_argument("--base-index", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument(
        "--paraphrase-index", type=Path, default=DEFAULT_PARAPHRASE_INDEX
    )
    parser.add_argument("--faiss-index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha debe estar entre 0 y 1")
    if args.base_index.resolve() == args.paraphrase_index.resolve():
        parser.error("base-index y paraphrase-index deben ser distintos")

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [query for query in queries if query.get("category") in TARGET_CATEGORIES]
    query_counts = {
        category: sum(query["category"] == category for query in queries)
        for category in TARGET_CATEGORIES
    }
    if any(count == 0 for count in query_counts.values()):
        raise ValueError(f"Faltan consultas por categoría: {query_counts}")

    aliases_by_command = load_relevance(args.corpus)
    relevant_by_query: dict[str, set[str]] = {}
    for query in queries:
        relevant = set().union(
            *(aliases_by_command.get(command, set())
              for command in query["relevant_commands"])
        )
        if not relevant:
            raise ValueError(f"Sin aliases relevantes para consulta {query['id']}")
        relevant_by_query[query["id"]] = relevant

    searchers = {
        "base": LuceneSearcher(str(args.base_index)),
        "paraphrases": LuceneSearcher(str(args.paraphrase_index)),
    }
    indexed_counts = {name: searcher.num_docs for name, searcher in searchers.items()}
    if len(set(indexed_counts.values())) != 1:
        raise ValueError(f"Los índices no tienen igual cantidad de documentos: {indexed_counts}")
    if next(iter(indexed_counts.values())) <= 0:
        raise ValueError(f"Índices BM25 vacíos: {indexed_counts}")

    faiss_index = faiss.read_index(str(args.faiss_index))
    dense_aliases = np.load(args.aliases, allow_pickle=False).astype(str)
    if faiss_index.ntotal != len(dense_aliases):
        raise ValueError("El índice FAISS y aliases.npy no están alineados.")
    encoder = SentenceTransformerEncoder(EMBEDDER_NAME)
    hybrid = {
        name: HybridBM25Retriever(
            searcher,
            faiss_index,
            dense_aliases,
            encoder,
            top_k=TOP_K,
            alpha=args.alpha,
            allowed_categories=DEBIAN_CATEGORIES,
        )
        for name, searcher in searchers.items()
    }

    per_query: dict[str, dict[str, Any]] = {}
    raw_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {variant: [] for variant in searchers}
        for mode in ("bm25", "hybrid")
    }
    latency: dict[str, dict[str, list[float]]] = {
        mode: {variant: [] for variant in searchers}
        for mode in ("bm25", "hybrid")
    }

    for query in queries:
        query_id = query["id"]
        relevant = relevant_by_query[query_id]
        detail: dict[str, Any] = {
            "id": query_id,
            "category": query["category"],
            "query": query["query"],
            "relevant_commands": query["relevant_commands"],
            "relevant_alias_count": len(relevant),
            "variants": {},
        }
        for variant, searcher in searchers.items():
            started = time.perf_counter()
            bm25_aliases = [
                parse_hit_alias(hit)
                for hit in searcher.search(query["query"], k=TOP_K)
            ]
            latency["bm25"][variant].append(time.perf_counter() - started)
            bm25_metrics = metrics(bm25_aliases, relevant)
            raw_metrics["bm25"][variant].append(
                {"id": query_id, "category": query["category"], **bm25_metrics}
            )

            started = time.perf_counter()
            hybrid_results = hybrid[variant].retrieve(query["query"], TOP_K)
            latency["hybrid"][variant].append(time.perf_counter() - started)
            hybrid_aliases = [alias for alias, _ in hybrid_results]
            hybrid_metrics = metrics(hybrid_aliases, relevant)
            raw_metrics["hybrid"][variant].append(
                {"id": query_id, "category": query["category"], **hybrid_metrics}
            )
            detail["variants"][variant] = {
                "bm25_top_10": bm25_aliases[:10],
                "bm25": bm25_metrics,
                "hybrid_top_10": hybrid_aliases[:10],
                "hybrid": hybrid_metrics,
            }
        for mode in ("bm25", "hybrid"):
            base = detail["variants"]["base"][mode]
            paraphrases = detail["variants"]["paraphrases"][mode]
            detail[f"{mode}_delta"] = {
                metric: round(paraphrases[metric] - base[metric], 4)
                for metric in base
            }
        per_query[query_id] = detail

    summaries: dict[str, Any] = {}
    for mode, variants in raw_metrics.items():
        summaries[mode] = {}
        for variant, records in variants.items():
            by_category = {
                category: summarize(
                    [record for record in records if record["category"] == category]
                )
                for category in TARGET_CATEGORIES
            }
            summaries[mode][variant] = {
                "overall": summarize(records),
                "by_category": by_category,
                "mean_latency_ms": round(
                    statistics.mean(latency[mode][variant]) * 1000, 2
                ),
            }
        summaries[mode]["delta_paraphrases_minus_base"] = {
            "overall": {
                metric: round(
                    summaries[mode]["paraphrases"]["overall"][metric]
                    - summaries[mode]["base"]["overall"][metric],
                    4,
                )
                for metric in ("hit@1", "hit@3", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr")
            },
            "by_category": {
                category: {
                    metric: round(
                        summaries[mode]["paraphrases"]["by_category"][category][metric]
                        - summaries[mode]["base"]["by_category"][category][metric],
                        4,
                    )
                    for metric in ("hit@1", "hit@3", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr")
                }
                for category in TARGET_CATEGORIES
            },
        }

    index_hashes = {
        variant: sha256_index(path)
        for variant, path in (
            ("base", args.base_index),
            ("paraphrases", args.paraphrase_index),
        )
    }
    corpus_hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.corpus.glob("*.yaml"))
    }
    if not corpus_hashes:
        raise ValueError(f"No se encontraron archivos YAML en {args.corpus}")
    try:
        java_version = subprocess.run(
            ["java", "-version"],
            check=True,
            capture_output=True,
            text=True,
        ).stderr.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise RuntimeError("No se pudo determinar la versión local de Java") from error
    report = {
        "schema_version": 1,
        "kind": "debian12_bm25_paraphrase_comparison",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyserini": importlib.metadata.version("pyserini"),
            "java": java_version,
        },
        "query_count": len(queries),
        "query_counts_by_category": query_counts,
        "document_count_per_index": indexed_counts,
        "categories": list(DEBIAN_CATEGORIES),
        "hybrid_alpha_dense_weight": args.alpha,
        "metrics": summaries,
        "inputs": {
            "queries": {
                "path": str(args.queries),
                "sha256": sha256_file(args.queries),
            },
            "corpus_dir": str(args.corpus),
            "corpus_files_sha256": corpus_hashes,
            "faiss_index_sha256": sha256_file(args.faiss_index),
            "aliases_sha256": sha256_file(args.aliases),
            "bm25_index_sha256": index_hashes,
        },
        "queries": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Consultas por categoría: {query_counts}")
    print(f"Documentos por índice: {indexed_counts}")
    for mode in ("bm25", "hybrid"):
        print(f"\n{mode.upper()}")
        for variant in ("base", "paraphrases"):
            result = summaries[mode][variant]["overall"]
            print(
                f"{variant:12} Hit@1={result['hit@1']:.4f} "
                f"Hit@5={result['hit@5']:.4f} "
                f"Hit@10={result['hit@10']:.4f} "
                f"MRR={result['mrr']:.4f}"
            )
        print("delta:", summaries[mode]["delta_paraphrases_minus_base"]["overall"])
        for category in TARGET_CATEGORIES:
            print(
                category,
                summaries[mode]["delta_paraphrases_minus_base"]["by_category"][category],
            )
    print(f"\nInforme guardado: {args.output}")


if __name__ == "__main__":
    main()
