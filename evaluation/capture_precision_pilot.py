#!/usr/bin/env python3
"""Capture structured end-to-end outputs for the deterministic precision pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from semantic_search_engine.bootstrap import create_engine
from semantic_search_engine.shared.config import load_config
from semantic_search_engine.shared.contracts import QueryRequest
from evaluation.evaluate_retrieval import validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "evaluation/precision_pilot_queries.json"
DEFAULT_CONFIG = ROOT / "semantic_search_engine/config/default.yaml"
DEFAULT_OUTPUT = ROOT / "evaluation/precision_pilot_baseline.json"
SNAPSHOT = ROOT / "evaluation/retrieval_snapshot_v2.json"
CORPUS = ROOT / "dataset_builder/corpus"
FAISS_INDEX = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"
ALIASES = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_index_files(index_dir: Path) -> dict[str, str]:
    index_dir = index_dir.resolve()
    files = sorted(
        path for path in index_dir.rglob("*")
        if path.is_file() and path.name != "write.lock"
    )
    if not files:
        raise ValueError(f"No se encontraron archivos de índice en {index_dir}")
    return {
        path.relative_to(index_dir).as_posix(): sha256_file(path)
        for path in files
    }


def resident_memory_mib() -> float:
    with Path("/proc/self/status").open(encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("Could not read resident memory from /proc/self/status")


def select_queries(
    payload: Any,
    splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    queries = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries must contain a non-empty JSON array")
    if splits is None:
        return queries
    if not isinstance(payload, dict):
        raise ValueError("Split filtering requires benchmark cases with split metadata")
    if not all(
        isinstance(item, dict) and isinstance(item.get("split"), str)
        for item in queries
    ):
        raise ValueError("Every benchmark case must have split metadata")
    selected = [item for item in queries if item["split"] in splits]
    if not selected:
        raise ValueError(f"No benchmark cases found for splits: {', '.join(sorted(splits))}")
    return selected


def capture(
    queries_path: Path,
    config_path: Path,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    queries_path = queries_path.resolve()
    config_path = config_path.resolve()
    query_payload = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = select_queries(query_payload, splits)

    retrieval_scope = query_payload.get("retrieval_scope") if isinstance(query_payload, dict) else None
    config = load_config(config_path)
    if retrieval_scope is not None:
        expected_categories = retrieval_scope.get("categories")
        if not isinstance(expected_categories, list) or not expected_categories:
            raise ValueError("Benchmark retrieval_scope must contain non-empty categories")
        if config.allowed_categories is None or set(config.allowed_categories) != set(expected_categories):
            raise ValueError(
                "Engine config allowed_categories must match benchmark retrieval_scope.categories"
            )
    elif config.allowed_categories is not None:
        raise ValueError(
            "A domain-filtered engine config requires benchmark retrieval_scope metadata"
        )

    snapshot_sha256 = validate_snapshot(
        SNAPSHOT,
        ROOT / "evaluation/evaluation_v2_queries.json",
        CORPUS,
        FAISS_INDEX,
        ALIASES,
    )
    bm25_index_files = (
        hash_index_files(config.bm25_index_path)
        if config.retrieval_mode == "hybrid" and config.bm25_index_path is not None
        else None
    )
    engine = create_engine(config_path)
    startup_memory_mib = resident_memory_mib()
    startup_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    records = []
    for item in queries:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("id", "area", "query", "expected_policy")
        ):
            raise ValueError("Every pilot case requires id, area, query, and expected_policy")
        started = time.perf_counter()
        response = engine.search_response(QueryRequest(item["query"]))
        elapsed_ms = (time.perf_counter() - started) * 1000
        records.append({
            **item,
            "latency_ms": round(elapsed_ms, 3),
            "response": response.to_dict(),
        })
    loaded_memory_mib = resident_memory_mib()
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    try:
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Could not identify the repository revision") from error

    return {
        "schema_version": 1,
        "kind": (
            "end_to_end_precision_baseline"
            if isinstance(query_payload, dict)
            else "exploratory_baseline_pilot"
        ),
        "scoring_status": (
            "not scored: benchmark annotations must be adjudicated before metrics are valid"
        ),
        "revision": revision,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "configuration": {
            "path": config_path.relative_to(ROOT).as_posix()
            if config_path.is_relative_to(ROOT)
            else str(config_path),
            "sha256": sha256_file(config_path),
        },
        "retrieval_artifacts": {
            "mode": config.retrieval_mode,
            "hybrid_alpha": config.hybrid_alpha if config.retrieval_mode == "hybrid" else None,
            "bm25_index_path": (
                config.bm25_index_path.relative_to(ROOT).as_posix()
                if config.bm25_index_path and config.bm25_index_path.is_relative_to(ROOT)
                else str(config.bm25_index_path) if config.bm25_index_path else None
            ),
            "bm25_index_files": bm25_index_files,
            "bm25_search_phrases_path": (
                config.bm25_search_phrases_path.relative_to(ROOT).as_posix()
                if config.bm25_search_phrases_path
                and config.bm25_search_phrases_path.is_relative_to(ROOT)
                else str(config.bm25_search_phrases_path)
                if config.bm25_search_phrases_path
                else None
            ),
            "bm25_search_phrases_sha256": (
                sha256_file(config.bm25_search_phrases_path)
                if config.bm25_search_phrases_path
                else None
            ),
        },
        "retrieval_scope": retrieval_scope or {
            "name": "unrestricted",
            "categories": None,
        },
        "snapshot": {
            "path": SNAPSHOT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SNAPSHOT),
            "validated_sha256": snapshot_sha256,
        },
        "runtime_measurements": {
            "rss_after_engine_startup_mib": round(startup_memory_mib, 3),
            "rss_after_pilot_mib": round(loaded_memory_mib, 3),
            "peak_rss_kib": peak_rss_kib,
            "peak_rss_after_engine_startup_kib": startup_peak_rss_kib,
        },
        "inputs": {
            "queries_path": queries_path.relative_to(ROOT).as_posix()
            if queries_path.is_relative_to(ROOT)
            else str(queries_path),
            "queries_sha256": sha256_file(queries_path),
            "count": len(queries),
            "splits": sorted(splits) if splits is not None else "all",
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "holdout"),
        action="append",
        dest="splits",
        help="Capture only the named benchmark partition; repeat to select multiple.",
    )
    args = parser.parse_args()

    report = capture(
        args.queries,
        args.config,
        set(args.splits) if args.splits else None,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "revision": report["revision"],
        "cases": report["inputs"]["count"],
        "scoring_status": report["scoring_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
