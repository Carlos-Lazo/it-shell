#!/usr/bin/env python3
"""
evaluate_bm25_hybrid.py
-----------------------
Benchmark comparativo BM25 puro y BM25+Dense híbrido contra el Baseline FAISS.

Produce un JSON con el mismo esquema que reranker_benchmark_v2_top15.json:
  {
    "initial_candidates": ...,
    "latency_ms": { "retrieval_per_query": ..., "rerank_per_query": ..., "total_per_query": ... },
    "baseline_summary": { "queries": ..., "command_hit@k": ..., "ndcg@k": ..., "mrr": ... },
    "reranked_summary": { ... },          # aquí = métricas BM25 o híbrido
    "queries_detail": [ ... ]
  }

Uso (desde la raíz del repo, con el venv del indexer):

  # Solo BM25
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \\
  dataset_builder/indexer/.venv/bin/python evaluation/evaluate_bm25_hybrid.py \\
    --mode bm25 \\
    --queries evaluation/evaluation_v2_queries.json \\
    --output evaluation/bm25_benchmark_top15.json

  # Híbrido BM25 + Dense (alpha = 0.5)
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \\
  dataset_builder/indexer/.venv/bin/python evaluation/evaluate_bm25_hybrid.py \\
    --mode hybrid \\
    --alpha 0.5 \\
    --queries evaluation/evaluation_v2_queries.json \\
    --output evaluation/hybrid_bm25_dense_top15.json
"""

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
from pyserini.search.lucene import LuceneSearcher

from semantic_search_engine.retrieval_service.bm25_search import HybridBM25Retriever
from semantic_search_engine.retrieval_service.retriever import SentenceTransformerEncoder
# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERIES  = _ROOT / "evaluation" / "evaluation_v2_queries.json"
DEFAULT_CORPUS   = _ROOT / "dataset_builder" / "corpus"
DEFAULT_BM25     = _ROOT / "dataset_builder" / "indexer" / "artifacts" / "bm25_indexs_v2"
DEFAULT_FAISS    = _ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "index.faiss"
DEFAULT_ALIASES  = _ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "aliases.npy"
DEFAULT_SNAPSHOT = _ROOT / "evaluation" / "retrieval_snapshot_v2.json"
DEFAULT_OUT      = _ROOT / "evaluation" / "bm25_benchmark_top15.json"
EMBEDDER_NAME    = "sentence-transformers/all-mpnet-base-v2"

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
INITIAL_TOP_K = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_index_files(index_dir: Path) -> dict[str, str]:
    index_dir = index_dir.resolve()
    files = sorted(path for path in index_dir.rglob("*") if path.is_file() and path.name != "write.lock")
    if not files:
        raise ValueError(f"No se encontraron archivos de índice en {index_dir}")
    return {
        path.relative_to(_ROOT).as_posix(): sha256_file(path)
        for path in files
    }


def validate_snapshot(
    manifest_path: Path,
    queries_path: Path,
    faiss_path: Path,
    aliases_path: Path,
    bm25_path: Path,
) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_bm25_dir = manifest["bm25"]["directory"]
    actual_bm25_dir = bm25_path.resolve().relative_to(_ROOT).as_posix()
    if actual_bm25_dir != expected_bm25_dir:
        raise ValueError(
            f"El índice BM25 no coincide con el snapshot: esperado {expected_bm25_dir}, "
            f"recibido {actual_bm25_dir}"
        )
    expected_files = manifest["bm25"]["files"]
    actual_files = hash_index_files(bm25_path)
    if expected_files != actual_files:
        raise ValueError("Los checksums del índice BM25 no coinciden con el manifiesto")
    checks = (
        (queries_path, manifest["evaluation"]["sha256"], "consultas"),
        (faiss_path, manifest["artifacts"][faiss_path.resolve().relative_to(_ROOT).as_posix()], "índice FAISS"),
        (aliases_path, manifest["artifacts"][aliases_path.resolve().relative_to(_ROOT).as_posix()], "aliases"),
    )
    for path, expected_hash, label in checks:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"El checksum de {label} no coincide con el snapshot")
    return sha256_file(manifest_path)


# ---------------------------------------------------------------------------
# Helpers de corpus
# ---------------------------------------------------------------------------
def load_corpus_metadata(
    corpus_dir: Path, indexed_aliases: set[str]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Devuelve:
      by_command  → {command_token: {alias1, alias2, …}}
      alias_intents → {alias: intent}
    """
    by_command: dict[str, set[str]] = defaultdict(set)
    alias_intents: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload.get("category", ""))
        for row in payload.get("documents", []):
            alias   = f"{category}:{row['id']}"
            command = str(row.get("command", "")).split(maxsplit=1)[0]
            if alias in indexed_aliases and command:
                by_command[command].add(alias)
                alias_intents[alias] = str(row.get("intent", ""))
    return by_command, alias_intents


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def dcg(relevance: list[bool]) -> float:
    return sum(1.0 / math.log2(pos + 2) for pos, hit in enumerate(relevance) if hit)


def compute_metrics(hits: list[bool], relevant_count: int) -> dict[str, float]:
    first_rank = next((r for r, h in enumerate(hits, 1) if h), None)
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        at_k = hits[:k]
        metrics[f"command_hit@{k}"] = float(any(at_k))
        ideal = dcg([True] * min(k, relevant_count))
        metrics[f"ndcg@{k}"] = (dcg(at_k) / ideal) if ideal > 0 else 0.0
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0
    return metrics


def summarize(records: list[dict]) -> dict[str, float]:
    n = len(records)
    summary: dict[str, float] = {"queries": n}
    for k in K_VALUES:
        summary[f"command_hit@{k}"] = round(sum(r[f"command_hit@{k}"] for r in records) / n, 4)
        summary[f"ndcg@{k}"]        = round(sum(r[f"ndcg@{k}"]        for r in records) / n, 4)
    summary["mrr"] = round(sum(r["mrr"] for r in records) / n, 4)
    return summary


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-12 else (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Búsquedas
# ---------------------------------------------------------------------------
def bm25_ranked(searcher: LuceneSearcher, query: str, top_k: int) -> list[str]:
    """Devuelve lista de aliases ordenados por BM25."""
    hits = searcher.search(query, k=top_k)
    result = []
    for h in hits:
        try:
            raw = json.loads(h.raw)
            result.append(raw["id"])
        except Exception:
            result.append(str(h.docid))
    return result


def faiss_ranked(
    query: str,
    faiss_index,
    aliases: np.ndarray,
    embedder: SentenceTransformer,
    top_k: int,
) -> list[str]:
    """Devuelve lista de aliases ordenados por similitud coseno FAISS."""
    q_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    _, positions = faiss_index.search(q_vec, top_k)
    return [str(aliases[p]) for p in positions[0]]


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------
def run_benchmark(
    queries: list[dict],
    aliases: np.ndarray,
    searcher: LuceneSearcher,
    faiss_index,
    embedder: SentenceTransformerEncoder,
    command_aliases: dict[str, set[str]],
    alias_intents: dict[str, str],
    mode: str,
    alpha: float,
    top_k: int,
) -> dict:
    """
    Genera las métricas para el modo indicado:
      'bm25'   → baseline = FAISS puro, experimento = BM25 puro
      'hybrid' → baseline = FAISS puro, experimento = Híbrido BM25+Dense
    """
    # ---------- Codificar todas las queries de una vez (batch para FAISS) ----------
    t0 = time.perf_counter()
    query_texts = [q["query"] for q in queries]
    q_vecs = embedder.encode(
        query_texts
    )
    faiss_scores, faiss_positions = faiss_index.search(q_vecs, top_k)
    retrieval_latency = (time.perf_counter() - t0) / len(queries)
    hybrid_retriever = (
        HybridBM25Retriever(searcher, faiss_index, aliases, embedder, top_k)
        if mode == "hybrid" else None
    )

    baseline_records: list[dict] = []
    experiment_records: list[dict] = []
    experiment_times: list[float] = []

    for item, pos_row, _score_row in zip(queries, faiss_positions, faiss_scores):
        query_text = item["query"]
        relevant   = set().union(*(command_aliases.get(cmd, set()) for cmd in item["relevant_commands"]))

        # ---- Baseline FAISS ----
        baseline_aliases = [str(aliases[p]) for p in pos_row]
        baseline_hits    = [a in relevant for a in baseline_aliases]
        baseline_m       = compute_metrics(baseline_hits, len(relevant))
        baseline_records.append({"id": item["id"], "query": query_text, **baseline_m})

        # ---- Experimento ----
        t_exp = time.perf_counter()
        if mode == "bm25":
            exp_aliases = bm25_ranked(searcher, query_text, top_k)
        elif hybrid_retriever is not None:
            exp_aliases = [
                alias
                for alias, _, _ in hybrid_retriever.search(query_text, top_k, alpha)
            ]
        else:
            raise ValueError(f"Modo de benchmark no soportado: {mode}")
        experiment_times.append(time.perf_counter() - t_exp)

        exp_hits = [a in relevant for a in exp_aliases]
        exp_m    = compute_metrics(exp_hits, len(relevant))
        experiment_records.append({
            "id": item["id"],
            "query": query_text,
            "relevant_commands": item["relevant_commands"],
            "top_5_baseline":   baseline_aliases[:5],
            "top_5_reranked":   exp_aliases[:5],          # mismo campo para compatibilidad con compare_all.py
            **exp_m,
        })

    avg_exp_latency = sum(experiment_times) / len(experiment_times)

    return {
        "mode": mode,
        "alpha": alpha if mode == "hybrid" else None,
        "initial_candidates": top_k,
        "latency_ms": {
            "retrieval_per_query": round(retrieval_latency * 1000, 2),
            "rerank_per_query":    round(avg_exp_latency  * 1000, 2),   # "rerank" = experimento
            "total_per_query":     round((retrieval_latency + avg_exp_latency) * 1000, 2),
        },
        "baseline_summary":  summarize(baseline_records),
        "reranked_summary":  summarize(experiment_records),   # misma clave → compatible con compare_all.py
        "queries_detail":    experiment_records,
    }


# ---------------------------------------------------------------------------
# Consola
# ---------------------------------------------------------------------------
def print_table(results: dict) -> None:
    mode   = results.get("mode", "?")
    alpha  = results.get("alpha")
    label  = f"BM25" if mode == "bm25" else f"Híbrido BM25+Dense (α={alpha})"
    base   = results["baseline_summary"]
    exp    = results["reranked_summary"]
    lat    = results["latency_ms"]
    width  = 70

    print("\n" + "=" * width)
    print(f"  BASELINE (FAISS MPNet)  vs  {label}")
    print("=" * width)
    print(f"{'Métrica':<18} | {'Baseline':<14} | {label:<22} | {'Δ':<9}")
    print("-" * width)
    for m in [
        "command_hit@1", "command_hit@3", "command_hit@5", "command_hit@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "mrr",
    ]:
        b, e = base[m], exp[m]
        diff = e - b
        sign = "+" if diff >= 0 else ""
        if "hit" in m:
            print(f"{m:<18} | {b*100:>12.2f}% | {e*100:>20.2f}% | {sign}{diff*100:>+7.2f}%")
        else:
            print(f"{m:<18} | {b:>14.4f} | {e:>22.4f} | {sign}{diff:>+9.4f}")
    print("-" * width)
    print(
        f"Latencia → Retrieval: {lat['retrieval_per_query']} ms | "
        f"{label}: {lat['rerank_per_query']} ms | "
        f"Total: {lat['total_per_query']} ms"
    )
    print("=" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark BM25 puro o Híbrido BM25+Dense contra Baseline FAISS."
    )
    parser.add_argument(
        "--mode", choices=["bm25", "hybrid"], default="bm25",
        help="'bm25' = solo BM25, 'hybrid' = BM25 + FAISS dense (default: bm25)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Peso del componente denso en modo hybrid [0-1] (default: 0.5)"
    )
    parser.add_argument("--queries",       type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus",        type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--bm25-index",    type=Path, default=DEFAULT_BM25)
    parser.add_argument("--faiss-index",   type=Path, default=DEFAULT_FAISS)
    parser.add_argument("--aliases",       type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--initial-top-k", type=int,  default=INITIAL_TOP_K)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output",        type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    snapshot_sha256 = validate_snapshot(
        args.snapshot_manifest,
        args.queries,
        args.faiss_index,
        args.aliases,
        args.bm25_index,
    )

    # ---- Cargar recursos ----
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    print(f"[INFO] {len(queries)} consultas cargadas desde {args.queries}")

    searcher = LuceneSearcher(str(args.bm25_index))
    print(f"[INFO] Índice BM25 cargado: {args.bm25_index}")

    faiss_index = faiss.read_index(str(args.faiss_index))
    aliases     = np.load(str(args.aliases), allow_pickle=False).astype(str)
    print(f"[INFO] Índice FAISS cargado: {faiss_index.ntotal} vectores")

    command_aliases, alias_intents = load_corpus_metadata(args.corpus, set(aliases))
    print(f"[INFO] Corpus cargado: {len(alias_intents)} aliases en {args.corpus}")

    print(f"[INFO] Cargando embedder: {EMBEDDER_NAME} ...")
    embedder = SentenceTransformerEncoder(EMBEDDER_NAME)

    # ---- Ejecutar benchmark ----
    label = "BM25 puro" if args.mode == "bm25" else f"Híbrido BM25+Dense (α={args.alpha})"
    print(f"[INFO] Ejecutando benchmark '{label}' (Top-{args.initial_top_k} inicial)...")
    results = run_benchmark(
        queries, aliases, searcher, faiss_index, embedder,
        command_aliases, alias_intents,
        mode=args.mode, alpha=args.alpha, top_k=args.initial_top_k,
    )
    results["provenance"] = {
        "snapshot_manifest_sha256": snapshot_sha256,
        "queries_sha256": sha256_file(args.queries),
        "faiss_index_sha256": sha256_file(args.faiss_index),
        "aliases_sha256": sha256_file(args.aliases),
        "bm25_index_files": hash_index_files(args.bm25_index),
        "bm25_index_directory": args.bm25_index.resolve().relative_to(_ROOT).as_posix(),
    }

    # ---- Guardar JSON ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Resultados exportados → {args.output}")

    # ---- Tabla comparativa ----
    print_table(results)


if __name__ == "__main__":
    main()
