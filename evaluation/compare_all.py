#!/usr/bin/env python3
"""Consolida las métricas de retrieval de la fase 2 en un único reporte.

Los benchmarks de rerankers guardan el baseline FAISS y el resultado del
cross-encoder. Los de BM25/híbrido usan ``reranked_summary`` para el sistema
evaluado. Este script hace esa elección explícita y falla ante un esquema
incompleto, en lugar de presentar ceros engañosos.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation"
OUT_JSON = EVAL_DIR / "comparative_all.json"
OUT_TABLE = EVAL_DIR / "comparative_table.md"

METRICS = (
    "command_hit@1", "command_hit@3", "command_hit@5", "command_hit@10",
    "mrr", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    model: str
    files: tuple[str, ...]
    summary_key: str
    latency_kind: str


SPECS = (
    BenchmarkSpec("Baseline (FAISS MPNet)", "all-mpnet-base-v2",
                  ("retrieval_benchmark_v2.json",), "metrics", "retrieval"),
    BenchmarkSpec("Reranker MiniLM-L6", "cross-encoder/ms-marco-MiniLM-L-6-v2",
                  ("reranker_benchmark_v2_top15.json",), "reranked_summary", "total"),
    BenchmarkSpec("Reranker DistilRoBERTa", "cross-encoder/stsb-distilroberta-base",
                  ("reranker_benchmark_distilroberta.json",), "reranked_summary", "total"),
    BenchmarkSpec("Reranker BGE-base", "BAAI/bge-reranker-base",
                  ("reranker_benchmark_bge.json",), "reranked_summary", "total"),
    BenchmarkSpec("BM25 (Pyserini/Lucene)", "BM25",
                  ("bm25_benchmark_v2_top15.json",), "reranked_summary", "experiment"),
    BenchmarkSpec("Híbrido BM25 + Dense", "BM25 + all-mpnet-base-v2",
                  ("hybrid_benchmark_v2_top15.json", "hybrid_benchmark_top15.json", "hybrid_bm25_dense_top15.json"),
                  "reranked_summary", "experiment"),
)


def load_json(candidates: tuple[str, ...]) -> tuple[dict[str, Any], Path] | None:
    """Devuelve el primer artefacto existente de una lista de nombres."""
    for filename in candidates:
        path = EVAL_DIR / filename
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}: {exc}") from exc
    print(f"[WARN] Benchmark no encontrado: {', '.join(candidates)}")
    return None


def validate_summary(summary: Any, source: Path, key: str) -> dict[str, float | int]:
    if not isinstance(summary, dict):
        raise ValueError(f"{source}: falta el objeto {key!r}.")
    missing = [metric for metric in ("queries", *METRICS) if metric not in summary]
    if missing:
        raise ValueError(f"{source}: {key!r} no contiene: {', '.join(missing)}.")
    if not isinstance(summary["queries"], int) or summary["queries"] <= 0:
        raise ValueError(f"{source}: 'queries' debe ser un entero positivo.")
    normalized: dict[str, float | int] = {"queries": summary["queries"]}
    for metric in METRICS:
        value = summary[metric]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{source}: {metric!r} debe estar entre 0 y 1.")
        normalized[metric] = float(value)
    return normalized


def normalize_latency(data: dict[str, Any], kind: str, source: Path) -> dict[str, float]:
    latency = data.get("latency_ms")
    if not isinstance(latency, dict):
        raise ValueError(f"{source}: falta el objeto 'latency_ms'.")
    retrieval = float(latency.get("retrieval_per_query", 0.0))
    experiment = float(latency.get("rerank_per_query", 0.0))
    reported_total = float(latency.get("total_per_query", retrieval + experiment))
    if min(retrieval, experiment, reported_total) < 0:
        raise ValueError(f"{source}: la latencia no puede ser negativa.")

    # BM25 e híbrido reutilizan el campo histórico "rerank_per_query" para
    # medir el sistema experimental. Sumar el baseline otra vez lo duplicaría.
    method = retrieval if kind == "retrieval" else reported_total if kind == "total" else experiment
    return {
        "method_per_query": round(method, 2),
        "retrieval_per_query": round(retrieval, 2),
        "experiment_per_query": round(experiment, 2),
        "reported_total_per_query": round(reported_total, 2),
    }


def build_config(spec: BenchmarkSpec) -> dict[str, Any] | None:
    loaded = load_json(spec.files)
    if loaded is None:
        return None
    data, source = loaded
    summary = validate_summary(data.get(spec.summary_key), source, spec.summary_key)
    name, model = spec.name, spec.model
    if data.get("mode") == "hybrid":
        alpha = data.get("alpha")
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError(f"{source}: alpha debe estar entre 0 y 1 para el híbrido.")
        name, model = f"{name} (α={alpha:g})", f"{model} (α={alpha:g})"
    source_baseline = data.get("baseline_summary")
    return {
        "name": name,
        "model": model,
        "source_file": source.name,
        "summary_source": spec.summary_key,
        "summary": summary,
        "latency_ms": normalize_latency(data, spec.latency_kind, source),
        "source_baseline_summary": (
            validate_summary(source_baseline, source, "baseline_summary")
            if source_baseline is not None else None
        ),
        "provenance": data.get("provenance"),
    }


def validate_comparability(configs: list[dict[str, Any]]) -> None:
    counts = {config["summary"]["queries"] for config in configs}
    if len(counts) > 1:
        details = ", ".join(f"{c['name']}={c['summary']['queries']}" for c in configs)
        raise ValueError(f"Los benchmarks no usan el mismo número de consultas: {details}")


def annotate_reference_consistency(configs: list[dict[str, Any]]) -> None:
    """Marca resultados que fueron producidos con otro baseline FAISS.

    Igualar el número de consultas no prueba que índice, corpus y qrels hayan
    sido idénticos. Los artefactos disponibles incluyen su baseline interno;
    usarlo evita mezclar resultados de snapshots distintos como si fueran una
    comparación experimental válida.
    """
    reference = next(config for config in configs if config["name"] == "Baseline (FAISS MPNet)")
    reference_summary = reference["summary"]
    reference_snapshot = (reference["provenance"] or {}).get("snapshot_manifest_sha256")
    for config in configs:
        source_baseline = config["source_baseline_summary"]
        summary_matches = source_baseline is None or all(
            source_baseline[metric] == reference_summary[metric]
            for metric in ("queries", *METRICS)
        )
        snapshot = (config["provenance"] or {}).get("snapshot_manifest_sha256")
        snapshot_matches = (
            config["name"] == "Baseline (FAISS MPNet)"
            or (reference_snapshot is not None and snapshot == reference_snapshot)
        )
        matches = summary_matches and snapshot_matches
        config["comparable_with_reference"] = matches
        config["comparison_note"] = (
            "Snapshot y baseline FAISS coinciden"
            if matches else "Snapshot no verificado o baseline FAISS distinto: requiere reejecución"
        )
        del config["source_baseline_summary"]


def build_table(configs: list[dict[str, Any]]) -> str:
    header = (
        "| Configuración | Comparable | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | Latencia método (ms) |\n"
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for config in configs:
        summary, latency = config["summary"], config["latency_ms"]["method_per_query"]
        rows.append(
            f"| {config['name']} | {'Sí' if config['comparable_with_reference'] else 'No'} | {summary['command_hit@1']:.4f} | {summary['command_hit@3']:.4f} "
            f"| {summary['command_hit@5']:.4f} | {summary['command_hit@10']:.4f} | {summary['mrr']:.4f} "
            f"| {summary['ndcg@1']:.4f} | {summary['ndcg@3']:.4f} | {summary['ndcg@5']:.4f} "
            f"| {summary['ndcg@10']:.4f} | {latency:.2f} |\n"
        )
    return header + "".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Valida e imprime sin sobrescribir los reportes.")
    args = parser.parse_args()

    configs = [config for spec in SPECS if (config := build_config(spec)) is not None]
    if not configs:
        raise SystemExit("[ERROR] No se encontró ningún benchmark para consolidar.")
    validate_comparability(configs)
    annotate_reference_consistency(configs)
    report = {
        "benchmark": {"queries": configs[0]["summary"]["queries"], "metrics": list(METRICS)},
        "configs": configs,
    }
    table = build_table(configs)
    if args.dry_run:
        print(table)
        return

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_TABLE.write_text(table, encoding="utf-8")
    print(f"[INFO] Archivo unificado → {OUT_JSON}")
    print(f"[INFO] Tabla markdown    → {OUT_TABLE}")
    print(table)


if __name__ == "__main__":
    main()
