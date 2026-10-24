#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script:  compare_all.py
<<<<<<< HEAD
Objetivo: Unificar todos los resultados de evaluación (baseline, rerankers y
         variante híbrida) en un único JSON y generar una tabla markdown.
=======
Objetivo: Unificar todos los resultados de evaluación (baseline, rerankers,
          BM25 puro e híbrido) en un único JSON y generar una tabla markdown.

Uso (desde la raíz del repo):
    dataset_builder/indexer/.venv/bin/python evaluation/compare_all.py
>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
"""

import json
import pathlib
from typing import Dict, Any, List

# ----------------------------------------------------------------------
<<<<<<< HEAD
#   Configuración de rutas (ajusta si cambias la estructura)
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent   # <repo root>
EVAL_DIR = ROOT / "evaluation"

# Archivos que ya existen (baseline + rerankers)
BASELINE_FILE   = EVAL_DIR / "reranker_benchmark_v2_top15.json"
DISTIL_FILE     = EVAL_DIR / "reranker_benchmark_distilroberta.json"
BGE_FILE        = EVAL_DIR / "reranker_benchmark_bge.json"
HYBRID_FILE     = EVAL_DIR / "hybrid_benchmark_top15.json"   # opcional

OUT_JSON   = EVAL_DIR / "comparative_all.json"
OUT_TABLE  = EVAL_DIR / "comparative_table.md"
=======
#   Configuración de rutas
# ----------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).parent.parent   # <repo root>
EVAL_DIR = ROOT / "evaluation"

# Archivos generados por los distintos benchmarks
BASELINE_FILE    = EVAL_DIR / "reranker_benchmark_v2_top15.json"       # baseline + MiniLM
DISTIL_FILE      = EVAL_DIR / "reranker_benchmark_distilroberta.json"  # DistilRoBERTa
BGE_FILE         = EVAL_DIR / "reranker_benchmark_bge.json"            # BGE reranker
BM25_FILE        = EVAL_DIR / "bm25_benchmark_top15.json"              # BM25 puro
HYBRID_FILE      = EVAL_DIR / "hybrid_bm25_dense_top15.json"          # BM25 + Dense

OUT_JSON  = EVAL_DIR / "comparative_all.json"
OUT_TABLE = EVAL_DIR / "comparative_table.md"

>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)

# ----------------------------------------------------------------------
def load_json(p: pathlib.Path) -> Dict[str, Any]:
    """Carga un JSON y devuelve su contenido ({} si no existe)."""
    if not p.is_file():
<<<<<<< HEAD
        print(f"[WARN] File not found: {p}")
=======
        print(f"[WARN] Archivo no encontrado, se omite: {p}")
>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

<<<<<<< HEAD
def extract_config(data: Dict[str, Any], label: str, model_name: str) -> Dict[str, Any]:
    """
    Normaliza la estructura que aparece en los benchmarks.
    Cada benchmark tiene: baseline_summary, reranked_summary y latency_ms.
    """
    cfg = {
        "name": label,
        "model": model_name,
        "summary": data.get("reranked_summary", data.get("baseline_summary", {})),
        "latency_ms": data.get("latency_ms", {})
    }
    # En el caso del baseline puro el reranked_summary es idéntico al baseline,
    # así que lo usamos directamente.
    return cfg

def build_table(configs: List[Dict[str, Any]]) -> str:
    """Genera tabla markdown alineada con los campos de interés."""
    header = (
        "| Config | Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@1 | nDCG@5 | Total latency ms |\n"
        "|--------|-------|------:|------:|------:|-------:|-----:|-------:|-------:|-----------------:|\n"
    )
    rows = []
    for c in configs:
        s = c["summary"]
=======

def extract_config(data: Dict[str, Any], label: str, model_name: str) -> Dict[str, Any]:
    """
    Normaliza la estructura de un benchmark a:
      { name, model, summary (métricas del experimento), latency_ms }

    Los benchmarks de reranker tienen 'reranked_summary'.
    El baseline puro usa 'baseline_summary'.
    Los nuevos benchmarks BM25/hybrid también usan 'reranked_summary'.
    """
    summary = data.get("reranked_summary") or data.get("baseline_summary") or {}
    return {
        "name":       label,
        "model":      model_name,
        "summary":    summary,
        "latency_ms": data.get("latency_ms", {}),
    }


def build_table(configs: List[Dict[str, Any]]) -> str:
    """Genera tabla markdown con los campos de interés."""
    header = (
        "| Config | Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@1 | nDCG@5 | Lat. total (ms) |\n"
        "|--------|-------|------:|------:|------:|-------:|----:|-------:|-------:|----------------:|\n"
    )
    rows = []
    for c in configs:
        s   = c["summary"]
>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
        lat = c["latency_ms"]
        total = lat.get("total_per_query") or (
            lat.get("retrieval_per_query", 0) + lat.get("rerank_per_query", 0)
        )
        rows.append(
<<<<<<< HEAD
            f"| {c['name']} | {c['model']} | {s.get('command_hit@1',0):.4f} "
            f"| {s.get('command_hit@3',0):.4f} | {s.get('command_hit@5',0):.4f} "
            f"| {s.get('command_hit@10',0):.4f} | {s.get('mrr',0):.4f} "
            f"| {s.get('ndcg@1',0):.4f} | {s.get('ndcg@5',0):.4f} | {total:.2f} |\n"
        )
    return header + "".join(rows)

def main() -> None:
    configs = []

    # ---------- Baseline (FAISS‑MPNet) ----------
    base = load_json(BASELINE_FILE)
    if base:
        configs.append(
            extract_config(base, "Baseline (FAISS MPNet)", "FAISS + MPNet")
=======
            f"| {c['name']} | {c['model']} "
            f"| {s.get('command_hit@1',  0):.4f} "
            f"| {s.get('command_hit@3',  0):.4f} "
            f"| {s.get('command_hit@5',  0):.4f} "
            f"| {s.get('command_hit@10', 0):.4f} "
            f"| {s.get('mrr',            0):.4f} "
            f"| {s.get('ndcg@1',         0):.4f} "
            f"| {s.get('ndcg@5',         0):.4f} "
            f"| {total:.2f} |\n"
        )
    return header + "".join(rows)


def main() -> None:
    configs: List[Dict[str, Any]] = []

    # ---------- Baseline (FAISS + MPNet, sin reranker) ----------
    base = load_json(BASELINE_FILE)
    if base:
        # En reranker_benchmark_v2_top15.json el baseline_summary es el FAISS puro
        baseline_summary = base.get("baseline_summary", {})
        baseline_lat     = {
            "retrieval_per_query": base.get("latency_ms", {}).get("retrieval_per_query", 0),
            "rerank_per_query":    0,
            "total_per_query":     base.get("latency_ms", {}).get("retrieval_per_query", 0),
        }
        configs.append({
            "name":       "Baseline (FAISS MPNet)",
            "model":      "all-mpnet-base-v2",
            "summary":    baseline_summary,
            "latency_ms": baseline_lat,
        })

    # ---------- Reranker MiniLM (cross-encoder/ms-marco-MiniLM-L-6-v2) ----------
    if base:
        configs.append(
            extract_config(base, "Reranker MiniLM‑L6", "ms‑marco‑MiniLM‑L‑6‑v2")
>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
        )

    # ---------- Reranker DistilRoBERTa ----------
    dist = load_json(DISTIL_FILE)
    if dist:
        configs.append(
<<<<<<< HEAD
            extract_config(dist, "Reranker DistilRoBERTa‑base", "stsb‑distilroberta‑base")
=======
            extract_config(dist, "Reranker DistilRoBERTa", "stsb‑distilroberta‑base")
>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
        )

    # ---------- Reranker BGE ----------
    bge = load_json(BGE_FILE)
    if bge:
        configs.append(
<<<<<<< HEAD
            extract_config(bge, "Reranker BGE‑base", "bge‑reranker‑base")
        )

    # ---------- Híbrido BM25 + dense (opcional) ----------
    hybrid = load_json(HYBRID_FILE)
    if hybrid:
        # el script híbrido ya devuelve sólo la lista rankeada y latencia;
        # usamos los mismos nombres de métricas que el baseline para comparar.
        # Si el archivo contiene un bloque "metrics" añádelo aquí.
        summary = hybrid.get("metrics", {})
        configs.append(
            {
                "name": "Hybrid BM25 + dense",
                "model": "BM25 + MPNet (alpha=0.7)",
                "summary": summary,
                "latency_ms": hybrid.get("latency_ms", {})
            }
        )

    # ---------- Escritura de resultados ----------
    out = {"configs": configs}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    OUT_TABLE.write_text(build_table(configs))

    print(f"[INFO] Unified file → {OUT_JSON}")
    print(f"[INFO] Markdown table → {OUT_TABLE}")

if __name__ == "__main__":
    main()
=======
            extract_config(bge, "Reranker BGE‑base", "bge‑reranker‑base")
        )

    # ---------- BM25 puro ----------
    bm25 = load_json(BM25_FILE)
    if bm25:
        configs.append(
            extract_config(bm25, "BM25 puro (Pyserini)", "BM25 (Lucene)")
        )

    # ---------- Híbrido BM25 + Dense ----------
    hybrid = load_json(HYBRID_FILE)
    if hybrid:
        alpha = hybrid.get("alpha") or 0.5
        configs.append(
            extract_config(
                hybrid,
                f"Híbrido BM25+Dense (α={alpha})",
                f"BM25 + MPNet (α={alpha})",
            )
        )

    if not configs:
        print("[ERROR] No se encontró ningún archivo de benchmark. Genera al menos uno primero.")
        return

    # ---------- Escritura de resultados ----------
    out = {"configs": configs}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    OUT_TABLE.write_text(build_table(configs))

    print(f"[INFO] Archivo unificado → {OUT_JSON}")
    print(f"[INFO] Tabla markdown    → {OUT_TABLE}")
    print(f"\n{'=' * 60}")
    print(build_table(configs))


if __name__ == "__main__":
    main()

>>>>>>> 31829c18 (feat: bm25 dense n hybrid benchmark)
