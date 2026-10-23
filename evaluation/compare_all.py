#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script:  compare_all.py
Objetivo: Unificar todos los resultados de evaluación (baseline, rerankers y
         variante híbrida) en un único JSON y generar una tabla markdown.
"""

import json
import pathlib
from typing import Dict, Any, List

# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
def load_json(p: pathlib.Path) -> Dict[str, Any]:
    """Carga un JSON y devuelve su contenido ({} si no existe)."""
    if not p.is_file():
        print(f"[WARN] File not found: {p}")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

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
        lat = c["latency_ms"]
        total = lat.get("total_per_query") or (
            lat.get("retrieval_per_query", 0) + lat.get("rerank_per_query", 0)
        )
        rows.append(
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
        )

    # ---------- Reranker DistilRoBERTa ----------
    dist = load_json(DISTIL_FILE)
    if dist:
        configs.append(
            extract_config(dist, "Reranker DistilRoBERTa‑base", "stsb‑distilroberta‑base")
        )

    # ---------- Reranker BGE ----------
    bge = load_json(BGE_FILE)
    if bge:
        configs.append(
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