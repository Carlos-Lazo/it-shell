#!/usr/bin/env python3
"""Construye un índice FAISS FlatIP (1024-d) para los embeddings de BAAI/bge-m3."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INDEXER_DIR = BASE_DIR.parent
DEFAULT_OUTPUT = INDEXER_DIR / "artifacts/bge-m3"
DEFAULT_VECTORS = INDEXER_DIR.parent / "embedder/artifacts/bge_m3_vectors.npz"
DIMENSIONS = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS, help="Ruta al archivo .npz de vectores BGE-M3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directorio donde guardar index.faiss y aliases.npy")
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS, help="Dimensión vectorial (por defecto: 1024)")
    parser.add_argument("--threads", type=int, default=24, help="Número de hilos OpenMP para FAISS (por defecto: 24)")
    args = parser.parse_args()

    if args.threads is not None and args.threads > 0:
        faiss.omp_set_num_threads(args.threads)
        print(f"[Indexer BGE-M3] FAISS configurado con {args.threads} hilos OpenMP.")

    print(f"[Indexer BGE-M3] Cargando vectores desde: {args.vectors}")
    with np.load(args.vectors) as data:
        vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
        aliases = data["ids"].astype(str)

    expected_dim = args.dimensions or vectors.shape[1]
    validate(vectors, aliases, expected_dim)

    # Los vectores ya están normalizados: producto interno = similitud coseno.
    index = faiss.IndexFlatIP(expected_dim)
    index.add(vectors)

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.faiss"
    aliases_path = args.output / "aliases.npy"
    faiss.write_index(index, str(index_path))
    np.save(aliases_path, aliases)
    print(f"[Indexer BGE-M3] Índice creado exitosamente: {index.ntotal} vectores × {expected_dim} dimensiones")
    print(f"[Indexer BGE-M3] Salida: {index_path}")


def validate(vectors: np.ndarray, aliases: np.ndarray, expected_dim: int) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError(f"Se esperaban vectores N × {expected_dim}; recibido {vectors.shape}")
    if vectors.shape[0] != aliases.shape[0]:
        raise ValueError("La cantidad de aliases no coincide con los vectores")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Los vectores deben estar normalizados para usar similitud coseno")


if __name__ == "__main__":
    main()

