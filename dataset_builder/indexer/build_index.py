#!/usr/bin/env python3
"""Construye un índice FAISS de similitud coseno para los embeddings MPNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

DIMENSIONS = 768
OUTPUT_FILE = "artifacts/"
MODEL = "all-mpnet-base-v2"
VECTORS = "../embedder/artifacts/vectors.npz"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path(VECTORS))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_FILE+MODEL))
    args = parser.parse_args()

    with np.load(args.vectors) as data:
        vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
        aliases = data["ids"].astype(str)
    validate(vectors, aliases)

    # Los vectores ya están normalizados: producto interno = similitud coseno.
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(vectors)

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.faiss"
    aliases_path = args.output / "aliases.npy"
    faiss.write_index(index, str(index_path))
    np.save(aliases_path, aliases)
    print(f"Índice: {index.ntotal} vectores × {DIMENSIONS} dimensiones")
    print(f"Salida: {index_path}")


def validate(vectors: np.ndarray, aliases: np.ndarray) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSIONS:
        raise ValueError(f"Se esperaban vectores N × {DIMENSIONS}; recibido {vectors.shape}")
    if vectors.shape[0] != aliases.shape[0]:
        raise ValueError("La cantidad de aliases no coincide con los vectores")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Los vectores deben estar normalizados para usar similitud coseno")


if __name__ == "__main__":
    main()