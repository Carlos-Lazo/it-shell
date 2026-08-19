#!/usr/bin/env python3
"""Consulta el índice FAISS MPNet y devuelve aliases ordenados por similitud."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="intención en lenguaje natural")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=Path("artifacts/all-mpnet-base-v2/index.faiss"))
    parser.add_argument("--aliases", type=Path, default=Path("artifacts/all-mpnet-base-v2/aliases.npy"))
    args = parser.parse_args()

    index = faiss.read_index(str(args.index))
    aliases = np.load(args.aliases, allow_pickle=False).astype(str)
    if index.ntotal != len(aliases):
        raise ValueError("El índice y el archivo de aliases no están alineados")
    if args.top_k < 1 or args.top_k > index.ntotal:
        raise ValueError(f"top-k debe estar entre 1 y {index.ntotal}")

    model = SentenceTransformer(MODEL_NAME)
    query = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, positions = index.search(query, args.top_k)
    for rank, (score, position) in enumerate(zip(scores[0], positions[0], strict=True), start=1):
        print(f"{rank}. {aliases[position]}  similitud={score:.4f}")


if __name__ == "__main__":
    main()