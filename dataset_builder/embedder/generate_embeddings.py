#!/usr/bin/env python3
"""Genera pares embedding-alias listos para una base vectorial."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import load_corpus

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_NAME = "mpnet_base_v2_vectors.npz"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("../corpus"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    intents = [document.text_for_embedding() for document in documents]
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        intents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    aliases = np.asarray([document.record_id for document in documents], dtype=str)
    output_path = args.output / VECTOR_NAME
    np.savez_compressed(output_path, vectors=vectors, ids=aliases)
    print(f"Embeddings: {vectors.shape[0]} x {vectors.shape[1]}")
    print(f"Salida: {output_path}")


if __name__ == "__main__":
    main()