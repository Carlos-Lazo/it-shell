#!/usr/bin/env python3
"""Genera embeddings BAAI/bge-m3 (1024-d) para el corpus, optimizado para cluster CPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Permite importar corpus desde dataset_builder/embedder/
BASE_DIR = Path(__file__).resolve().parent
EMBEDDER_DIR = BASE_DIR.parent
if str(EMBEDDER_DIR) not in sys.path:
    sys.path.insert(0, str(EMBEDDER_DIR))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from corpus import load_corpus

MODEL_NAME = "BAAI/bge-m3"
VECTOR_NAME = "bge_m3_vectors.npz"
DEFAULT_CORPUS = EMBEDDER_DIR.parent / "corpus"
DEFAULT_OUTPUT = EMBEDDER_DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Ruta al directorio de corpus YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directorio donde guardar los vectores")
    parser.add_argument("--model", default=MODEL_NAME, help="Modelo de embeddings (por defecto BAAI/bge-m3)")
    parser.add_argument(
        "--vector-name",
        default=VECTOR_NAME,
        help="Nombre del archivo npz de salida (por defecto bge_m3_vectors.npz)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote para inferencia (por defecto 32)")
    parser.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Número de hilos de CPU para PyTorch cuando corre en CPU (por defecto: 24)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo ('cuda', 'cpu', o auto-detectado si no se especifica)",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Configurar paralelismo de CPU para entornos de cluster / multi-core si corre en CPU
    if device == "cpu" and args.threads is not None and args.threads > 0:
        torch.set_num_threads(args.threads)
        interop_threads = min(4, max(1, args.threads // 6))
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        print(f"[Embedder BGE-M3] PyTorch configurado en CPU con {args.threads} hilos intra-op ({interop_threads} inter-op).")
    elif device.startswith("cuda"):
        print(f"[Embedder BGE-M3] GPU detectada: {torch.cuda.get_device_name(0)}")

    print(f"[Embedder BGE-M3] Cargando corpus desde: {args.corpus}")
    documents = load_corpus(args.corpus)
    intents = [document.text_for_embedding() for document in documents]
    print(f"[Embedder BGE-M3] Total de documentos cargados: {len(documents)}")

    print(f"[Embedder BGE-M3] Cargando modelo: {args.model} en dispositivo: {device}")
    started_model = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    model_load_s = time.perf_counter() - started_model
    print(f"[Embedder BGE-M3] Modelo cargado en {model_load_s:.2f}s (dim={model.get_sentence_embedding_dimension()}).")

    print(f"[Embedder BGE-M3] Codificando {len(intents)} intents (batch_size={args.batch_size})...")
    started_encode = time.perf_counter()
    vectors = model.encode(
        intents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    encode_s = time.perf_counter() - started_encode
    print(f"[Embedder BGE-M3] Codificación completada en {encode_s:.2f}s ({len(intents)/max(encode_s, 0.001):.1f} docs/s).")

    args.output.mkdir(parents=True, exist_ok=True)
    aliases = np.asarray([document.record_id for document in documents], dtype=str)
    output_path = args.output / args.vector_name
    np.savez_compressed(output_path, vectors=vectors, ids=aliases)
    print(f"[Embedder BGE-M3] Embeddings guardados: {vectors.shape[0]} vectores × {vectors.shape[1]} dimensiones")
    print(f"[Embedder BGE-M3] Salida: {output_path}")


if __name__ == "__main__":
    main()

