#!/usr/bin/env python3
"""Genera embeddings BAAI/bge-m3 (1024-d) para el corpus en cluster CPU.

Ubicación esperada: al mismo nivel que los directorios corpus/ y artifacts/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = SCRIPT_DIR / "corpus" if (SCRIPT_DIR / "corpus").exists() else Path("corpus")
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" if (SCRIPT_DIR / "artifacts").exists() else Path("artifacts")
MODEL_NAME = "BAAI/bge-m3"
VECTOR_NAME = "bge_m3_vectors.npz"


@dataclass(frozen=True)
class CorpusDocument:
    record_id: str
    category: str
    source_id: int
    intent: str
    command: str
    example: str

    def text_for_embedding(self) -> str:
        if not self.intent:
            raise ValueError(f"{self.record_id} no tiene intent")
        return self.intent


def load_corpus(corpus_dir: Path) -> list[CorpusDocument]:
    """Carga y valida los documentos YAML del corpus."""
    # Si existe módulo corpus.py en el path, intentamos usarlo
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from corpus import load_corpus as external_load_corpus  # type: ignore
        return external_load_corpus(corpus_dir)
    except Exception:
        pass

    documents: list[CorpusDocument] = []
    seen_ids: set[str] = set()
    yaml_files = sorted(corpus_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No se encontraron archivos .yaml en {corpus_dir}")

    for path in yaml_files:
        with path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        category = str(payload.get("category", "")).strip()
        if not category:
            raise ValueError(f"{path}: falta category")
        rows = payload.get("documents", [])
        if not isinstance(rows, list):
            raise ValueError(f"{path}: documents debe ser una lista")
        for row in rows:
            source_id = row.get("id")
            if not isinstance(source_id, int):
                raise ValueError(f"{path}: id inválido")
            record_id = f"{category}:{source_id}"
            if record_id in seen_ids:
                raise ValueError(f"ID duplicado: {record_id}")
            seen_ids.add(record_id)
            documents.append(
                CorpusDocument(
                    record_id=record_id,
                    category=category,
                    source_id=source_id,
                    intent=str(row.get("intent", "")).strip(),
                    command=str(row.get("command", "")).strip(),
                    example=str(row.get("example", "")).strip(),
                )
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Ruta al directorio corpus/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta al directorio artifacts/")
    parser.add_argument("--model", default=MODEL_NAME, help="Modelo HuggingFace (por defecto: BAAI/bge-m3)")
    parser.add_argument(
        "--vector-name",
        default=VECTOR_NAME,
        help="Nombre del archivo npz de salida (por defecto: bge_m3_vectors.npz)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote (batch size)")
    parser.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Número de hilos de CPU para PyTorch (por defecto: 24)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo ('cuda', 'cpu', o auto-detectado)",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración de 24 hilos de CPU para cluster
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
        print(f"[Embedder] PyTorch configurado en CPU con {args.threads} hilos intra-op ({interop_threads} inter-op).")
    elif device.startswith("cuda"):
        print(f"[Embedder] Aceleración GPU detectada: {torch.cuda.get_device_name(0)}")

    print(f"[Embedder] Cargando corpus desde: {args.corpus}")
    documents = load_corpus(args.corpus)
    intents = [document.text_for_embedding() for document in documents]
    print(f"[Embedder] Total de documentos cargados: {len(documents)}")

    print(f"[Embedder] Cargando modelo: {args.model} en dispositivo: {device}")
    started_model = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    model_load_s = time.perf_counter() - started_model
    print(f"[Embedder] Modelo cargado en {model_load_s:.2f}s (dim={model.get_sentence_embedding_dimension()}).")

    print(f"[Embedder] Codificando {len(intents)} intents con batch_size={args.batch_size}...")
    started_encode = time.perf_counter()
    vectors = model.encode(
        intents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    encode_s = time.perf_counter() - started_encode
    print(f"[Embedder] Codificación completada en {encode_s:.2f}s ({len(intents)/max(encode_s, 0.001):.1f} docs/s).")

    args.output.mkdir(parents=True, exist_ok=True)
    aliases = np.asarray([document.record_id for document in documents], dtype=str)
    output_path = args.output / args.vector_name
    np.savez_compressed(output_path, vectors=vectors, ids=aliases)
    print(f"[Embedder] Embeddings guardados: {vectors.shape[0]} vectores × {vectors.shape[1]} dimensiones")
    print(f"[Embedder] Salida: {output_path}")


if __name__ == "__main__":
    main()

