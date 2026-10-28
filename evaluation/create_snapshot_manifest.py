#!/usr/bin/env python3
"""Validate and record the exact inputs used by the 160-query retrieval benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import yaml
from pyserini.search.lucene import LuceneSearcher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "evaluation/evaluation_v2_queries.json"
DEFAULT_CORPUS = ROOT / "dataset_builder/corpus"
DEFAULT_VECTORS = ROOT / "dataset_builder/embedder/artifacts/vectors.npz"
DEFAULT_INDEX = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss"
DEFAULT_ALIASES = ROOT / "dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy"
DEFAULT_BM25_INDEX = ROOT / "dataset_builder/indexer/artifacts/bm25_indexs_v2"
DEFAULT_OUTPUT = ROOT / "evaluation/retrieval_snapshot_v2.json"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(directory: Path) -> dict[str, str]:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "write.lock")
    if not files:
        raise ValueError(f"No se encontraron archivos de índice en {directory}")
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in files
    }


def load_corpus_aliases(corpus_dir: Path) -> tuple[set[str], dict[str, set[str]], list[Path]]:
    aliases: set[str] = set()
    commands: dict[str, set[str]] = {}
    paths = sorted(corpus_dir.glob("*.yaml"))
    if not paths:
        raise ValueError(f"No se encontraron archivos YAML en {corpus_dir}")

    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        category = str(payload.get("category", "")).strip()
        if not category:
            raise ValueError(f"{path}: falta la categoría")
        for row in payload.get("documents", []):
            alias = f"{category}:{row['id']}"
            if alias in aliases:
                raise ValueError(f"Alias duplicado en el corpus: {alias}")
            aliases.add(alias)
            command = str(row.get("command", "")).split(maxsplit=1)
            if command:
                commands.setdefault(command[0], set()).add(alias)
    return aliases, commands, paths


def build_manifest(
    queries_path: Path,
    corpus_dir: Path,
    vectors_path: Path,
    index_path: Path,
    aliases_path: Path,
    bm25_index_path: Path,
) -> dict:
    with np.load(vectors_path, allow_pickle=False) as archive:
        vectors = np.ascontiguousarray(archive["vectors"], dtype=np.float32)
        vector_ids = archive["ids"].astype(str)
    aliases = np.load(aliases_path, allow_pickle=False).astype(str)
    index = faiss.read_index(str(index_path))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    corpus_aliases, command_aliases, corpus_paths = load_corpus_aliases(corpus_dir)
    if not bm25_index_path.is_dir():
        raise ValueError(f"No se encontró el índice BM25: {bm25_index_path}")
    bm25_files = sha256_tree(bm25_index_path)
    bm25_document_count = LuceneSearcher(str(bm25_index_path)).num_docs
    if bm25_document_count != len(corpus_aliases):
        raise ValueError(
            f"El índice BM25 contiene {bm25_document_count} documentos; "
            f"el corpus tiene {len(corpus_aliases)} aliases"
        )

    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f"Se esperaban vectores N x 768; recibido {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Los vectores contienen valores no finitos")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("Los vectores no están normalizados")
    if not np.array_equal(vector_ids, aliases):
        raise ValueError("Los IDs de vectors.npz no coinciden en orden con aliases.npy")
    if index.ntotal != len(aliases) or index.d != vectors.shape[1]:
        raise ValueError("El índice FAISS no está alineado en cantidad o dimensión")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("El índice FAISS no usa producto interno")
    if set(aliases) != corpus_aliases:
        missing = len(corpus_aliases - set(aliases))
        unknown = len(set(aliases) - corpus_aliases)
        raise ValueError(
            f"Aliases FAISS/corpus desalineados: {missing} ausentes del índice, {unknown} fuera del corpus"
        )
    if len(queries) != 160:
        raise ValueError(f"El benchmark oficial debe contener 160 consultas; recibido {len(queries)}")

    missing_commands = sorted({
        command
        for query in queries
        for command in query["relevant_commands"]
        if not command_aliases.get(command)
    })
    if missing_commands:
        raise ValueError(f"Comandos relevantes sin aliases en el corpus: {', '.join(missing_commands)}")

    reconstructed = np.empty_like(vectors)
    index.reconstruct_n(0, index.ntotal, reconstructed)
    if not np.array_equal(reconstructed, vectors):
        raise ValueError("Los vectores reconstruidos de FAISS difieren del archivo vectors.npz")

    files = [queries_path, vectors_path, aliases_path, index_path, *corpus_paths]
    return {
        "snapshot_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "embedding": {
            "model": MODEL_NAME,
            "dimensions": int(vectors.shape[1]),
            "vectors": int(vectors.shape[0]),
            "normalized": True,
            "index_metric": "inner_product",
        },
        "bm25": {
            "directory": bm25_index_path.relative_to(ROOT).as_posix(),
            "documents": bm25_document_count,
            "files": bm25_files,
        },
        "corpus": {
            "directory": corpus_dir.relative_to(ROOT).as_posix(),
            "yaml_files": len(corpus_paths),
            "documents": len(corpus_aliases),
        },
        "evaluation": {
            "queries_file": queries_path.relative_to(ROOT).as_posix(),
            "queries": len(queries),
            "sha256": sha256_file(queries_path),
        },
        "validation": {
            "vector_ids_match_aliases_in_order": True,
            "index_reconstructs_vectors_in_order": True,
            "index_aliases_match_corpus": True,
            "all_relevant_commands_present": True,
        },
        "artifacts": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = build_manifest(
        args.queries.resolve(),
        args.corpus.resolve(),
        args.vectors.resolve(),
        args.index.resolve(),
        args.aliases.resolve(),
        args.bm25_index.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Snapshot validado y registrado: {args.output}")
    print(
        f"Consultas: {manifest['evaluation']['queries']} | "
        f"Documentos: {manifest['corpus']['documents']} | "
        f"Vectores: {manifest['embedding']['vectors']}"
    )


if __name__ == "__main__":
    main()
