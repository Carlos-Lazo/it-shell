#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bm25_indexer.py
----------------
Construye un índice BM25 (Lucene) a partir del corpus YAML que se
encuentra en ``dataset_builder/corpus/``.  

El script acepta dos formas de entrada:
  * **--corpus <folder>**   → busca todos los *.yaml* dentro del folder.
  * **--corpus <file.json>** → comportamiento legacy (un JSON con
    [{ "id": ..., "command": ... }, …]).

Requisitos:
    
"""

import argparse
import json
import pathlib
import sys
import tempfile
from typing import List, Dict, Any

# La clase LuceneIndexer no expone un método .main en la versión 0.22.0.
# En su lugar utilizamos el módulo ejecutable de Pyserini vía subprocess, que
# acepta los mismos argumentos de línea de comandos.
import subprocess

import yaml                     # PyYAML (ya está en requirements.txt)

# Nueva API de Pyserini (v0.22.0)
from pyserini.index.lucene import LuceneIndexer   # clase que implementa la indexación

CORPUS_PATH = "../corpus"
OUTPUT_INDEX_PATH = "artifacts/bm25_indexs_v2"
BASE_THREADS = 4
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_yaml_folder(folder: pathlib.Path = CORPUS_PATH) -> List[Dict[str, Any]]:
    """
    Recorre *folder* y carga todos los archivos *.yaml*.
    Devuelve una lista plana de dicts con las claves:
        - id      : (int o str) identificador único del comando
        - command : (str)   texto del comando
        - intent  : (str)   descripción 
    """
    records: List[Dict[str, Any]] = []

    for yaml_path in sorted(folder.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Cada archivo tiene la estructura: {category:…, documents: [{id,…}]}
        category = str(data.get("category", ""))
        docs = data.get("documents", [])
        for doc in docs:
            # Usamos "category:id" como alias, igual que el resto del sistema
            rec: Dict[str, Any] = {
                "id": f"{category}:{doc.get('id')}",
                "command": doc.get("command", ""),
                "intent": doc.get("intent", ""),
            }
            records.append(rec)

    return records


def _load_json_corpus(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Compatibilidad con la versión original que aceptaba un único JSON.
    El JSON debe ser una lista de objetos con al menos ``id`` y ``command``.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalizamos los ids a string y nos aseguramos de que haya 'command'
    records = []
    for entry in data:
        if "command" not in entry:
            continue
        rec = {
            "id": str(entry.get("id")),
            "command": entry["command"],
            "intent": entry["intent"],
        }
        records.append(rec)
    return records


def _prepare_input(
    records: List[Dict[str, Any]],
    tmp_dir: pathlib.Path,
) -> pathlib.Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmp_dir / "corpus.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as out:
        for rec in records:
            doc = {
                "id": rec["id"],
                "contents": rec["command"],
            }
            # Si el registro lleva una descripción, la guardamos
            if "intent" in rec:
                doc["intent"] = rec["intent"]
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return jsonl_path


def build_bm25_index(
    corpus_path: pathlib.Path = CORPUS_PATH,
    index_dir: pathlib.Path = OUTPUT_INDEX_PATH,
    threads: int = BASE_THREADS,
) -> None:
    """
    Crea el índice BM25 a partir de ``corpus_path``.

    ``corpus_path`` puede ser:
        * una carpeta con archivos *.yaml*  → se procesa con ``_load_yaml_folder``.
        * un archivo *.json*                → se procesa con ``_load_json_corpus``.
    """
    # -----------------------------------------------------------------
    # 1️⃣  Cargar los registros del corpus (yaml o json)
    # -----------------------------------------------------------------
    if corpus_path.is_dir():
        records = _load_yaml_folder(corpus_path)
    elif corpus_path.suffix.lower() in {".json", ".jsonl"}:
        records = _load_json_corpus(corpus_path)
    else:
        sys.exit(
            f"[ERROR] El argumento --corpus debe ser un directorio de YAML "
            f"o un archivo JSON. Recibido: {corpus_path}"
        )

    if not records:
        sys.exit("[ERROR] No se encontraron documentos en el corpus.")

    # -----------------------------------------------------------------
    # 2️⃣  Preparar el archivo JSONL temporal que Pyserini consume
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="it-shell-bm25-input-") as temp_dir:
        tmp_input = pathlib.Path(temp_dir)
        _prepare_input(records, tmp_input)

        # -----------------------------------------------------------------
        # 3️⃣  Ejecutar el indexador de Pyserini
        # -----------------------------------------------------------------
        # --input apunta al JSONL temporal y --index al destino Lucene.
        pyserini_args = [
            "--input", str(tmp_input.resolve()),
            "--index", str(index_dir.resolve()),
            "--collection", "JsonCollection",
            "--threads", str(threads),
            "--storeRaw",
            "--storeDocvectors",
        ]

        print(f"[INFO] Building BM25 index at {index_dir}")
        cmd = [sys.executable, "-m", "pyserini.index.lucene"] + pyserini_args
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"[ERROR] Indexing failed (code {result.returncode})")
    print("[INFO] BM25 index created successfully")


# ----------------------------------------------------------------------
# CLI entry‑point (mantiene la compatibilidad con la versión anterior)
# ----------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a BM25 (Lucene) index from a command corpus. "
            "The corpus can be a directory of *.yaml* files (default for this "
            "project) or a legacy JSON file."
        )
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=pathlib.Path(CORPUS_PATH),
        
        help=(
            "Path to the corpus. Either a folder containing *.yaml* files "
            "(recommended) or a JSON file with [{'id':…, 'command':…}] entries."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(OUTPUT_INDEX_PATH),
        
        help="Directory where the Lucene BM25 index will be written.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of indexing threads (default: 4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    build_bm25_index(args.corpus, args.output, args.threads)