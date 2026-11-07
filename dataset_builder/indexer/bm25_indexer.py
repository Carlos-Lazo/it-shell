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

import subprocess

import yaml                     # PyYAML (ya está en requirements.txt)

CORPUS_PATH = "../corpus"
OUTPUT_INDEX_PATH = "artifacts/bm25_indexs_v2"
BASE_THREADS = 4
REPRESENTATIONS = ("command", "enriched", "curated", "enriched_paraphrases")
MAX_INTENT_CHARS = 256
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
                "example": doc.get("example", ""),
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
            "intent": entry.get("intent", ""),
            "example": entry.get("example", ""),
        }
        records.append(rec)
    return records


def _contents_for_record(
    record: Dict[str, Any],
    representation: str,
    search_phrases: List[str] | None = None,
) -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Representación BM25 inválida: {representation!r}; "
            f"opciones: {', '.join(REPRESENTATIONS)}"
        )
    fields = [record.get("command", "")]
    if representation in {"enriched", "enriched_paraphrases"}:
        intent = record.get("intent", "")
        if isinstance(intent, str):
            intent = " ".join(intent.split())
            if len(intent) > MAX_INTENT_CHARS:
                intent = intent[:MAX_INTENT_CHARS].rsplit(" ", 1)[0]
        fields.extend((intent, record.get("example", "")))
    elif representation == "curated":
        fields.extend(search_phrases or ())
    if representation == "enriched_paraphrases":
        fields.extend(search_phrases or ())
    return " ".join(
        value.strip()
        for value in fields
        if isinstance(value, str) and value.strip()
    )


def _prepare_input(
    records: List[Dict[str, Any]],
    tmp_dir: pathlib.Path,
    representation: str = "command",
    search_phrases: Dict[str, List[str]] | None = None,
) -> pathlib.Path:
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Representación BM25 inválida: {representation!r}; "
            f"opciones: {', '.join(REPRESENTATIONS)}"
        )
    phrase_representations = {"curated", "enriched_paraphrases"}
    if representation in phrase_representations and search_phrases is None:
        raise ValueError(
            f"La representación {representation} requiere un archivo de frases"
        )
    if representation not in phrase_representations and search_phrases is not None:
        raise ValueError(
            "Las frases de búsqueda solo se usan con representation=curated "
            "o enriched_paraphrases"
        )
    known_aliases = {str(record["id"]) for record in records}
    unknown_aliases = set(search_phrases or ()) - known_aliases
    if unknown_aliases:
        raise ValueError(
            "Aliases sin documento en el corpus: " + ", ".join(sorted(unknown_aliases))
        )
    if representation == "enriched_paraphrases":
        phrase_categories = {
            alias.partition(":")[0] for alias in (search_phrases or ())
        }
        missing_aliases = {
            str(record["id"])
            for record in records
            if str(record["id"]).partition(":")[0] in phrase_categories
            and str(record["id"]) not in (search_phrases or {})
        }
        if missing_aliases:
            raise ValueError(
                "Aliases sin entrada de paráfrasis: "
                + ", ".join(sorted(missing_aliases))
            )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmp_dir / "corpus.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as out:
        for rec in records:
            doc = {
                "id": rec["id"],
                "contents": _contents_for_record(
                    rec,
                    representation,
                    (search_phrases or {}).get(str(rec["id"])),
                ),
            }
            for field in ("command", "intent", "example"):
                if field in rec:
                    doc[field] = rec[field]
            if representation in {"curated", "enriched_paraphrases"}:
                doc["search_phrases"] = (search_phrases or {}).get(str(rec["id"]), [])
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return jsonl_path


def _load_search_phrases(
    paths: pathlib.Path | List[pathlib.Path],
    *,
    allow_empty: bool = False,
    deduplicate: bool = False,
) -> Dict[str, List[str]]:
    if isinstance(paths, pathlib.Path):
        paths = [paths]
    phrases: Dict[str, List[str]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"No se pudo cargar el archivo de frases {path}: {error}"
            ) from error
        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                f"El archivo de frases {path} debe ser un objeto alias -> lista de frases"
            )
        for alias, values in payload.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Cada alias de frase debe ser una cadena no vacía")
            if alias in phrases:
                raise ValueError(f"Alias duplicado en archivos de frases: {alias}")
            if not isinstance(values, list) or (
                not allow_empty and not values
            ) or not all(isinstance(value, str) and value.strip() for value in values):
                qualifier = "una lista de textos" if allow_empty else "una lista no vacía"
                raise ValueError(f"Las frases para {alias!r} deben ser {qualifier}")
            normalized = [value.strip() for value in values]
            if deduplicate:
                normalized = list(dict.fromkeys(normalized))
            elif len(set(normalized)) != len(normalized):
                raise ValueError(f"Las frases para {alias!r} no deben repetirse")
            phrases[alias] = normalized
    return phrases


def build_bm25_index(
    corpus_path: pathlib.Path = CORPUS_PATH,
    index_dir: pathlib.Path = OUTPUT_INDEX_PATH,
    threads: int = BASE_THREADS,
    representation: str = "command",
    search_phrases_path: pathlib.Path | List[pathlib.Path] | None = None,
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
    phrase_representations = {"curated", "enriched_paraphrases"}
    if representation in phrase_representations and search_phrases_path is None:
        raise ValueError(
            f"--search-phrases es obligatorio con --representation {representation}"
        )
    if representation not in phrase_representations and search_phrases_path is not None:
        raise ValueError(
            "--search-phrases requiere --representation curated "
            "o enriched_paraphrases"
        )
    search_phrases = (
        _load_search_phrases(
            search_phrases_path,
            allow_empty=representation == "enriched_paraphrases",
            deduplicate=representation == "enriched_paraphrases",
        )
        if search_phrases_path is not None
        else None
    )

    # -----------------------------------------------------------------
    # 2️⃣  Preparar el archivo JSONL temporal que Pyserini consume
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="it-shell-bm25-input-") as temp_dir:
        tmp_input = pathlib.Path(temp_dir)
        _prepare_input(records, tmp_input, representation, search_phrases)

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
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="command",
        help=(
            "Text indexed in BM25: command preserves the legacy representation; "
            f"enriched indexes command, up to {MAX_INTENT_CHARS} characters of intent, "
            "and example; curated indexes command and manually reviewed search phrases; "
            "enriched_paraphrases adds search paraphrases to the enriched content."
        ),
    )
    parser.add_argument(
        "--search-phrases",
        type=pathlib.Path,
        nargs="+",
        help=(
            "One or more JSON files mapping corpus aliases to search phrases. "
            "Required by curated and enriched_paraphrases."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    build_bm25_index(
        args.corpus,
        args.output,
        args.threads,
        args.representation,
        args.search_phrases,
    )