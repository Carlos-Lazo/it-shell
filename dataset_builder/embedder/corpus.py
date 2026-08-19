"""Carga y valida los documentos YAML producidos por el clasificador."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    """Lee todos los YAML del corpus y verifica IDs únicos entre categorías."""
    documents: list[CorpusDocument] = []
    seen_ids: set[str] = set()
    for path in sorted(corpus_dir.glob("*.yaml")):
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
            documents.append(CorpusDocument(
                record_id=record_id,
                category=category,
                source_id=source_id,
                intent=str(row.get("intent", "")).strip(),
                command=str(row.get("command", "")).strip(),
                example=str(row.get("example", "")).strip(),
            ))
    if not documents:
        raise ValueError(f"No se encontraron documentos en {corpus_dir}")
    return documents