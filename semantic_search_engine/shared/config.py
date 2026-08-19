from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class SelectionConfig:
    min_score: float
    alternative_margin: float
    max_alternatives: int

@dataclass(frozen=True)
class EngineConfig:
    index_path: Path
    aliases_path: Path
    corpus_path: Path
    embedding_model: str
    reranker_model: str | None
    retrieval_top_k: int
    selection: SelectionConfig

def load_config(path: Path) -> EngineConfig:
    data=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    root=path.parent.parent
    select=data.get('selection', {})
    return EngineConfig(
        index_path=(root/data['index_path']).resolve(),
        aliases_path=(root/data['aliases_path']).resolve(),
        corpus_path=(root/data['corpus_path']).resolve(),
        embedding_model=data['embedding_model'],
        reranker_model=data.get('reranker_model'),
        retrieval_top_k=int(data.get('retrieval_top_k', 50)),
        selection=SelectionConfig(float(select.get('min_score', 0)), float(select.get('alternative_margin', .04)), int(select.get('max_alternatives', 3))),
    )
