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
    allowed_categories: tuple[str, ...] | None = None
    retrieval_mode: str = "dense"
    bm25_index_path: Path | None = None
    bm25_search_phrases_path: Path | None = None
    hybrid_alpha: float = 0.5
    assistant_backend: str = "rules"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 45.0

def load_config(path: Path) -> EngineConfig:
    data=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    root=path.parent.parent
    select=data.get('selection', {})
    raw_allowed_categories = data.get('allowed_categories')
    if raw_allowed_categories is not None:
        if (
            not isinstance(raw_allowed_categories, list)
            or not raw_allowed_categories
            or not all(isinstance(category, str) and category.strip() for category in raw_allowed_categories)
        ):
            raise ValueError(
                "allowed_categories debe ser una lista no vacía de nombres de categoría"
            )
        if len(set(raw_allowed_categories)) != len(raw_allowed_categories):
            raise ValueError("allowed_categories no debe contener duplicados")
        allowed_categories = tuple(raw_allowed_categories)
    else:
        allowed_categories = None
    retrieval_mode = data.get('retrieval_mode', 'dense')
    if retrieval_mode not in {'dense', 'hybrid'}:
        raise ValueError("retrieval_mode debe ser 'dense' o 'hybrid'")
    raw_bm25_index_path = data.get('bm25_index_path')
    if raw_bm25_index_path is not None and (
        not isinstance(raw_bm25_index_path, str) or not raw_bm25_index_path.strip()
    ):
        raise ValueError("bm25_index_path debe ser una ruta no vacía")
    if retrieval_mode == 'hybrid' and raw_bm25_index_path is None:
        raise ValueError("bm25_index_path es obligatorio con retrieval_mode='hybrid'")
    raw_search_phrases_path = data.get('bm25_search_phrases_path')
    if raw_search_phrases_path is not None and (
        not isinstance(raw_search_phrases_path, str) or not raw_search_phrases_path.strip()
    ):
        raise ValueError("bm25_search_phrases_path debe ser una ruta no vacía")
    search_phrases_path = (
        (root/raw_search_phrases_path).resolve()
        if raw_search_phrases_path is not None
        else None
    )
    if search_phrases_path is not None and not search_phrases_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de frases BM25: {search_phrases_path}"
        )
    raw_hybrid_alpha = data.get('hybrid_alpha', 0.5)
    if isinstance(raw_hybrid_alpha, bool) or not isinstance(raw_hybrid_alpha, (int, float)):
        raise ValueError("hybrid_alpha debe ser un número entre 0 y 1")
    hybrid_alpha = float(raw_hybrid_alpha)
    if not 0 <= hybrid_alpha <= 1:
        raise ValueError("hybrid_alpha debe ser un número entre 0 y 1")
    assistant_backend = data.get('assistant', 'rules')
    if not isinstance(assistant_backend, str) or assistant_backend not in {'rules', 'ollama'}:
        raise ValueError("assistant debe ser 'rules' u 'ollama'")
    ollama_model = data.get('ollama_model', 'qwen2.5:1.5b')
    if not isinstance(ollama_model, str) or not ollama_model.strip():
        raise ValueError("ollama_model debe ser un nombre de modelo no vacío")
    ollama_base_url = data.get('ollama_base_url', 'http://localhost:11434')
    if not isinstance(ollama_base_url, str) or not ollama_base_url.strip():
        raise ValueError("ollama_base_url debe ser una URL no vacía")
    raw_ollama_timeout = data.get('ollama_timeout', 45.0)
    if (
        isinstance(raw_ollama_timeout, bool)
        or not isinstance(raw_ollama_timeout, (int, float))
        or raw_ollama_timeout <= 0
    ):
        raise ValueError("ollama_timeout debe ser un número positivo")
    return EngineConfig(
        index_path=(root/data['index_path']).resolve(),
        aliases_path=(root/data['aliases_path']).resolve(),
        corpus_path=(root/data['corpus_path']).resolve(),
        embedding_model=data['embedding_model'],
        reranker_model=data.get('reranker_model'),
        retrieval_top_k=int(data.get('retrieval_top_k', 50)),
        selection=SelectionConfig(float(select.get('min_score', 0)), float(select.get('alternative_margin', .04)), int(select.get('max_alternatives', 3))),
        allowed_categories=allowed_categories,
        retrieval_mode=retrieval_mode,
        bm25_index_path=(
            (root/raw_bm25_index_path).resolve()
            if raw_bm25_index_path is not None
            else None
        ),
        bm25_search_phrases_path=search_phrases_path,
        hybrid_alpha=hybrid_alpha,
        assistant_backend=assistant_backend,
        ollama_model=ollama_model.strip(),
        ollama_base_url=ollama_base_url.strip().rstrip('/'),
        ollama_timeout=float(raw_ollama_timeout),
    )
