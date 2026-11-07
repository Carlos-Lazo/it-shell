from pathlib import Path
from typing import Protocol
import faiss
import numpy as np

class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...

class SentenceTransformerEncoder:
    def __init__(self, model_name: str): self.model_name, self._model = model_name, None
    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model=SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

class FaissRetriever:
    def __init__(
        self,
        index_path: Path,
        aliases_path: Path,
        encoder: Encoder,
        allowed_categories: tuple[str, ...] | None = None,
        default_boosted_categories: tuple[str, ...] | None = None,
        default_category_boost: float = 0.0,
    ):
        self.index=faiss.read_index(str(index_path))
        self.aliases=np.load(aliases_path, allow_pickle=False).astype(str)
        self.encoder=encoder
        self.allowed_categories = (
            frozenset(allowed_categories) if allowed_categories is not None else None
        )
        self.default_boosted_categories = (
            frozenset(default_boosted_categories)
            if default_boosted_categories is not None
            else None
        )
        self.default_category_boost = default_category_boost
        if self.index.ntotal != len(self.aliases): raise ValueError('Índice y aliases no están alineados')
        if self.allowed_categories is not None:
            indexed_categories = {self._category(alias) for alias in self.aliases}
            unknown_categories = self.allowed_categories - indexed_categories
            if unknown_categories:
                raise ValueError(
                    "Categorías ausentes en aliases FAISS: "
                    + ", ".join(sorted(unknown_categories))
                )

    @staticmethod
    def _category(alias: str) -> str:
        category, separator, _ = alias.partition(":")
        if not separator or not category:
            raise ValueError(f"Alias FAISS inválido: {alias!r}")
        return category

    def retrieve(
        self,
        query: str,
        top_k: int,
        boosted_categories: tuple[str, ...] | frozenset[str] | None = None,
        category_boost: float = 0.0,
    ) -> list[tuple[str, float]]:
        default_boosted = getattr(self, "default_boosted_categories", None)
        default_boost = getattr(self, "default_category_boost", 0.0)

        effective_boosted = (
            frozenset(boosted_categories)
            if boosted_categories is not None
            else default_boosted
        )
        effective_boost = (
            category_boost if category_boost > 0.0 else default_boost
        )
        has_soft_boost = effective_boosted is not None and effective_boost > 0.0

        if self.allowed_categories is not None:
            search_k = self.index.ntotal
        elif has_soft_boost:
            # Over-retrieve candidates to allow boosted relevant candidates to climb
            search_k = min(max(top_k * 10, 100), self.index.ntotal)
        else:
            search_k = min(top_k, self.index.ntotal)

        scores, positions = self.index.search(
            self.encoder.encode([query]),
            search_k,
        )
        results = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            alias = str(self.aliases[position])
            cat = self._category(alias)
            if (
                self.allowed_categories is not None
                and cat not in self.allowed_categories
            ):
                continue

            final_score = float(score)
            if has_soft_boost and cat in effective_boosted:
                final_score += effective_boost

            results.append((alias, final_score))

        if has_soft_boost:
            results.sort(key=lambda item: item[1], reverse=True)

        return results[:top_k]
