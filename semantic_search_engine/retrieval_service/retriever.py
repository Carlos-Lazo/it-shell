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
    def __init__(self, index_path: Path, aliases_path: Path, encoder: Encoder):
        self.index=faiss.read_index(str(index_path))
        self.aliases=np.load(aliases_path, allow_pickle=False).astype(str)
        self.encoder=encoder
        if self.index.ntotal != len(self.aliases): raise ValueError('Índice y aliases no están alineados')
    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores, positions=self.index.search(self.encoder.encode([query]), min(top_k, self.index.ntotal))
        return [(self.aliases[pos], float(score)) for score, pos in zip(scores[0], positions[0], strict=True)]
