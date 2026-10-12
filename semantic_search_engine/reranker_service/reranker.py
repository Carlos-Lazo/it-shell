from dataclasses import replace

class IdentityReranker:
    """Modo inicial reproducible: conserva el ranking del retrieval."""
    def rank(self, query, candidates): return candidates

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name, self._model = model_name, None

    def rank(self, query, candidates):
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, candidate.intent) for candidate in candidates])
        return sorted(
            (replace(item, reranker_score=float(score)) for item, score in zip(candidates, scores, strict=True)),
            key=lambda item: item.score,
            reverse=True,
        )
