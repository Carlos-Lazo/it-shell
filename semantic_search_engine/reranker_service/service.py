class RerankerService:
    def __init__(self, reranker): self.reranker=reranker
    def rerank(self, query, candidates): return self.reranker.rank(query, candidates)
