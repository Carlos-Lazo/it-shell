from semantic_search_engine.shared.contracts import QueryRequest, SearchResponse
class SemanticSearchEngine:
    def __init__(self, retrieval, reranker, pipeline, prompts, formatter): self.retrieval, self.reranker, self.pipeline, self.prompts, self.formatter=retrieval, reranker, pipeline, prompts, formatter
    def search(self, request: QueryRequest):
        analysis, plan, retrieved=self.retrieval.retrieve_plan(request.query, request.retrieval_top_k)
        proposal=self.pipeline.propose(plan, retrieved, self.reranker)
        response=SearchResponse(
            analysis,
            plan,
            proposal,
            self.prompts.clarification(proposal),
            self.prompts.explain(proposal),
        )
        return self.formatter.format(response, request.output_format)
