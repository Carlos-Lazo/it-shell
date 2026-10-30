from semantic_search_engine.shared.contracts import QueryRequest, SearchResponse
class SemanticSearchEngine:
    def __init__(self, retrieval, reranker, pipeline, prompts, formatter): self.retrieval, self.reranker, self.pipeline, self.prompts, self.formatter=retrieval, reranker, pipeline, prompts, formatter

    def search_response(self, request: QueryRequest) -> SearchResponse:
        analysis, plan, retrieved=self.retrieval.retrieve_plan(request.query, request.retrieval_top_k)
        proposal=self.pipeline.propose(plan, retrieved, self.reranker)
        return SearchResponse(
            analysis,
            plan,
            proposal,
            self.prompts.clarification(proposal),
            self.prompts.explain(proposal),
        )

    def search(self, request: QueryRequest):
        response = self.search_response(request)
        return self.formatter.format(response, request.output_format)
