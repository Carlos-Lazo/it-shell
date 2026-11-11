import logging

from semantic_search_engine.llm.ollama_assistant import (
    OllamaConnectionError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, analyzer, planner, retriever, query_assistant=None):
        self.analyzer, self.planner, self.retriever = analyzer, planner, retriever
        self.query_assistant = query_assistant

    def retrieve_plan(self, query, top_k):
        analysis=self.analyzer.analyze(query)
        plan=self.planner.plan(analysis)
        retrieval_queries = {step.id: step.query for step in plan.steps}
        if self.query_assistant is not None:
            try:
                assistant_analysis = self.query_assistant.analyze_query(query)
            except (OllamaConnectionError, OllamaResponseError) as error:
                logger.warning(
                    "Ollama query analysis failed (%s); using deterministic queries.",
                    error,
                )
            else:
                assistant_steps = assistant_analysis.steps
                if (
                    not (assistant_analysis.raw_response or {}).get("fallback")
                    and len(assistant_steps) == len(plan.steps)
                ):
                    for step, assistant_step in zip(
                        plan.steps, assistant_steps, strict=True
                    ):
                        search_query = assistant_step.search_query.strip()
                        if search_query:
                            retrieval_queries[step.id] = search_query
                elif not (assistant_analysis.raw_response or {}).get("fallback"):
                    logger.warning(
                        "Ollama returned %d search steps for a deterministic plan "
                        "with %d steps; using deterministic queries.",
                        len(assistant_steps),
                        len(plan.steps),
                    )
        return analysis, plan, {
            step.id: self.retriever.retrieve(retrieval_queries[step.id], top_k)
            for step in plan.steps
        }
