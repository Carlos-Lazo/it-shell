import logging

from semantic_search_engine.llm.ollama_assistant import (
    CandidateOption,
    OllamaConnectionError,
    OllamaResponseError,
)
from semantic_search_engine.retrieval_service.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


class PipelineBuilderService:
    def __init__(self, resolver, builder, validator, candidate_assistant=None):
        self.resolver, self.builder, self.validator = resolver, builder, validator
        self.candidate_assistant = candidate_assistant

    def propose(self, plan, retrieved, reranker_service):
        ranked={}
        for step in plan.steps:
            candidates=[self.resolver.resolve(alias, score) for alias, score in retrieved[step.id]]
            step_candidates = reranker_service.rerank(step.query, candidates)
            ranked[step.id] = self._select_candidate(step.query, step.arguments, step_candidates)
        return self.validator.validate(self.builder.build(plan, ranked))

    def _select_candidate(self, query, arguments, candidates):
        if self.candidate_assistant is None:
            return candidates
        eligible = [
            candidate
            for candidate in candidates
            if candidate.retrieval_score >= self.builder.selection.min_score
        ]
        if not eligible:
            return candidates

        options = [
            CandidateOption(
                index=index,
                alias=candidate.alias,
                command=candidate.example or candidate.command,
                intent=candidate.intent,
                example=candidate.example,
                score=candidate.score,
            )
            for index, candidate in enumerate(eligible, start=1)
        ]
        language = QueryAnalyzer().analyze(query).language
        try:
            selection = self.candidate_assistant.select_and_synthesize(
                query=query,
                step_task=query,
                candidates=options,
                arguments=dict(arguments),
                language=language,
            )
        except (OllamaConnectionError, OllamaResponseError) as error:
            logger.warning(
                "Ollama candidate selection failed (%s); preserving retrieval order.",
                error,
            )
            return candidates

        selected = next(
            (
                candidate
                for candidate in eligible
                if candidate.alias == selection.selected_alias
            ),
            None,
        )
        if selected is None:
            if selection.selected_alias is not None:
                logger.warning(
                    "Ollama selected an alias outside the candidate set; "
                    "preserving retrieval order."
                )
            return candidates
        return [selected, *(candidate for candidate in candidates if candidate is not selected)]
