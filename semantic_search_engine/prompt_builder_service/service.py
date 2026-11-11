import logging

from semantic_search_engine.prompt_builder_service.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)


class PromptBuilderService:
    def __init__(self, builder, llm_client: LLMClient | None = None):
        self.builder = builder
        self.llm_client = llm_client

    def clarification(self, proposal):
        return self.builder.clarification(proposal)

    def explain(self, proposal) -> str:
        if self.llm_client is not None:
            prompt = self.builder.explanation_prompt(proposal)
            try:
                explanation = self.llm_client.generate(prompt)
            except LLMClientError as error:
                logger.warning(
                    "Optional explanation provider failed (%s); using deterministic explanation.",
                    error,
                )
                return self.builder.offline_explanation(proposal)
            if explanation.strip():
                return explanation.strip()
        return self.builder.offline_explanation(proposal)
