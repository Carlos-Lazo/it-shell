from pathlib import Path
from semantic_search_engine.engine import SemanticSearchEngine
from semantic_search_engine.shared.config import load_config
from semantic_search_engine.retrieval_service.query_analyzer import QueryAnalyzer
from semantic_search_engine.retrieval_service.planner import RulePlanner
from semantic_search_engine.retrieval_service.retriever import FaissRetriever, SentenceTransformerEncoder
from semantic_search_engine.retrieval_service.service import RetrievalService
from semantic_search_engine.reranker_service.reranker import CrossEncoderReranker, IdentityReranker
from semantic_search_engine.reranker_service.service import RerankerService
from semantic_search_engine.pipeline_builder_service.command_resolver import CommandResolver
from semantic_search_engine.pipeline_builder_service.pipeline_builder import PipelineBuilder
from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator
from semantic_search_engine.pipeline_builder_service.service import PipelineBuilderService
from semantic_search_engine.prompt_builder_service.prompt_builder import PromptBuilder
from semantic_search_engine.prompt_builder_service.service import PromptBuilderService
from semantic_search_engine.response_formatter_service.service import ResponseFormatterService
from semantic_search_engine.llm.ollama_assistant import OllamaAssistant


def create_engine(config_path: Path, assistant_backend: str | None = None):
    config=load_config(config_path)
    selected_assistant = (
        config.assistant_backend
        if assistant_backend is None
        else assistant_backend
    )
    if not isinstance(selected_assistant, str) or selected_assistant not in {"rules", "ollama"}:
        raise ValueError("assistant_backend debe ser 'rules' u 'ollama'")
    assistant = (
        OllamaAssistant(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            timeout=config.ollama_timeout,
        )
        if selected_assistant == "ollama"
        else None
    )
    encoder = SentenceTransformerEncoder(config.embedding_model)
    if config.retrieval_mode == "hybrid":
        if config.bm25_index_path is None:
            raise ValueError("bm25_index_path es obligatorio con retrieval_mode='hybrid'")
        from semantic_search_engine.retrieval_service.bm25_search import HybridBM25Retriever

        retriever = HybridBM25Retriever.from_paths(
            config.bm25_index_path,
            config.index_path,
            config.aliases_path,
            encoder,
            top_k=config.retrieval_top_k,
            alpha=config.hybrid_alpha,
            allowed_categories=config.allowed_categories,
        )
    else:
        retriever = FaissRetriever(
            config.index_path,
            config.aliases_path,
            encoder,
            config.allowed_categories,
        )
    retrieval=RetrievalService(
        QueryAnalyzer(),
        RulePlanner(),
        retriever,
        query_assistant=assistant,
    )
    reranker=CrossEncoderReranker(config.reranker_model) if config.reranker_model else IdentityReranker()
    pipeline=PipelineBuilderService(
        CommandResolver(config.corpus_path),
        PipelineBuilder(config.selection),
        ProposalValidator(),
        candidate_assistant=assistant,
    )
    prompts = PromptBuilderService(PromptBuilder(), llm_client=assistant)
    return SemanticSearchEngine(
        retrieval,
        RerankerService(reranker),
        pipeline,
        prompts,
        ResponseFormatterService(),
    )
