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

def create_engine(config_path: Path):
    config=load_config(config_path)
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
    )
    reranker=CrossEncoderReranker(config.reranker_model) if config.reranker_model else IdentityReranker()
    pipeline=PipelineBuilderService(CommandResolver(config.corpus_path), PipelineBuilder(config.selection), ProposalValidator())
    return SemanticSearchEngine(retrieval, RerankerService(reranker), pipeline, PromptBuilderService(PromptBuilder()), ResponseFormatterService())
