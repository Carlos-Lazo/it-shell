from pathlib import Path
from semantic_search.engine import SemanticSearchEngine
from semantic_search.shared.config import load_config
from semantic_search.retrieval_service.query_analyzer import QueryAnalyzer
from semantic_search.retrieval_service.planner import RulePlanner
from semantic_search.retrieval_service.retriever import FaissRetriever, SentenceTransformerEncoder
from semantic_search.retrieval_service.service import RetrievalService
from semantic_search.reranker_service.reranker import CrossEncoderReranker, IdentityReranker
from semantic_search.reranker_service.service import RerankerService
from semantic_search.pipeline_builder_service.command_resolver import CommandResolver
from semantic_search.pipeline_builder_service.pipeline_builder import PipelineBuilder
from semantic_search.pipeline_builder_service.validator import ProposalValidator
from semantic_search.pipeline_builder_service.service import PipelineBuilderService
from semantic_search.prompt_builder_service.prompt_builder import PromptBuilder
from semantic_search.prompt_builder_service.service import PromptBuilderService
from semantic_search.response_formatter_service.service import ResponseFormatterService

def create_engine(config_path: Path):
    config=load_config(config_path)
    retrieval=RetrievalService(QueryAnalyzer(), RulePlanner(), FaissRetriever(config.index_path, config.aliases_path, SentenceTransformerEncoder(config.embedding_model)))
    reranker=CrossEncoderReranker(config.reranker_model) if config.reranker_model else IdentityReranker()
    pipeline=PipelineBuilderService(CommandResolver(config.corpus_path), PipelineBuilder(config.selection), ProposalValidator())
    return SemanticSearchEngine(retrieval, RerankerService(reranker), pipeline, PromptBuilderService(PromptBuilder()), ResponseFormatterService())
