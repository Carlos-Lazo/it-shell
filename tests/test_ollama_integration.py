from unittest.mock import MagicMock, patch

import pytest
import yaml

from semantic_search_engine.bootstrap import create_engine
from semantic_search_engine.llm.ollama_assistant import (
    AnalysisStep,
    LLMQueryAnalysis,
    LLMSelectionResult,
    OllamaAssistant,
)
from semantic_search_engine.pipeline_builder_service.pipeline_builder import PipelineBuilder
from semantic_search_engine.pipeline_builder_service.service import PipelineBuilderService
from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator
from semantic_search_engine.retrieval_service.service import RetrievalService
from semantic_search_engine.shared.config import EngineConfig, SelectionConfig, load_config
from semantic_search_engine.shared.contracts import CandidateCommand, PlanStep, QueryAnalysis, QueryPlan


def write_config(tmp_path, **values):
    root = tmp_path / "engine"
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    config = {
        "index_path": "index.faiss",
        "aliases_path": "aliases.npy",
        "corpus_path": "corpus",
        "embedding_model": "local-model",
        "selection": {},
        **values,
    }
    path = config_dir / "engine.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_config_defaults_to_rules_and_reads_ollama_settings(tmp_path):
    defaults = load_config(write_config(tmp_path / "defaults"))
    assert defaults.assistant_backend == "rules"
    assert defaults.ollama_model == "qwen2.5:1.5b"

    configured = load_config(write_config(
        tmp_path / "configured",
        assistant="ollama",
        ollama_model="qwen2.5:3b",
        ollama_base_url="http://127.0.0.1:11434/",
        ollama_timeout=12,
    ))
    assert configured.assistant_backend == "ollama"
    assert configured.ollama_model == "qwen2.5:3b"
    assert configured.ollama_base_url == "http://127.0.0.1:11434"
    assert configured.ollama_timeout == 12


@pytest.mark.parametrize(
    ("config_values", "message"),
    [
        ({"assistant": "unknown"}, "assistant debe ser"),
        ({"ollama_model": ""}, "ollama_model debe ser"),
        ({"ollama_base_url": ""}, "ollama_base_url debe ser"),
        ({"ollama_timeout": 0}, "ollama_timeout debe ser"),
    ],
)
def test_config_rejects_invalid_assistant_settings(tmp_path, config_values, message):
    with pytest.raises(ValueError, match=message):
        load_config(write_config(tmp_path, **config_values))


def test_bootstrap_wires_one_ollama_assistant_to_retrieval_selection_and_explanation():
    config = EngineConfig(
        index_path=MagicMock(),
        aliases_path=MagicMock(),
        corpus_path=MagicMock(),
        embedding_model="local-model",
        reranker_model=None,
        retrieval_top_k=15,
        selection=SelectionConfig(0.45, 0.04, 3),
        assistant_backend="ollama",
        ollama_model="qwen2.5:3b",
        ollama_base_url="http://localhost:11434",
        ollama_timeout=12,
    )
    with (
        patch("semantic_search_engine.bootstrap.load_config", return_value=config),
        patch("semantic_search_engine.bootstrap.SentenceTransformerEncoder"),
        patch("semantic_search_engine.bootstrap.FaissRetriever"),
        patch("semantic_search_engine.bootstrap.RetrievalService") as retrieval_service,
        patch("semantic_search_engine.bootstrap.IdentityReranker"),
        patch("semantic_search_engine.bootstrap.RerankerService"),
        patch("semantic_search_engine.bootstrap.CommandResolver"),
        patch("semantic_search_engine.bootstrap.PipelineBuilder"),
        patch("semantic_search_engine.bootstrap.ProposalValidator"),
        patch("semantic_search_engine.bootstrap.PipelineBuilderService") as pipeline_service,
        patch("semantic_search_engine.bootstrap.PromptBuilder"),
        patch("semantic_search_engine.bootstrap.PromptBuilderService") as prompt_service,
        patch("semantic_search_engine.bootstrap.ResponseFormatterService"),
        patch("semantic_search_engine.bootstrap.OllamaAssistant") as assistant_factory,
    ):
        create_engine(MagicMock())

    assistant_factory.assert_called_once_with(
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
        timeout=12,
    )
    assistant = assistant_factory.return_value
    assert retrieval_service.call_args.kwargs["query_assistant"] is assistant
    assert pipeline_service.call_args.kwargs["candidate_assistant"] is assistant
    assert prompt_service.call_args.kwargs["llm_client"] is assistant


def test_bootstrap_rejects_invalid_assistant_override():
    config = EngineConfig(
        index_path=MagicMock(),
        aliases_path=MagicMock(),
        corpus_path=MagicMock(),
        embedding_model="local-model",
        reranker_model=None,
        retrieval_top_k=15,
        selection=SelectionConfig(0.45, 0.04, 3),
    )
    with patch("semantic_search_engine.bootstrap.load_config", return_value=config):
        with pytest.raises(ValueError, match="assistant_backend debe ser"):
            create_engine(MagicMock(), assistant_backend="invalid")


def test_retrieval_uses_assistant_search_query_without_changing_deterministic_plan():
    query = "buscar archivos grandes"
    rule_analysis = QueryAnalysis(query, query, "es", False)
    plan = QueryPlan(query, (PlanStep("step_1", query),))
    retriever = MagicMock()
    assistant = MagicMock()
    assistant.analyze_query.return_value = LLMQueryAnalysis(
        original_query=query,
        predicted_category="filesystem",
        category_confidence="high",
        language="es",
        steps=[AnalysisStep(1, "buscar archivos", "find large files")],
        raw_response={},
    )
    retrieval = RetrievalService(
        MagicMock(analyze=MagicMock(return_value=rule_analysis)),
        MagicMock(plan=MagicMock(return_value=plan)),
        retriever,
        query_assistant=assistant,
    )

    result_analysis, result_plan, _ = retrieval.retrieve_plan(query, 10)

    assert result_analysis is rule_analysis
    assert result_plan is plan
    retriever.retrieve.assert_called_once_with("find large files", 10)


def test_pipeline_uses_only_assistant_selected_candidate_and_validates_command():
    candidates = {
        "filesystem:1": CandidateCommand(
            "filesystem:1", "filesystem", 1, "create a file", "touch", "touch <path>", 0.9
        ),
        "filesystem:2": CandidateCommand(
            "filesystem:2", "filesystem", 2, "create directories", "mkdir -p", "mkdir -p <path>", 0.8
        ),
    }
    resolver = MagicMock(resolve=MagicMock(side_effect=lambda alias, score: candidates[alias]))
    assistant = MagicMock()
    assistant.select_and_synthesize.return_value = LLMSelectionResult(
        selected_index=2,
        selected_alias="filesystem:2",
        confidence="high",
        assembled_command="rm -rf /",
        explanation="",
    )
    service = PipelineBuilderService(
        resolver,
        PipelineBuilder(SelectionConfig(0.45, 0.04, 3)),
        ProposalValidator(),
        candidate_assistant=assistant,
    )
    plan = QueryPlan(
        "crear directorio",
        (PlanStep("step_1", "crear directorio en /tmp/output", arguments=(("path", "/tmp/output"),)),),
    )

    proposal = service.propose(
        plan,
        {"step_1": [("filesystem:1", 0.9), ("filesystem:2", 0.8)]},
        MagicMock(rerank=MagicMock(side_effect=lambda query, items: items)),
    )

    step = proposal.steps[0]
    assert step.selected.alias == "filesystem:2"
    assert step.command_template == "mkdir -p /tmp/output"
    assert proposal.executable is False
    assert not any(issue.code == "destructive_command" for issue in proposal.issues)


def test_ollama_assistant_adapts_explanation_to_llm_client():
    assistant = OllamaAssistant()
    assistant._chat_request = MagicMock(return_value={"explanation": "Explicación breve"})

    assert assistant.generate("Contexto validado") == "Explicación breve"
    assert assistant._chat_request.call_args.kwargs["schema"]["required"] == ["explanation"]
