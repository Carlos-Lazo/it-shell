from unittest.mock import MagicMock

from evaluation.compare_bge_ollama import (
    RecordingOllamaAssistant,
    query_metrics,
    render_markdown,
    summarize_records,
)
from semantic_search_engine.llm.ollama_assistant import CandidateOption


def test_query_metrics_scores_candidates_and_final_selection_by_command():
    metrics = query_metrics(
        ["filesystem:1", "filesystem:2", "networking:1"],
        "mkdir",
        {"mkdir"},
        {
            "filesystem:1": "touch",
            "filesystem:2": "mkdir",
            "networking:1": "ip",
        },
    )

    assert metrics["command_hit@1"] == 0.0
    assert metrics["command_hit@5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["selected_command_correct"] == 1.0
    assert metrics["selected_given_hit@15"] == 1.0


def test_query_metrics_marks_missing_or_incorrect_final_selection():
    metrics = query_metrics(
        ["networking:1"],
        "ip",
        {"ss"},
        {"networking:1": "ip"},
    )

    assert metrics["candidate_hit@15"] == 0.0
    assert metrics["selected_command_correct"] == 0.0
    assert metrics["selected_given_hit@15"] == 0.0


def test_summarize_records_aggregates_correctness_and_latency():
    summary = summarize_records([
        {
            "selected_command_correct": 1.0,
            "selected_given_hit@15": 1.0,
            "candidate_hit@15": 1.0,
            "mrr": 0.5,
            "command_hit@1": 0.0,
            "command_hit@5": 1.0,
            "command_hit@10": 1.0,
            "command_hit@15": 1.0,
            "latency_seconds": 2.0,
        },
        {
            "selected_command_correct": 0.0,
            "selected_given_hit@15": 0.0,
            "candidate_hit@15": 0.0,
            "mrr": 0.0,
            "command_hit@1": 0.0,
            "command_hit@5": 0.0,
            "command_hit@10": 0.0,
            "command_hit@15": 0.0,
            "latency_seconds": 4.0,
        },
    ])

    assert summary["queries"] == 2
    assert summary["selected_command_correct"] == 0.5
    assert summary["command_hit@10"] == 0.5
    assert summary["mean_latency_seconds"] == 3.0


def test_render_markdown_calls_out_test_and_bge_provenance_limit():
    report = {
        "query_counts": {"total": 122, "development": 86, "test": 36},
        "provenance": {
            "index_vectors": 27413,
            "embedding_dimension": 1024,
            "ollama": {
                "version": "0.40.13",
                "model": "qwen2.5:1.5b",
                "digest": "abc123",
            },
            "bge_corpus_note": "BGE source hashes unavailable.",
        },
        "methods": {
            method: {
                "summary": {
                    "command_hit@1": 0.1,
                    "command_hit@5": 0.3,
                    "command_hit@10": 0.4,
                    "mrr": 0.2,
                    "selected_command_correct": 0.25,
                    "mean_latency_seconds": 1,
                },
                "by_partition_and_language": {},
                "assistant_chat_calls": 0,
                "fallbacks": {"selection": 0},
            }
            for method in ("rules", "ollama")
        },
    }

    rendered = render_markdown(report)

    assert "86 development, 36 test" in rendered
    assert "test se reporta con parámetros fijos" in rendered
    assert "BGE source hashes unavailable." in rendered


def test_recording_assistant_uses_short_schema_and_spanish_selection_prompt():
    assistant = RecordingOllamaAssistant()
    assistant._chat_request = MagicMock(return_value={"selected_index": 2})
    candidates = [
        CandidateOption(1, "filesystem:1", "touch", "create a file"),
        CandidateOption(2, "filesystem:2", "mkdir -p", "create parent directories"),
    ]

    selected_alias = assistant.select_candidate(
        "crear directorios padre",
        candidates,
        "es",
    )

    assert selected_alias == "filesystem:2"
    request = assistant._chat_request.call_args
    assert request.kwargs["schema"]["required"] == ["selected_index"]
    assert request.kwargs["num_predict"] == 24
    assert "Elegí el candidato" in request.args[0][0]["content"]
