from __future__ import annotations

import yaml
import pytest

from evaluation.evaluate_capacity_retrieval import load_capacity_queries
from evaluation.evaluate_capacity_retrieval import (
    score_rankings,
    select_hybrid_alpha,
    summarize_method,
)


def write_yaml(path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_load_capacity_queries_uses_source_alias_command_and_partition(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_yaml(corpus / "filesystem.yaml", {
        "category": "filesystem",
        "documents": [{"id": 10, "command": "du", "intent": "estimate usage"}],
    })
    capacities = tmp_path / "capacities.yaml"
    write_yaml(capacities, {
        "schema_version": 1,
        "metadata": {"query_review_status": "reviewed"},
        "categories": [{
            "category": "filesystem",
            "capacities": [{
                "capability_id": "filesystem.du",
                "source_aliases": ["filesystem:10"],
                "partition": "test",
                "queries": [{
                    "text": "How much disk space does this use?",
                    "language": "en",
                    "status": "reviewed",
                }],
            }],
        }],
    })

    queries = load_capacity_queries(capacities, corpus, {"filesystem:10"})

    assert queries == [{
        "id": "filesystem.du::1",
        "query": "How much disk space does this use?",
        "language": "en",
        "partition": "test",
        "category": "filesystem",
        "capability_id": "filesystem.du",
        "relevant_commands": ["du"],
    }]


def test_select_hybrid_alpha_uses_development_metrics_and_stable_tiebreaks():
    summaries = {
        0.25: {"command_hit@10": 0.7, "mrr": 0.4, "ndcg@10": 0.3},
        0.5: {"command_hit@10": 0.7, "mrr": 0.4, "ndcg@10": 0.3},
        0.75: {"command_hit@10": 0.7, "mrr": 0.4, "ndcg@10": 0.3},
    }

    assert select_hybrid_alpha(summaries) == 0.5
    summaries[0.25]["mrr"] = 0.41
    assert select_hybrid_alpha(summaries) == 0.25


def test_score_rankings_credits_all_aliases_of_relevant_command():
    queries = [{
        "id": "filesystem.du::1",
        "query": "check disk usage",
        "relevant_commands": ["du"],
        "partition": "development",
        "category": "filesystem",
    }]
    rankings = {
        "filesystem.du::1": ["filesystem:1", "filesystem:2", "networking:3"],
    }
    command_aliases = {
        "du": {"filesystem:1", "filesystem:2"},
    }

    records = score_rankings(queries, rankings, command_aliases, "test")

    assert records[0]["first_relevant_rank"] == 1
    assert records[0]["command_hit@1"] == 1.0
    assert records[0]["command_hit@3"] == 1.0


def test_summarize_method_includes_language_metrics():
    records = [{
        "id": "filesystem.du::1",
        "language": "es",
        "partition": "development",
        "category": "filesystem",
        "command_hit@1": 1.0,
        "command_hit@3": 1.0,
        "command_hit@5": 1.0,
        "command_hit@10": 1.0,
        "ndcg@1": 1.0,
        "ndcg@3": 1.0,
        "ndcg@5": 1.0,
        "ndcg@10": 1.0,
        "mrr": 1.0,
    }]

    summary = summarize_method(records)

    assert summary["language_metrics"]["es"]["queries"] == 1
    assert summary["language_metrics"]["es"]["command_hit@10"] == 1.0


@pytest.mark.parametrize(
    ("partition", "status", "indexed_aliases", "error"),
    [
        ("unassigned", "reviewed", {"filesystem:10"}, "invalid partition"),
        ("test", "pending_review", {"filesystem:10"}, "not marked reviewed"),
        ("test", "reviewed", set(), "not in the FAISS index"),
    ],
)
def test_load_capacity_queries_rejects_unscorable_cases(
    tmp_path, partition, status, indexed_aliases, error
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_yaml(corpus / "filesystem.yaml", {
        "category": "filesystem",
        "documents": [{"id": 10, "command": "du", "intent": "estimate usage"}],
    })
    capacities = tmp_path / "capacities.yaml"
    write_yaml(capacities, {
        "schema_version": 1,
        "metadata": {"query_review_status": "reviewed"},
        "categories": [{
            "category": "filesystem",
            "capacities": [{
                "capability_id": "filesystem.du",
                "source_aliases": ["filesystem:10"],
                "partition": partition,
                "queries": [{
                    "text": "check disk usage",
                    "language": "en",
                    "status": status,
                }],
            }],
        }],
    })

    with pytest.raises(ValueError, match=error):
        load_capacity_queries(capacities, corpus, indexed_aliases)
