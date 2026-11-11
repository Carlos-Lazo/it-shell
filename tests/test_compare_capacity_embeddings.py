import json

import numpy as np
import pytest

from evaluation.compare_capacity_embeddings import (
    corpus_matches_snapshot,
    rank_in_alias_universe,
)


class FakeIndex:
    ntotal = 5

    def search(self, vectors, top_k):
        positions = np.tile(np.arange(self.ntotal), (len(vectors), 1))
        scores = np.tile(np.arange(self.ntotal, 0, -1), (len(vectors), 1))
        return scores[:, :top_k], positions[:, :top_k]


def test_rank_in_alias_universe_filters_before_applying_top_k():
    aliases = np.asarray(["shared:a", "private:b", "private:c", "private:d", "shared:e"])

    rankings = rank_in_alias_universe(
        FakeIndex(),
        np.ones((1, 2), dtype=np.float32),
        aliases,
        {"shared:a", "shared:c", "shared:e"},
        top_k=2,
    )

    assert rankings == {0: ["shared:a", "shared:e"]}


def test_rank_in_alias_universe_rejects_top_k_larger_than_shared_set():
    with pytest.raises(ValueError, match="shared alias universe"):
        rank_in_alias_universe(
            FakeIndex(),
            np.ones((1, 2), dtype=np.float32),
            np.asarray(["a", "b", "c", "d", "e"]),
            {"a", "b"},
            top_k=3,
        )


def test_corpus_matches_snapshot_checks_all_corpus_yaml_hashes(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({
        "artifacts": {
            "dataset_builder/corpus/filesystem.yaml": "abc",
            "dataset_builder/corpus/networking.yaml": "def",
            "dataset_builder/embedder/vectors.npz": "ignored",
        },
    }), encoding="utf-8")

    assert corpus_matches_snapshot({
        "dataset_builder/corpus/filesystem.yaml": "abc",
        "dataset_builder/corpus/networking.yaml": "def",
    }, snapshot)
    assert not corpus_matches_snapshot({
        "dataset_builder/corpus/filesystem.yaml": "abc",
    }, snapshot)
