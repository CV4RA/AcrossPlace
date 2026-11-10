"""Tests for the scientific controls and metric edge cases."""
from copy import deepcopy
import numpy as np
import pytest
from benchmark import (paired_view_rows, nested_galleries, quality_pair,
                       retrieval_hits, metrics, paired_ci, immutable)


def place():
    return {"place_id": "city:1", "split": "test", "metadata": {"city": "city"},
            "images": [{"path": "s", "image_id": "s1"}, {"path": "t", "image_id": "t1"}],
            "source_image_paths": ["s"], "target_image_paths": ["t"],
            "texts": [{"text_id": "text1", "text": "brick house", "quality_status": "valid", "source_images": ["s"]}]}


def test_source_target_control_preserves_query_and_identity():
    original = place()
    before = deepcopy(original)
    selected, excluded = paired_view_rows([original])
    row = selected[0]
    assert not excluded
    assert row["source"]["path"] == row["text"]["source_images"][0]
    assert row["target"]["path"] not in row["text"]["source_images"]
    assert row["source"]["place_id"] == row["target"]["place_id"]
    assert original == before


def test_overlap_fails_before_inference():
    p = place()
    p["target_image_paths"] = ["s"]
    with pytest.raises(ValueError, match="overlap"):
        paired_view_rows([p])


def test_multiple_source_caption_is_not_exact_same_view_control():
    p = place()
    p["texts"][0]["source_images"] = ["s", "other"]
    selected, excluded = paired_view_rows([p])
    assert selected == [] and excluded == ["city:1"]


def test_nested_galleries_fix_queries_and_naturally_include_positives():
    ids = list("abcdefghijkl")
    pools = nested_galleries(ids, ["a", "b"], 42, [2, 5, 12])
    assert len(pools["5"]) == 5
    assert pools["12"][:5] == pools["5"]
    assert pools["5"][:2] == ["a", "b"]
    assert pools == nested_galleries(list(reversed(ids)), ["a", "b"], 42, [2, 5, 12])


def test_quality_matching_controls_type_source_and_length():
    base = {"text_id": "a", "text": "brick house", "quality_status": "valid", "type": "concise", "source_images": ["s"]}
    wrong_type = dict(base, text_id="b", quality_status="uncertain", type="relational")
    wrong_source = dict(base, text_id="c", quality_status="uncertain", source_images=["other"])
    right = dict(base, text_id="d", quality_status="uncertain", text="a brick house")
    assert quality_pair("p", [base, wrong_type, wrong_source], strict=True) is None
    a, b = quality_pair("p", [base, wrong_type, wrong_source, right], strict=True)
    assert a["text_id"] == "a" and b["text_id"] == "d"


def test_exact_metrics_multiple_positives_and_small_galleries():
    q = np.eye(2)
    g = np.array([[1, 0], [1, 0], [0, 1]])
    hits = retrieval_hits(q, g, ["a", "b"], ["a", "a", "b"])
    assert metrics(hits) == {"Recall@1": 1., "Recall@5": 1., "Recall@10": 1.}


def test_ties_have_reproducible_gallery_order():
    hits = retrieval_hits([[1, 0]], [[1, 0], [1, 0]], ["positive"], ["negative", "positive"])
    assert hits == [[False, True, True]]


def test_missing_positive_and_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="positive"):
        retrieval_hits([[1, 0]], [[1, 0]], ["a"], ["b"])
    with pytest.raises(ValueError, match="finite"):
        retrieval_hits([[float("nan"), 0]], [[1, 0]], ["a"], ["a"])


def test_paired_bootstrap_direction():
    a = [[False, True, True], [False, True, True]]
    b = [[True, True, True], [True, True, True]]
    result = paired_ci(a, b, repetitions=100)
    assert result["Recall@1"]["b_minus_a_pp"] == 100
    assert result["Recall@1"]["ci95_pp"] == [100, 100]


def test_freeze_rejects_changed_identity(tmp_path):
    p = tmp_path / "freeze.json"
    immutable(p, {"a": 1})
    immutable(p, {"a": 1})
    with pytest.raises(ValueError, match="changed"):
        immutable(p, {"a": 2})
