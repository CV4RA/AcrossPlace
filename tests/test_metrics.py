import pytest
import torch

from place_retrieval.metrics import evaluate, recall, select_fusion, shortlist


def test_metrics_rank_samples_without_place_deduplication():
    gallery = [{"place_id": "wrong"}] * 5 + [{"place_id": "correct"}]
    result = recall(torch.tensor([[0, 1, 2, 3, 4, 5]]), [{"place_id": "correct"}], gallery)
    assert result == {"Recall@1": 0, "Recall@5": 0, "Recall@10": 1}


def test_shortlist_has_no_forced_positives_and_exact_pair_offsets():
    features = {"image": torch.eye(12), "text": torch.eye(12)}
    # Positives are deliberately at the worst-ranked index for a selected query.
    rows = {m: [{"place_id": f"p-{i}"} for i in range(12)] for m in features}
    features["text"][0] = torch.arange(12).float()
    candidates = shortlist(features, rows, 10)
    assert 0 not in candidates["t2i"][0].tolist()
    pair_scores = features["text"] @ features["image"].T
    for key, matrix in (("t2i", pair_scores), ("i2t", pair_scores.T)):
        expected = matrix.gather(1, candidates[key])
        actual = candidates["coarse_scores"][candidates[key + "_offsets"]]
        torch.testing.assert_close(actual, expected)
    assert len(candidates["pairs"].unique(dim=0)) == len(candidates["pairs"])
    result = evaluate(candidates, candidates["coarse_scores"], 0)
    assert result["T2I"]["Recall@10"] < 1
    with pytest.raises(ValueError, match="validation"):
        select_fusion(candidates, candidates["coarse_scores"], [0, 2], "test")
    with pytest.raises(ValueError, match="nonempty"):
        select_fusion(candidates, candidates["coarse_scores"], [], "val")


def test_invalid_pair_scores_rejected():
    features = {"image": torch.eye(2), "text": torch.eye(2)}
    rows = {m: [{"place_id": str(i)} for i in range(2)] for m in features}
    candidates = shortlist(features, rows, 10)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(candidates, torch.tensor([float("nan")]), 50)
