import json
import math
from pathlib import Path

import pytest
import torch

from place_retrieval.data import image_path, records, training_batches, verify_formal
from place_retrieval.training import place_loss


def test_source_views_excluded_and_schedule_covers_places(tiny_config):
    c = tiny_config
    verify_formal(c["data"])
    rows = records(c["data"], "train")
    assert len(rows["image"]) == 4
    assert all(not r["path"].endswith("-0.jpg") for r in rows["image"])
    a = training_batches(rows, c["training"], 42)
    assert a == training_batches(rows, c["training"], 42)
    assert {rows["image"][i]["place_id"] for b in a for i in b["image"]} == {"train-0", "train-1"}
    assert all(image_path(r["path"], c["data"]).is_file() for r in rows["image"])
    bad = records(c["data"], "val")
    with pytest.raises(ValueError, match="Held-out"):
        training_batches(bad, c["training"], 42)


def test_annotation_mutation_and_path_escape_rejected(tiny_config):
    c = tiny_config["data"]
    path = Path(c["formal"]) / "valid.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_formal(c)
    for path in ("/other/foo.jpg", "/canonical/../foo.jpg"):
        with pytest.raises(ValueError):
            image_path(path, c)


def test_all_same_place_positives_have_nonzero_loss_floor():
    images = torch.randn(2, 5, requires_grad=True)
    texts = torch.randn(4, 5, requires_grad=True)
    ir = [{"place_id": "same"}] * 2
    tr = [{"place_id": "same"}] * 4
    # Identical logits have a known floor; multiple positives cannot all have probability 1.
    value = place_loss(images * 0, texts * 0, ir, tr, 0.07, 25)
    assert value.item() == pytest.approx(0.5 * (math.log(2) + math.log(4)))
    value.backward()
    assert torch.isfinite(images.grad).all()


def test_nearby_negatives_ignored_missing_gps_not_ignored():
    embeddings = torch.eye(2, requires_grad=True)
    close = [{"place_id": str(i), "gps": [0, 0]} for i in range(2)]
    assert place_loss(embeddings, embeddings, close, close, 1, 25).item() == pytest.approx(0)
    missing = [{"place_id": str(i)} for i in range(2)]
    assert place_loss(embeddings, embeddings, close, missing, 1, 25).item() > 0
    with pytest.raises(ValueError, match="counterpart"):
        place_loss(embeddings, embeddings, close, [{"place_id": "other"}] * 2, 1, 25)
