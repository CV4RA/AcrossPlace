from copy import deepcopy
from pathlib import Path

import pytest
import torch

from place_retrieval.data import records
from place_retrieval.evaluation import test as evaluate_test, validate
from place_retrieval.io import read_json
from place_retrieval.models import load_siglip, raw_features, restore_coarse
from place_retrieval.training import place_loss, train


def test_real_siglip_lora_updates_both_encoders_and_restores(tiny_config):
    c = tiny_config
    model, processor, heads = load_siglip(c["models"]["siglip"], c["training"], "cpu")
    model.train()
    pixels = torch.randn(2, 3, 16, 16)
    tokens = processor(text=["red building", "blue doorway"])
    images = heads.encode(raw_features(model, pixels, "image"), "image")
    texts = heads.encode(raw_features(model, tokens, "text"), "text")
    rows = [{"place_id": str(i)} for i in range(2)]
    loss = place_loss(images, texts, rows, rows, 0.07, 25)
    loss.backward()
    for tower in ("vision_model", "text_model"):
        gradients = [p.grad for n, p in model.named_parameters() if tower in n and "lora_B" in n]
        assert gradients and any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)


def test_full_train_validation_test_and_frozen_selection(tiny_config):
    c = tiny_config
    checkpoint = train(c)
    assert checkpoint.is_file()
    selection_path = validate(c, checkpoint)
    first = evaluate_test(c, selection_path)
    second = evaluate_test(c, selection_path)
    assert first == second
    assert first["hyperparameter_search_on_test"] is False
    assert first["selection"]["selected_on"] == "val"
    assert first["selection"]["evidence_weight"] == 0
    for direction in ("T2I", "I2T"):
        assert set(first["selected_model"][direction]) == {"Recall@1", "Recall@5", "Recall@10"}
        assert all(0 <= v <= 1 for v in first["selected_model"][direction].values())
    changed = deepcopy(c)
    changed["evaluation"]["candidates"] += 1
    with pytest.raises(ValueError, match="frozen selection"):
        evaluate_test(changed, selection_path)
    # Reusing an output without explicitly requesting resume is refused.
    with pytest.raises(ValueError, match="resume"):
        train(c)


def test_epoch_boundary_resume_matches_uninterrupted_training(tiny_config, monkeypatch):
    import place_retrieval.training as training

    c = tiny_config
    train(c)
    reference = torch.load(Path(c["output"]) / "coarse/last.pt", weights_only=True)
    resumed = deepcopy(c)
    resumed["output"] = str(Path(c["output"]).parent / "resumed")
    original_encode = training.encode_split
    calls = 0

    def interrupt_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption in epoch two")
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(training, "encode_split", interrupt_second_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(resumed)
    last = Path(resumed["output"]) / "coarse/last.pt"
    assert torch.load(last, weights_only=True)["epoch"] == 1
    monkeypatch.setattr(training, "encode_split", original_encode)
    train(resumed, resume=True)
    restored = torch.load(last, weights_only=True)
    assert restored["epoch"] == reference["epoch"] == 2
    for component in ("lora", "heads"):
        for key in reference[component]:
            torch.testing.assert_close(reference[component][key], restored[component][key], rtol=0, atol=0)
    assert [h["loss"] for h in restored["history"]] == [h["loss"] for h in reference["history"]]
    assert read_json(last.parent / "completion.json")["status"] == "complete"
