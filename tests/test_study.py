from copy import deepcopy
from pathlib import Path

import pytest
import torch

from place_retrieval.models import adapter_state, load_siglip, raw_features, restore_coarse
from place_retrieval.metrics import exact_topk
from place_retrieval.study.protocol import allocate
from place_retrieval.training import objective, train


@pytest.mark.parametrize("towers", [[], ["vision"], ["text"], ["vision", "text"]])
def test_ablation_towers_are_exact(tiny_config, towers):
    c = tiny_config
    c["training"]["adapt_towers"] = towers
    model, processor, heads = load_siglip(c["models"]["siglip"], c["training"], "cpu")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    for tower in ("vision", "text"):
        assert any(name.startswith(tower + "_model") for name in names) == (tower in towers)
    assert all("lora_" in name for name in names)
    assert bool(adapter_state(model)) == bool(towers)


@pytest.mark.parametrize("heads,towers", [(False, ["vision", "text"]), (True, [])])
def test_variant_training_and_checkpoint_restore(tiny_config, heads, towers):
    c = tiny_config
    c["training"].update(residual_heads=heads, adapt_towers=towers, epochs=1)
    path = train(c)
    state = torch.load(path, weights_only=True)
    assert bool(state["heads"]) == heads
    assert bool(state["lora"]) == bool(towers)
    model, processor, projection = load_siglip(c["models"]["siglip"], c["training"], "cpu")
    restore_coarse(model, projection, state)


def test_blocked_retrieval_matches_dense_without_ties():
    torch.manual_seed(19)
    query, gallery = torch.randn(19, 13), torch.randn(23, 13)
    expected = (query @ gallery.T).topk(10, dim=1).indices
    assert torch.equal(exact_topk(query, gallery, 10, block=3), expected)


def test_metadata_allocation_reproducible_and_city_disjoint():
    counts = {f"city{i}": 100 + i * 13 for i in range(15)}
    assignment = allocate(counts)
    assert assignment == allocate(dict(reversed(list(counts.items()))))
    assert sum(v == "val" for v in assignment.values()) == 4
    assert sum(v == "test" for v in assignment.values()) == 4
    assert sum(v == "train" for v in assignment.values()) == 7


def test_blip_itc_native_forward_and_lora(tiny_config):
    c = tiny_config
    t = dict(c["training"], lora_targets="all-linear", residual_heads=False, text_length=8)
    model, processor, heads = load_siglip(c["models"]["blip"], t, "cpu")
    model.train()
    images = heads.encode(raw_features(model, torch.randn(2, 3, 16, 16), "image"), "image")
    texts = heads.encode(raw_features(model, processor(text=["a", "b"]), "text"), "text")
    assert images.shape == texts.shape == (2, 16)
    (images @ texts.T).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if "lora_B" in name)


def test_ms_native_library_loss_is_finite_with_both_tower_gradients():
    pytest.importorskip("pytorch_metric_learning")
    images = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    texts = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1).requires_grad_()
    ir = [{"place_id": str(i // 2)} for i in range(4)]
    tr = [{"place_id": str(i // 3)} for i in range(6)]
    loss = objective(images, texts, ir, tr, {"loss": "ms"})
    assert torch.isfinite(loss)
    loss.backward()
    assert images.grad is not None and texts.grad is not None
