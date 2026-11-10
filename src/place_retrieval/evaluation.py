"""Freeze selection on validation and apply it unchanged to the test split."""
import os
from pathlib import Path

import torch

from .data import records, verify_formal
from .encoding import encode_split, score_blip
from .io import digest, log, model_identity, read_json, run_lock, save_torch, sha256, write_json
from .metrics import evaluate, select_fusion, shortlist
from .models import load_siglip, restore_coarse


def checkpoint_state(path, c, manifest, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"]["formal_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("The coarse checkpoint belongs to another formal dataset")
    if state.get("format") == "siglip2-place-lora-v1":
        if state["identity"]["siglip"] != model_identity(c["models"]["siglip"]):
            raise ValueError("Pretrained SigLIP2 files differ from the training run")
        keys = ("lora_rank", "lora_alpha", "lora_last_layers", "adapt_towers", "residual_heads", "lora_targets", "use_rslora", "lora_dropout")
        if any(state["training"].get(key) != c["training"].get(key) for key in keys):
            raise ValueError("LoRA architecture differs from the checkpoint")
    else:
        # Support the archived heads_lora_epoch3.pt used for the reported result.
        lora = state.get("lora_config", {})
        if lora.get("r") != c["training"]["lora_rank"] or lora.get("lora_alpha") != c["training"]["lora_alpha"]:
            raise ValueError("Legacy checkpoint has a different LoRA rank or scale")
    backbone, processor, heads = load_siglip(c["models"]["siglip"], c["training"], device)
    restore_coarse(backbone, heads, state)
    return backbone, processor, heads, state["epoch"]


def score_split(c, checkpoint, split, manifest):
    device = torch.device(c["device"])
    output = Path(c["output"]) / "evaluation" / split
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "formal_fingerprint": manifest["fingerprint"], "split": split,
        "coarse_sha256": sha256(checkpoint),
        "siglip": model_identity(c["models"]["siglip"]),
        "blip": model_identity(c["models"]["blip"]),
        "evaluation": {k: v for k, v in c["evaluation"].items() if k != "fusion_weights"},
    }
    identity["fingerprint"] = digest(identity)
    rows = records(c["data"], split)
    expected = manifest["splits"][split]
    if len(rows["image"]) != expected["gallery_images"] or len(rows["text"]) != expected["texts"]:
        raise ValueError("Split records disagree with the formal manifest counts")
    cache = output / f"{identity['fingerprint']}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Evaluation cache identity mismatch")
        return saved
    backbone, processor, heads, epoch = checkpoint_state(checkpoint, c, manifest, device)
    features = encode_split(backbone, processor, heads, rows, c, device)
    del backbone, processor, heads
    if device.type == "cuda":
        torch.cuda.empty_cache()
    candidates = shortlist(features, rows, c["evaluation"]["candidates"])
    scores = score_blip(candidates, c, device)
    saved = {"identity": identity, "coarse_epoch": epoch, "candidates": candidates, "scores": scores}
    save_torch(cache, saved)
    return saved


def validate(c, checkpoint):
    output = Path(c["output"]) / "evaluation"
    with run_lock(output):
        manifest = verify_formal(c["data"])
        saved = score_split(c, checkpoint, "val", manifest)
        result, grid = select_fusion(saved["candidates"], saved["scores"],
                                     c["evaluation"]["fusion_weights"], "val")
        selection_path = output / "selection.json"
        selection = {
            "format": "siglip2-blip-selection-v1", "selected_on": "val",
            "coarse_checkpoint": os.path.relpath(Path(checkpoint).resolve(), output.resolve()),
            "coarse_epoch": saved["coarse_epoch"],
            "coarse_sha256": saved["identity"]["coarse_sha256"],
            "siglip": saved["identity"]["siglip"], "blip": saved["identity"]["blip"],
            "formal_fingerprint": manifest["fingerprint"],
            "evaluation": saved["identity"]["evaluation"],
            "global_weight": result["global_weight"], "evidence_weight": 0,
        }
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ValueError("A different selection already exists. Use a new output directory.")
        write_json(output / "val_result.json", {
            "split": "val", "selection": selection, "selected_model": result, "fusion_grid": grid,
            "coarse_only": evaluate(saved["candidates"], saved["candidates"]["coarse_scores"], 0),
        })
        write_json(selection_path, selection)
        log({"selection": str(selection_path), "validation": result})
        return selection_path


def test(c, selection_path):
    output = Path(c["output"]) / "evaluation"
    with run_lock(output):
        selection = read_json(selection_path)
        if selection.get("format") != "siglip2-blip-selection-v1" or selection.get("selected_on") != "val":
            raise ValueError("Test evaluation requires a validation-generated selection")
        manifest = verify_formal(c["data"])
        checkpoint = (Path(selection_path).resolve().parent / selection["coarse_checkpoint"]).resolve()
        expected_eval = {k: v for k, v in c["evaluation"].items() if k != "fusion_weights"}
        checks = {
            "formal_fingerprint": manifest["fingerprint"], "coarse_sha256": sha256(checkpoint),
            "siglip": model_identity(c["models"]["siglip"]), "blip": model_identity(c["models"]["blip"]),
            "evaluation": expected_eval, "evidence_weight": 0,
        }
        if any(selection.get(key) != value for key, value in checks.items()):
            raise ValueError("Test data, models, or inference settings differ from the frozen selection")
        result_path = output / "test_result.json"
        if result_path.exists():
            previous = read_json(result_path)
            if previous["selection"] != selection:
                raise ValueError("A completed test used another selection")
            log(previous)
            return previous
        write_json(output / "test_selection.json", selection)
        saved = score_split(c, checkpoint, "test", manifest)
        result = {
            "split": "test", "selection": selection, "hyperparameter_search_on_test": False,
            "selected_model": evaluate(saved["candidates"], saved["scores"], selection["global_weight"]),
            "coarse_only": evaluate(saved["candidates"], saved["candidates"]["coarse_scores"], 0),
        }
        write_json(result_path, result)
        log(result)
        return result
