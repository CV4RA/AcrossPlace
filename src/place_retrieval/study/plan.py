"""Freeze the common protocol and unique three-server job assignments."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import yaml

from ..io import digest, load_config, read_json, write_json


def make_plan(root, base_config, expected_fingerprint, protocol="P0"):
    root = Path(root)
    manifest = read_json(root / f"protocol_{protocol.lower()}/formal/manifest.json")
    if manifest["fingerprint"] != expected_fingerprint:
        raise ValueError("Original P0 manifest identity differs")
    c = load_config(base_config)
    model_paths = {key: str(Path(value).resolve()) for key, value in c["models"].items()}
    model_root = Path(model_paths["siglip"]).parent
    c["data"].update(formal=str(root / f"protocol_{protocol.lower()}/formal"), image_root=str(Path(c["data"]["image_root"]).resolve()),
                     fingerprint=manifest["fingerprint"])
    c["models"] = model_paths
    c["visual_cache"] = str(root / "shared_visual")
    c["cpu_threads"] = 8
    c["training"].update(adapt_towers=["vision", "text"], residual_heads=True,
                         lora_targets="qv_last", use_rslora=False, lora_dropout=0.0, loss="place")
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    jobs = []

    def add(name, role, overrides, seeds=(42, 43, 44), action="train_validate"):
        for seed in seeds:
            cfg = deepcopy(c)
            cfg["seed"] = seed
            cfg["training"].update(overrides.get("training", {}))
            cfg["models"].update(overrides.get("models", {}))
            cfg["output"] = str(root / "runs" / f"{name}_s{seed}")
            cfg["study"] = {"method": name, "protocol": protocol, "action": action}
            job_id = f"{name}_s{seed}"
            path = config_root / f"{job_id}.yaml"
            content = yaml.safe_dump(cfg, sort_keys=False)
            if path.exists() and path.read_text() != content:
                raise ValueError("A different job configuration already exists")
            path.write_text(content)
            jobs.append({"id": job_id, "role": role, "action": action, "config": str(path),
                         "config_digest": digest(cfg), "output": cfg["output"]})

    # Start with a useful main reference and adaptation ablation on the two large GPUs.
    add("T4_siglip_place", "comparison", {})
    add("T3_siglip_ms", "auxiliary", {"training": {"loss": "ms", "ms_direction": "symmetric"}})
    add("T1_clip16_place", "auxiliary", {"models": {"siglip": str(model_root / "clip-vit-base-patch16")}})
    add("T2_lavpr_blip_ms", "comparison", {
        "models": {"siglip": model_paths["blip"]},
        "training": {"loss": "ms", "ms_direction": "i2t", "lora_targets": "all-linear",
                     "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.1, "use_rslora": True,
                     "residual_heads": False, "text_length": 64, "optimizer": "sgd",
                     "lora_lr": 0.05 * 16 / 60, "head_lr": 0.05 * 16 / 60, "weight_decay": 0.001}})
    add("A2_no_residual", "ablation", {"training": {"residual_heads": False}})
    add("A3_vision_only", "ablation", {"training": {"adapt_towers": ["vision"]}})
    add("A4_text_only", "ablation", {"training": {"adapt_towers": ["text"]}})
    add("A1_heads_only", "ablation", {"training": {"adapt_towers": []}})
    add("Z3_siglip", "auxiliary", {"training": {"adapt_towers": [], "residual_heads": False}}, seeds=(42,), action="zero_validate")
    add("Z4_blip_itc", "auxiliary", {
        "models": {"siglip": model_paths["blip"]},
        "training": {"adapt_towers": [], "residual_heads": False, "text_length": 64}}, seeds=(42,), action="zero_coarse")
    for patch in (16, 32):
        add(f"Z_clip{patch}", "auxiliary", {
            "models": {"siglip": str(model_root / f"clip-vit-base-patch{patch}")},
            "training": {"adapt_towers": [], "residual_heads": False}}, seeds=(42,), action="zero_coarse")
    result = {"version": "acrossplace-portable-study-v1", "protocol": protocol, "protocol_fingerprint": manifest["fingerprint"],
              "seeds": [42, 43, 44], "training_runs": 24, "jobs": jobs,
              "test_stage": "sealed; separate global-freeze phase", "historical_runs_reused": False,
              "comparison_budget": "common three-epoch budget; not a claim of converged native-schedule superiority",
              "native_baseline_adaptations": {"T2": "official BLIP ITC / all-linear rank64 rsLoRA / MS miner; P0 data, 16-place batch, scaled SGD LR and common cosine schedule"}}
    result["fingerprint"] = digest(result)
    path = root / "plan.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("The study plan was already frozen differently")
    write_json(path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--expected-fingerprint", required=True)
    p.add_argument("--protocol", choices=("P0", "P1"), default="P0")
    args = p.parse_args()
    result = make_plan(args.root, args.base_config, args.expected_fingerprint, args.protocol)
    print(json.dumps({"fingerprint": result["fingerprint"], "training_runs": result["training_runs"],
                      "jobs": len(result["jobs"])}, indent=2))
