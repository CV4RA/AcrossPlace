"""One declared training/validation or pretrained-only validation job."""
import argparse
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from ..data import records, verify_formal
from ..encoding import encode_split
from ..evaluation import validate
from ..io import digest, load_config, log, model_identity, save_torch, seed_all, write_json
from ..metrics import coverage
from ..models import adapter_state, load_siglip
from ..training import train


def ensure_model(c):
    path = Path(c["models"]["siglip"])
    if (path / "model.safetensors").exists():
        return
    kind = {"clip-vit-base-patch16": "clip16", "clip-vit-base-patch32": "clip32"}.get(path.name)
    if not kind:
        raise FileNotFoundError(f"Missing pretrained model: {path}")
    script = Path(__file__).resolve().parents[3] / "scripts/download_models.py"
    subprocess.run([sys.executable, str(script), "--output", str(path.parent), "--model", kind], check=True)


def run(c, action):
    torch.set_num_threads(c.get("cpu_threads", 8))
    ensure_model(c)
    if action == "train_validate":
        checkpoint = train(c, resume=(Path(c["output"]) / "coarse/last.pt").is_file())
        selection = validate(c, checkpoint)
        return {"checkpoint": str(checkpoint), "validation_selection": str(selection)}
    manifest = verify_formal(c["data"])
    seed_all(c["seed"])
    device = torch.device(c["device"])
    backbone, processor, heads = load_siglip(c["models"]["siglip"], c["training"], device)
    root = Path(c["output"])
    checkpoint = root / "coarse/best.pt"
    state = {"format": "siglip2-place-lora-v1", "epoch": 0, "training": c["training"],
             "identity": {"formal_fingerprint": manifest["fingerprint"],
                          "config": digest(c), "siglip": model_identity(c["models"]["siglip"])},
             "heads": heads.state_dict(), "lora": adapter_state(backbone), "pretrained_only": True}
    save_torch(checkpoint, state)
    rows = records(c["data"], "val")
    features = encode_split(backbone, processor, heads, rows, c, device)
    coarse = coverage(features, rows)
    save_torch(root / "coarse/val_features.pt", {"features": features, "records": rows, "identity": state["identity"]})
    write_json(root / "coarse/val_result.json", {"pretrained_only": True, "split": "val", "metrics": coarse})
    del backbone, processor, heads
    torch.cuda.empty_cache()
    if action == "zero_validate":
        selection = validate(c, checkpoint)
        return {"pretrained_only": True, "coarse_validation": coarse, "validation_selection": str(selection)}
    return {"pretrained_only": True, "coarse_validation": coarse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", required=True, choices=("train_validate", "zero_validate", "zero_coarse"))
    args = parser.parse_args()
    c = load_config(args.config)
    result = run(c, args.action)
    write_json(Path(c["output"]) / "job_complete.json", {"status": "complete", "config_digest": digest(c),
               "action": args.action, "test_executed": False, "result": result})
    log(result)
