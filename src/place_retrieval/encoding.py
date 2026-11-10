"""Coarse embeddings and the fixed pretrained BLIP cross-encoder."""
import time
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

from .data import ImageDataset
from .io import digest, log, model_identity, read_json, run_lock, sha256, write_json
from .models import context_length, load_blip, raw_features


@torch.inference_mode()
def encode_split(backbone, processor, heads, rows, c, device):
    backbone.eval()
    heads.eval()
    options = c["evaluation"]
    loader = DataLoader(ImageDataset(rows["image"], processor, c["data"]),
                        batch_size=options["encode_batch_size"], num_workers=options["workers"],
                        **({"multiprocessing_context": "spawn"} if options["workers"] else {}))
    images = []
    for pixels in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = raw_features(backbone, pixels.to(device), "image").float()
        images.append(raw.cpu())
    texts = []
    size = options["encode_batch_size"]
    length = context_length(backbone, c["training"])
    for start in range(0, len(rows["text"]), size):
        values = [r["text"] for r in rows["text"][start:start + size]]
        tokens = processor(text=values, padding="max_length", max_length=length,
                           truncation=True, return_tensors="pt").to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = raw_features(backbone, tokens, "text").float()
        texts.append(raw.cpu())
    # Match the archived inference path: project FP32 features in 2,048-sample chunks.
    features = {}
    for modality, chunks in (("image", images), ("text", texts)):
        raw = torch.cat(chunks)
        features[modality] = torch.cat([
            heads.encode(raw[start:start + 2048].to(device), modality).cpu()
            for start in range(0, len(raw), 2048)
        ])
    return features


@torch.inference_mode()
def score_blip(candidates, c, device):
    backbone, processor = load_blip(c["models"]["blip"], device)
    rows = candidates["records"]
    options = c["evaluation"]
    loader = DataLoader(ImageDataset(rows["image"], processor, c["data"]),
                        batch_size=options["visual_batch_size"], num_workers=options["workers"],
                        **({"multiprocessing_context": "spawn"} if options["workers"] else {}))
    visual = None
    bank_root = Path(c.get("visual_cache", str(Path(c["output"]) / "visual_cache")))
    identity = {"model": model_identity(c["models"]["blip"]),
                "images": digest([(r["image_id"], r.get("sha256")) for r in rows["image"]]),
                "device_compute": "bf16" if device.type == "cuda" else "float32"}
    bank_root = bank_root / digest(identity)
    with run_lock(bank_root):
        done = bank_root / "manifest.json"
        path = bank_root / "visual.npy"
        if done.exists():
            metadata = read_json(done)
            if metadata["identity"] != identity or sha256(path) != metadata["sha256"]:
                raise ValueError("Visual cache identity or hash changed")
        else:
            offset = 0
            for pixels in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    value = backbone.vision_model(pixel_values=pixels.to(device), return_dict=True).last_hidden_state
                if visual is None:
                    visual = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                                       shape=(len(rows["image"]), *value.shape[1:]))
                visual[offset:offset + len(value)] = value.half().cpu().numpy()
                offset += len(value)
            visual.flush()
            del visual
            write_json(done, {"identity": identity, "sha256": sha256(path)})
        visual = np.load(path, mmap_mode="r")
    results = []
    size = options["pair_batch_size"]
    started = time.monotonic()
    pairs = candidates["pairs"]
    for start in range(0, len(pairs), size):
        selected = pairs[start:start + size]
        texts = [rows["text"][i]["text"] for i in selected[:, 0].tolist()]
        tokens = processor.tokenizer(texts, padding="max_length", max_length=options["text_length"],
                                     truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
        images = torch.from_numpy(np.array(visual[selected[:, 1].numpy()], copy=True)).to(device).float()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            attention = torch.ones(images.shape[:2], dtype=torch.long, device=device)
            hidden = backbone.text_encoder(**tokens, encoder_hidden_states=images,
                                           encoder_attention_mask=attention,
                                           return_dict=True).last_hidden_state[:, 0]
            logits = backbone.itm_head(hidden).float()
        results.append((logits[:, 1] - logits[:, 0]).cpu())
        if start % (size * 250) == 0:
            log({"stage": "rerank", "completed": min(start + size, len(pairs)),
                 "total": len(pairs), "seconds": time.monotonic() - started})
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(results)
