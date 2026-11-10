"""Offline fixtures using actual tiny Transformers backbones and synthetic places."""
import hashlib
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import BlipConfig, BlipForImageTextRetrieval, SiglipConfig, SiglipModel
from transformers.feature_extraction_utils import BatchFeature

from place_retrieval.io import digest, sha256, write_json


class TinyProcessor:
    @property
    def tokenizer(self):
        return self

    def __call__(self, text=None, images=None, max_length=8, **kwargs):
        if images is not None:
            pixels = [torch.from_numpy(np.array(im.resize((16, 16)), copy=True)).permute(2, 0, 1).float() / 255
                      for im in images]
            return BatchFeature({"pixel_values": torch.stack(pixels)})
        if text is None:
            # BLIP calls its tokenizer with positional text arguments.
            raise ValueError("Text or images are required")
        ids = []
        for value in text:
            number = hashlib.sha256(value.encode()).digest()[0] % 20 + 4
            ids.append([2, number, 3] + [0] * (max_length - 3))
        tokens = torch.tensor(ids)
        return BatchFeature({"input_ids": tokens, "attention_mask": tokens.ne(0).long()})


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    torch.manual_seed(7)
    siglip = tmp_path / "siglip"
    sc = SiglipConfig(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=16, patch_size=8),
    )
    SiglipModel(sc).save_pretrained(siglip, safe_serialization=True)
    blip = tmp_path / "blip"
    bc = BlipConfig(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
                         encoder_hidden_size=16, bos_token_id=2, eos_token_id=3, sep_token_id=3),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8),
        image_text_hidden_size=16,
    )
    BlipForImageTextRetrieval(bc).save_pretrained(blip, safe_serialization=True)
    monkeypatch.setattr("place_retrieval.models.AutoProcessor.from_pretrained", lambda *a, **kw: TinyProcessor())
    monkeypatch.setattr("place_retrieval.models.BlipProcessor.from_pretrained", lambda *a, **kw: TinyProcessor())
    formal = tmp_path / "formal"
    formal.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        for i in range(2):
            place = f"{split}-{i}"
            images = []
            for j in range(3):
                name = f"{place}-{j}.jpg"
                Image.new("RGB", (16, 16), (30 + 90 * i, 20 + 50 * j, 180)).save(image_root / name)
                images.append({"image_id": name, "path": "/canonical/" + name,
                               "sha256": sha256(image_root / name)})
            rows.append({"place_id": place, "split": split, "images": images,
                         "source_image_paths": [images[0]["path"]],
                         "target_image_paths": [v["path"] for v in images[1:]],
                         "texts": [{"text_id": f"{place}-{j}", "text": f"place {place}, view {j}",
                                    "quality_status": "valid"} for j in range(2)]})
    valid = formal / "valid.jsonl"
    valid.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    fingerprint = digest(rows)
    write_json(formal / "manifest.json", {
        "fingerprint": fingerprint, "files": {"valid.jsonl": {"sha256": sha256(valid)}},
        "splits": {s: {"places": 2, "texts": 4, "gallery_images": 4} for s in ("train", "val", "test")},
    })
    return {
        "device": "cpu", "seed": 42, "output": str(tmp_path / "run"),
        "data": {"formal": str(formal), "image_root": str(image_root),
                 "canonical_prefix": "/canonical", "fingerprint": fingerprint},
        "models": {"siglip": str(siglip), "blip": str(blip)},
        "training": {"epochs": 2, "places_per_batch": 2, "images_per_place": 2,
                     "texts_per_place": 2, "head_lr": 0.001, "lora_lr": 0.001,
                     "weight_decay": 0.01, "temperature": 0.07, "tolerance_m": 25,
                     "warmup_ratio": 0.1, "max_grad_norm": 1.0,
                     "lora_rank": 2, "lora_alpha": 4, "lora_last_layers": 1, "workers": 0},
        "evaluation": {"encode_batch_size": 4, "visual_batch_size": 4, "pair_batch_size": 4,
                       "workers": 0, "text_length": 8, "candidates": 10, "fusion_weights": [0, 2]},
    }
