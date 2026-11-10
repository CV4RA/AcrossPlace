"""Read immutable formal splits and relocate image paths at access time."""
import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from torch.utils.data import Dataset

from .io import read_json, sha256


def verify_formal(c):
    root = Path(c["formal"])
    manifest = read_json(root / "manifest.json")
    if not c.get("fingerprint") or manifest["fingerprint"] != c["fingerprint"]:
        raise ValueError("Formal dataset fingerprint differs from the configuration")
    for name, metadata in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Formal dataset hash mismatch: {name}")
    return manifest


def records(c, split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split}")
    result = {"image": [], "text": []}
    with (Path(c["formal"]) / "valid.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            place = json.loads(line)
            if place["split"] != split:
                continue
            fields = {"place_id": place["place_id"], "split": split}
            targets = set(place["target_image_paths"])
            for image in place["images"]:
                if image["path"] in targets:
                    result["image"].append(dict(image, **fields, is_gallery=True))
            result["text"].extend(dict(text, **fields) for text in place["texts"])
    if any(not rows for rows in result.values()):
        raise ValueError(f"Empty modality in {split}")
    if any(r.get("quality_status", "valid") != "valid" for r in result["text"]):
        raise ValueError("The main experiment requires valid texts only")
    return result


def image_path(value, c):
    prefix = PurePosixPath(c["canonical_prefix"])
    try:
        relative = PurePosixPath(value).relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"Image path is outside canonical_prefix: {value}") from error
    if ".." in relative.parts:
        raise ValueError(f"Invalid image path: {value}")
    return Path(c["image_root"]).joinpath(*relative.parts)


class ImageDataset(Dataset):
    def __init__(self, rows, processor, data):
        self.rows = rows
        self.processor = processor
        self.data = data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(image_path(self.rows[index]["path"], self.data)) as image:
            return self.processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"][0]


def training_batches(rows, c, seed):
    groups = {name: defaultdict(list) for name in rows}
    for name, values in rows.items():
        for index, row in enumerate(values):
            if row["split"] != "train":
                raise ValueError("Held-out records entered training")
            groups[name][row["place_id"]].append(index)
    places = sorted(groups["image"])
    if set(places) != set(groups["text"]):
        raise ValueError("A training place is missing an image or text")
    rng = random.Random(seed)
    rng.shuffle(places)
    batches = []
    for start in range(0, len(places), c["places_per_batch"]):
        batch = places[start:start + c["places_per_batch"]]
        batches.append({
            name: [i for place in batch for i in rng.sample(groups[name][place], min(count, len(groups[name][place])))]
            for name, count in (("image", c["images_per_place"]), ("text", c["texts_per_place"]))
        })
    return batches
