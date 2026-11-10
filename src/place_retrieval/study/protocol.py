"""Create a deterministic city-disjoint protocol using metadata, never model scores."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random

from ..io import digest, read_json, sha256, write_json


def city_of(row):
    return row.get("metadata", {}).get("city") or row["place_id"].split(":")[0]


def allocate(counts, seed=2027):
    cities = sorted(counts)
    if len(cities) < 9:
        raise ValueError("Prospective protocol requires at least nine cities")
    random.Random(seed).shuffle(cities)
    target = sum(counts.values()) * 0.15
    assignment = {}
    for split in ("test", "val"):
        available = [city for city in cities if city not in assignment]
        selected = min(itertools.combinations(available, 4),
                       key=lambda group: abs(sum(counts[city] for city in group) - target))
        assignment.update({city: split for city in selected})
    assignment.update({city: "train" for city in cities if city not in assignment})
    return dict(sorted(assignment.items()))


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    original = read_json(source / "manifest.json")
    valid_hash = sha256(source / "valid.jsonl")
    if valid_hash != original["files"]["valid.jsonl"]["sha256"]:
        raise ValueError("Original formal valid file changed")
    counts = Counter()
    with (source / "valid.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            counts[city_of(json.loads(line))] += 1
    assignment = allocate(counts)
    identity = {"version": "p1-city-disjoint-v1", "source_fingerprint": original["fingerprint"],
                "source_valid_sha256": valid_hash, "allocation_seed": 2027,
                "allocation_rule": "four held-out cities each; closest place count to 15%; metadata only",
                "city_assignment": assignment, "city_place_counts": dict(sorted(counts.items()))}
    fingerprint = digest(identity)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved["fingerprint"] != fingerprint or sha256(destination / "valid.jsonl") != saved["files"]["valid.jsonl"]["sha256"]:
            raise ValueError("An incompatible prospective protocol already exists")
        return saved
    splits = {s: Counter() for s in ("train", "val", "test")}
    place_ids = set()
    duplicate_keys = {"sha256": {}, "panoid": {}}
    cross_split_duplicates = []
    temporary = destination / "valid.jsonl.tmp"
    with (source / "valid.jsonl").open(encoding="utf-8") as incoming, temporary.open("w", encoding="utf-8", newline="\n") as output:
        for line in incoming:
            row = json.loads(line)
            if row["place_id"] in place_ids:
                raise ValueError("Duplicate place ID")
            place_ids.add(row["place_id"])
            row["split"] = assignment[city_of(row)]
            split = row["split"]
            source_paths, target_paths = set(row["source_image_paths"]), set(row["target_image_paths"])
            if source_paths & target_paths:
                raise ValueError("Source and target views overlap")
            for image in row["images"]:
                for key, registry in duplicate_keys.items():
                    value = image.get(key)
                    if not value:
                        continue
                    if value in registry and registry[value] != split:
                        cross_split_duplicates.append({"key": key, "value": value})
                    registry[value] = split
            splits[split].update(places=1, texts=len(row["texts"]),
                                 gallery_images=sum(image["path"] in target_paths for image in row["images"]))
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    if cross_split_duplicates:
        write_json(destination / "failed_duplicate_audit.json", cross_split_duplicates)
        raise ValueError("Cross-split image/panorama duplicates require resolution before training")
    temporary.replace(destination / "valid.jsonl")
    manifest = {"version": identity["version"], "fingerprint": fingerprint, "identity": identity,
                "quality_policy": original.get("quality_policy"), "protocol": "B",
                "splits": {s: dict(count) for s, count in splits.items()},
                "files": {"valid.jsonl": {"sha256": sha256(destination / "valid.jsonl"), "rows": len(place_ids)}},
                "audit": {"place_ids_unique": True, "source_target_disjoint": True,
                          "cross_split_sha256_duplicates": 0, "cross_split_panoid_duplicates": 0},
                "historical_files_modified": False, "initialization": "public pretrained only; no historical trained adapters"}
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))
