"""Frozen benchmark diagnostics, independent of the main training pipelines.

No training or hyperparameter search is performed by this program.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

VERSION = "acrossviews-benchmark-diagnostics-v1"
P0_FP = "0aa126e367f28be0b4f63356081c55dc9c3525c99cd70ba81a5689dd9aa06d4a"
P1_FP = "fa9e4fe583d0e364883aa4a9629caf287e31d50e4f6a52babbbe476d7e430378"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f"Frozen identity changed: {path}")
    write(path, value)


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def ordered(values, seed, key=lambda x: x):
    return sorted(values, key=lambda x: (digest([seed, key(x)]), key(x)))


def paired_view_rows(places, seed=2027):
    """One provenance-bound caption and one source/target image per place."""
    selected, excluded = [], []
    for p in sorted(places, key=lambda x: x["place_id"]):
        images = {r["path"]: r for r in p["images"]}
        sources, targets = set(p["source_image_paths"]), set(p["target_image_paths"])
        if sources & targets:
            raise ValueError("Source and target views overlap")
        eligible = [t for t in p["texts"] if t.get("quality_status", "valid") == "valid"
                    and len(t.get("source_images", [])) == 1
                    and t["source_images"][0] in sources & images.keys()]
        if not eligible or not targets:
            excluded.append(p["place_id"])
            continue
        text = deepcopy(ordered(eligible, seed, lambda r: r["text_id"])[0])
        target = ordered(list(targets), seed)[0]
        fields = {"place_id": p["place_id"], "split": p["split"]}
        selected.append({"place_id": p["place_id"], "city": p["metadata"]["city"],
                         "text": dict(text, **fields),
                         "source": dict(images[text["source_images"][0]], **fields),
                         "target": dict(images[target], **fields)})
    return selected, excluded


def length_bin(text):
    size = len(text.split())
    return next((i for i, edge in enumerate((10, 25, 50, 100)) if size <= edge), 4)


def quality_pair(place, all_texts, strict):
    good = [t for t in all_texts if t.get("quality_status") == "valid"]
    uncertain = [t for t in all_texts if t.get("quality_status") == "uncertain"]
    candidates = []
    for a in good:
        for b in uncertain:
            if strict and (a.get("type") != b.get("type")
                           or sorted(a.get("source_images", [])) != sorted(b.get("source_images", []))
                           or length_bin(a["text"]) != length_bin(b["text"])):
                continue
            candidates.append((abs(len(a["text"].split()) - len(b["text"].split())),
                               digest([place, a["text_id"], b["text_id"]]), a, b))
    if not candidates:
        return None
    _, _, a, b = min(candidates, key=lambda x: x[:2])
    return a, b


def nested_galleries(ids, anchors, seed, sizes):
    """Candidate pools are fixed before scoring, with naturally present positives."""
    if not set(anchors) <= set(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid anchor/candidate identities")
    rest = ordered([p for p in ids if p not in set(anchors)], seed)
    values = list(anchors) + rest
    result = {}
    for size in sorted(set(sizes)):
        if size < len(anchors) or size > len(ids):
            raise ValueError("Candidate pool size outside valid bounds")
        result[str(size)] = values[:size]
    return result


def quality_rows(pairs, uncertain):
    result = {"unadjusted": [], "matched_type_source_length": []}
    for p in pairs:
        for label, strict in (("unadjusted", False), ("matched_type_source_length", True)):
            match = quality_pair(p["place_id"], uncertain.get(p["place_id"], []), strict)
            if match:
                a, b = match
                fields = {"place_id": p["place_id"], "split": p["text"]["split"]}
                result[label].append({"place_id": p["place_id"], "city": p["city"],
                                      "image": p["target"], "valid": dict(a, **fields),
                                      "uncertain": dict(b, **fields)})
    return result


def feature_records(spec):
    images, texts = {}, {}
    for p in spec["pairs"]:
        for name in ("source", "target"):
            images[p[name]["image_id"]] = p[name]
        texts[p["text"]["text_id"]] = p["text"]
    for values in spec["quality"].values():
        for p in values:
            for name in ("valid", "uncertain"):
                row = p[name]
                if row["text_id"] in texts and texts[row["text_id"]]["text"] != row["text"]:
                    raise ValueError("A text ID has conflicting content")
                texts[row["text_id"]] = row
    return {"image": [images[k] for k in sorted(images)], "text": [texts[k] for k in sorted(texts)]}


def dataset_spec(places, uncertain, gallery_seeds):
    pairs, excluded = paired_view_rows(places)
    ids = [p["place_id"] for p in pairs]
    if len(ids) < 1000:
        raise ValueError("The fixed 1,000-place comparison is not supported")
    matched = {str(s): ordered(ids, s)[:1000] for s in gallery_seeds}
    anchors = ordered(ids, 2027001)[:500]
    sizes = [500, 1000] + [n for n in (2000, 4000) if n < len(ids)] + [len(ids)]
    return {"pairs": pairs, "excluded_provenance": excluded,
            "quality": quality_rows(pairs, uncertain), "matched_1000": matched,
            "scale_anchors": anchors,
            "nested_galleries": {str(s): nested_galleries(ids, anchors, s, sizes) for s in gallery_seeds}}


def verify_inputs(plan):
    report = {}
    for path, expected in plan["input_files"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input file changed: {path}")
    for name, expected in read(Path(plan["root"]) / "code/release_files.json").items():
        if sha(Path(plan["root"]) / "code" / name) != expected:
            raise ValueError(f"Reference encoder source changed: {name}")
    for protocol, item in plan["protocols"].items():
        root = Path(item["formal"])
        manifest = read(root / "manifest.json")
        if manifest["fingerprint"] != item["fingerprint"]:
            raise ValueError(f"Unexpected formal fingerprint: {protocol}")
        for name, entry in manifest["files"].items():
            file = (root / name).resolve()
            if not file.is_relative_to(root.resolve()) or sha(file) != entry["sha256"]:
                raise ValueError(f"Annotation hash mismatch: {file}")
        report[protocol] = {"manifest_sha256": sha(root / "manifest.json"),
                            "files_verified": len(manifest["files"]), "splits": manifest["splits"]}
    for model in plan["models"]:
        if model.get("checkpoint"):
            selection = read(model["selection"])
            if selection["selected_on"] != "val" or sha(model["checkpoint"]) != selection["coarse_sha256"]:
                raise ValueError("Checkpoint differs from validation-frozen selection")
            if selection["formal_fingerprint"] != plan["protocols"][model["protocol"]]["fingerprint"]:
                raise ValueError("Checkpoint protocol mismatch")
    return report


def prepare(plan, root):
    report = verify_inputs(plan)
    human = read(plan["human_review"])
    if human.get("review_scope") != "original_full_text_against_source_image":
        raise ValueError("Ambiguous or incompatible human review scope")
    keep, all_places, uncertain = set(), {}, {}
    for protocol, item in plan["protocols"].items():
        places = [p for p in lines(Path(item["formal"]) / "valid.jsonl") if p["split"] == "test"]
        counts = Counter(t.get("quality_status", "valid") for p in places for t in p["texts"])
        if set(counts) != {"valid"}:
            raise ValueError("Original main texts changed")
        all_places[protocol] = places
        keep.update(p["place_id"] for p in places)
    for p in lines(plan["uncertain_file"]):
        if p["place_id"] in keep:
            uncertain[p["place_id"]] = p["texts"]
    specs = {k: dataset_spec(v, uncertain, plan["gallery_seeds"]) for k, v in all_places.items()}
    # Blind quality-review sample: chosen without model scores, balanced by rule status.
    review = []
    p1_lookup = {p["place_id"]: p for p in all_places["P1"]}
    candidates = defaultdict(list)
    for p in all_places["P1"]:
        for t in uncertain.get(p["place_id"], []):
            if t.get("quality_status") in ("valid", "uncertain") and t.get("source_images"):
                candidates[t["quality_status"]].append((p, t))
    for status in ("valid", "uncertain"):
        for p, t in ordered(candidates[status], 20271109, lambda x: x[1]["text_id"])[:120]:
            review.append({"audit_id": "", "place_id": p["place_id"], "text_id": t["text_id"],
                           "text": t["text"], "source_images": t["source_images"],
                           "target_images": p["target_image_paths"], "rule_status": status,
                           "source_correctness": "", "target_visibility": "", "place_specificity": "", "comment": ""})
    review = ordered(review, 20271110, lambda r: r["text_id"])
    for i, row in enumerate(review, 1):
        row["audit_id"] = f"Q{i:03d}"
    report["human_review"] = {"sha256": sha(plan["human_review"]), "scope": human["review_scope"],
                               "counts": dict(Counter(r["human_judgment"] for r in human["rows"])),
                               "n": len(human["rows"]),
                               "limitation": "Historical source-image audit only; not representative target-visibility ground truth"}
    image_rows = {}
    for spec in specs.values():
        for row in feature_records(spec)["image"]:
            image_rows[row["path"]] = row
    for row in review:
        for image in p1_lookup[row["place_id"]]["images"]:
            image_rows[image["path"]] = image
    def verify_image(row):
        prefix = plan["canonical_prefix"].rstrip("/") + "/"
        if not row["path"].startswith(prefix):
            raise ValueError("Image path outside canonical prefix")
        local = Path(plan["image_root"]) / row["path"][len(prefix):]
        if not local.resolve().is_relative_to(Path(plan["image_root"]).resolve()):
            raise ValueError("Unsafe image path")
        actual = sha(local)
        if actual != row["sha256"]:
            raise ValueError(f"Image identity mismatch: {local}")
        return {"path": row["path"], "sha256": actual}
    with ThreadPoolExecutor(max_workers=8) as pool:
        image_audit = list(pool.map(verify_image, image_rows.values()))
    report["images"] = {"verified": len(image_audit), "mismatches": 0, "missing": 0}
    report["specifications"] = {k: {"places": len(v["pairs"]), "excluded": len(v["excluded_provenance"]),
                                    "quality_pair_counts": {name: len(x) for name, x in v["quality"].items()}}
                                 for k, v in specs.items()}
    immutable(root / "input_audit.json", report)
    immutable(root / "image_audit.json", image_audit)
    for protocol, spec in specs.items():
        immutable(root / "specs" / f"{protocol}.json", spec)
    immutable(root / "quality_review_private.json", {"schema": VERSION, "rows": review})
    print(json.dumps({"stage": "prepared", **report["specifications"], "images": report["images"]}), flush=True)
    return specs


def model_config(model, plan):
    c = read(model["config"])
    if model.get("checkpoint"):
        expected = plan["protocols"][model["protocol"]]["fingerprint"]
        if c["data"]["fingerprint"] != expected:
            raise ValueError("Model configuration belongs to another protocol")
    else:
        c["models"]["siglip"] = model["directory"]
        c["training"].update(adapt_towers=[], residual_heads=False)
        if model["kind"] == "blip":
            c["training"]["text_length"] = 64
    c["data"].update(image_root=plan["image_root"], canonical_prefix=plan["canonical_prefix"])
    c["evaluation"].update(encode_batch_size=plan["encode_batch_size"], workers=plan["workers"])
    return c


def feature_bank(model, protocol, spec, plan, root):
    import torch
    from place_retrieval.encoding import encode_split
    from place_retrieval.evaluation import checkpoint_state
    from place_retrieval.io import model_identity, save_torch, seed_all
    from place_retrieval.models import load_siglip
    rows = feature_records(spec)
    c = model_config(model, plan)
    seed_all(42)
    identity = {"version": VERSION, "model": model["id"], "protocol": protocol,
                "model_files": model_identity(c["models"]["siglip"]),
                "checkpoint": sha(model["checkpoint"]) if model.get("checkpoint") else None,
                "rows": digest(rows), "precision": "cuda_bfloat16_project_fp32", "plan": digest(plan)}
    destination = root / "features" / f"{model['id']}_{protocol}.pt"
    receipt = destination.with_suffix(".json")
    if receipt.exists():
        previous = read(receipt)
        if previous["identity"] != identity or sha(destination) != previous["sha256"]:
            raise ValueError("Feature cache identity/hash mismatch")
        return torch.load(destination, weights_only=True, map_location="cpu")
    device = torch.device("cuda")
    if model.get("checkpoint"):
        manifest = read(Path(plan["protocols"][protocol]["formal"]) / "manifest.json")
        backbone, processor, heads, _ = checkpoint_state(model["checkpoint"], c, manifest, device)
        # Reject evaluation places that occurred in this supervised model's training data.
        train_ids = {p["place_id"] for p in lines(Path(plan["protocols"][protocol]["formal"]) / "valid.jsonl") if p["split"] == "train"}
        if train_ids & {r["place_id"] for r in rows["image"]}:
            raise ValueError("Supervised evaluation leaks training places")
    else:
        backbone, processor, heads = load_siglip(c["models"]["siglip"], c["training"], device)
    started = time.time()
    print(json.dumps({"stage": "encoding", "model": model["id"], "protocol": protocol,
                      "images": len(rows["image"]), "texts": len(rows["text"])}), flush=True)
    features = encode_split(backbone, processor, heads, rows, c, device)
    if any(not torch.isfinite(x).all() for x in features.values()):
        raise ValueError("Non-finite features")
    bank = {"features": features, "records": rows, "identity": identity}
    save_torch(destination, bank)
    write(receipt, {"identity": identity, "sha256": sha(destination), "seconds": time.time() - started})
    del backbone, processor, heads
    torch.cuda.empty_cache()
    return bank


def retrieval_hits(query_features, gallery_features, query_ids, gallery_ids):
    """Exact full-gallery retrieval; stable gallery order resolves exact ties."""
    import numpy as np
    q = np.asarray(query_features, dtype=np.float32)
    g = np.asarray(gallery_features, dtype=np.float32)
    if len(q) != len(query_ids) or len(g) != len(gallery_ids) or not len(q) or not len(g):
        raise ValueError("Empty or inconsistent retrieval inputs")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Non-finite retrieval features")
    known = set(gallery_ids)
    if any(p not in known for p in query_ids):
        raise ValueError("A query has no gallery positive")
    labels = np.array(gallery_ids)
    hits = []
    for start in range(0, len(q), 128):
        scores = q[start:start + 128] @ g.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :min(10, len(g))]
        correct = labels[order] == np.array(query_ids[start:start + 128])[:, None]
        hits.extend(np.stack([correct[:, :k].any(axis=1) for k in (1, 5, 10)], axis=1).tolist())
    return hits


def metrics(hits):
    if not hits:
        raise ValueError("No query hits")
    return {f"Recall@{k}": sum(row[i] for row in hits) / len(hits) for i, k in enumerate((1, 5, 10))}


def paired_ci(a, b, seed=2027, repetitions=2000):
    """Bootstrap paired place-query differences, in percentage points."""
    import numpy as np
    if len(a) != len(b) or not a:
        raise ValueError("Paired samples required")
    values = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.array([values[rng.integers(len(values), size=len(values))].mean(0) for _ in range(repetitions)])
    return {f"Recall@{k}": {"b_minus_a_pp": float(values[:, i].mean() * 100),
                             "ci95_pp": (np.quantile(stats[:, i], [.025, .975]) * 100).tolist()}
            for i, k in enumerate((1, 5, 10))}


class Evaluator:
    def __init__(self, bank):
        self.features = {k: v.numpy() for k, v in bank["features"].items()}
        self.index = {k: {r["image_id" if k == "image" else "text_id"]: i for i, r in enumerate(rows)}
                      for k, rows in bank["records"].items()}

    def run(self, images, texts, query_images=None, query_texts=None):
        def features(name, rows):
            key = "image_id" if name == "image" else "text_id"
            return self.features[name][[self.index[name][r[key]] for r in rows]]
        query_images = images if query_images is None else query_images
        query_texts = texts if query_texts is None else query_texts
        t2i = retrieval_hits(features("text", query_texts), features("image", images),
                             [r["place_id"] for r in query_texts], [r["place_id"] for r in images])
        i2t = retrieval_hits(features("image", query_images), features("text", texts),
                             [r["place_id"] for r in query_images], [r["place_id"] for r in texts])
        return {"T2I": metrics(t2i), "I2T": metrics(i2t)}, {"T2I": t2i, "I2T": i2t}


def evaluate_model(model, protocol, spec, bank, plan, root):
    dest = root / "results" / f"{model['id']}_{protocol}.json"
    identity = {"plan": digest(plan), "features": bank["identity"], "spec": digest(spec)}
    if dest.exists():
        previous = read(dest)
        if previous["identity"] != identity:
            raise ValueError("Result identity changed")
        return previous
    engine = Evaluator(bank)
    pairs = spec["pairs"]
    lookup = {p["place_id"]: p for p in pairs}
    texts = [p["text"] for p in pairs]
    same, same_hits = engine.run([p["source"] for p in pairs], texts)
    cross, cross_hits = engine.run([p["target"] for p in pairs], texts)
    result = {"identity": identity, "model": model["id"], "family": model["family"],
              "training_seed": model.get("training_seed"), "protocol": protocol,
              "A1": {"places": len(pairs), "same_view": same, "cross_view": cross,
                     "paired_differences": {d: paired_ci(same_hits[d], cross_hits[d]) for d in same_hits},
                     "per_query": {"place_ids": [p["place_id"] for p in pairs],
                                   "same_view_hits": same_hits, "cross_view_hits": cross_hits}},
              "A2": [], "A3": [], "A4": {}}
    for seed, ids in spec["matched_1000"].items():
        selected = [lookup[p] for p in ids]
        values, _ = engine.run([p["target"] for p in selected], [p["text"] for p in selected])
        result["A2"].append({"gallery_seed": int(seed), "places": len(ids), "metrics": values,
                             "city_counts": dict(Counter(p["city"] for p in selected))})
    anchors = [lookup[p] for p in spec["scale_anchors"]]
    for seed, pools in spec["nested_galleries"].items():
        for size, ids in pools.items():
            pool = [lookup[p] for p in ids]
            values, _ = engine.run([p["target"] for p in pool], [p["text"] for p in pool],
                                    [p["target"] for p in anchors], [p["text"] for p in anchors])
            result["A3"].append({"gallery_seed": int(seed), "gallery_places": int(size),
                                 "fixed_query_places": len(anchors), "metrics": values})
    for match, rows in spec["quality"].items():
        if not rows:
            result["A4"][match] = {"status": "unavailable", "reason": "No eligible matched quality pairs", "places": 0}
            continue
        img = [r["image"] for r in rows]
        by_place = {r["place_id"]: r for r in rows}
        # Fix the complete gallery and all background descriptions. Only replace
        # the paired cohort captions, so a small quality group is not an easy gallery.
        all_images = [p["target"] for p in pairs]
        valid_gallery = [by_place[p["place_id"]]["valid"] if p["place_id"] in by_place else p["text"] for p in pairs]
        uncertain_gallery = [by_place[p["place_id"]]["uncertain"] if p["place_id"] in by_place else p["text"] for p in pairs]
        a, ah = engine.run(all_images, valid_gallery, img, [r["valid"] for r in rows])
        b, bh = engine.run(all_images, uncertain_gallery, img, [r["uncertain"] for r in rows])
        result["A4"][match] = {"status": "complete", "places": len(rows), "gallery_places": len(pairs),
                               "rule_valid": a, "rule_uncertain": b,
                               "paired_differences": {d: paired_ci(ah[d], bh[d]) for d in ah},
                               "valid_mean_words": statistics.mean(len(r["valid"]["text"].split()) for r in rows),
                               "uncertain_mean_words": statistics.mean(len(r["uncertain"]["text"].split()) for r in rows),
                               "valid_types": dict(Counter(r["valid"].get("type") for r in rows)),
                               "uncertain_types": dict(Counter(r["uncertain"].get("type") for r in rows)),
                               "limitation": "Rule-status association, not human correctness or a causal quality intervention"}
    write(dest, result)
    print(json.dumps({"stage": "model_complete", "model": model["id"], "protocol": protocol,
                      "A1_cross": cross, "quality_n": {k: v["places"] for k, v in result["A4"].items()}}), flush=True)
    return result


def export_results(root, plan):
    all_results = [read(p) for p in sorted((root / "results").glob("*.json"))]
    rows = []
    for r in all_results:
        common = {k: r[k] for k in ("model", "family", "training_seed", "protocol")}
        for condition in ("same_view", "cross_view"):
            for direction, values in r["A1"][condition].items():
                rows.append(dict(common, experiment="A1", condition=condition, gallery_places=r["A1"]["places"],
                                 gallery_seed="", query_places=r["A1"]["places"], direction=direction, **values))
        for name in ("A2", "A3"):
            for item in r[name]:
                for direction, values in item["metrics"].items():
                    rows.append(dict(common, experiment=name, condition="cross_view", gallery_places=item.get("gallery_places", item.get("places")),
                                     gallery_seed=item["gallery_seed"], query_places=item.get("fixed_query_places", item.get("places")),
                                     direction=direction, **values))
        for condition, item in r["A4"].items():
            if item["status"] != "complete":
                continue
            for status in ("rule_valid", "rule_uncertain"):
                for direction, values in item[status].items():
                    rows.append(dict(common, experiment="A4", condition=f"{condition}/{status}", gallery_places=item["gallery_places"],
                                     gallery_seed="", query_places=item["places"], direction=direction, **values))
    if rows:
        with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(root / "summary.json", {"version": VERSION, "plan_sha256": sha(root / "plan.json"),
                                  "completed_models": len(all_results), "metric_rows": len(rows),
                                  "human_quality_evaluation": "pending independent expanded human annotations",
                                  "historical_human_audit": read(root / "input_audit.json")["human_review"]})
    report = ["# AcrossViews benchmark diagnostics", "",
              "All scores below are Recall percentages. No hyperparameters were selected on these diagnostic outcomes.", "",
              "## Scope and interpretation", "",
              "- Three reference families: pretrained SigLIP2, pretrained BLIP ITC, and place-supervised SigLIP2 (training seeds 42/43/44). No new training or BLIP reranking is performed here.",
              "- Primary controlled pools contain one provenance-bound description and one image per place. These are diagnostic results and must not replace the original all-sample benchmark scores.",
              "- A1: exactly the same description/place cohort and gallery cardinality; image evidence switches from that caption's source view to an independent target view. I2T images change with the condition and are paired by place.",
              "- A2: equal 1,000-place galleries, five metadata-only sampling seeds. Pretrained controls fix weights across protocols. Supervised P0/P1 models have different training sizes and city distributions, so their difference is a full-protocol comparison, not a causal estimate of city shift alone.",
              "- A3: the same 500 query places and their fixed queries are retained in nested galleries; candidate pools are defined before scoring. No candidate is inserted into a scored shortlist.",
              "- A4: valid/uncertain are automated rule statuses, not correctness labels. Both conditions retain the full controlled image gallery and the same text-gallery cardinality/background captions; only paired-cohort captions change. The matched analysis controls caption type, provenance view, and word-count bin; unadjusted comparisons remain confounded.",
              "- Historical human annotations concern 30 original descriptions against source images, not target visibility or population-level quality. A new blinded 240-description review packet is supplied; formal human-quality retrieval analysis remains pending its annotations.", "",
              "## A1: full controlled cohort", "",
              "| Model | Protocol | Train seed | Places | Source T2I R@1 | Cross T2I R@1 | Source I2T R@1 | Cross I2T R@1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in all_results:
        a, b = r["A1"]["same_view"], r["A1"]["cross_view"]
        report.append(f"| {r['model']} | {r['protocol']} | {r['training_seed'] or 'n/a'} | {r['A1']['places']} | {a['T2I']['Recall@1']*100:.3f} | {b['T2I']['Recall@1']*100:.3f} | {a['I2T']['Recall@1']*100:.3f} | {b['I2T']['Recall@1']*100:.3f} |")
    report += ["", "## A2: matched gallery size", "",
               "Means below average five gallery samples within each fixed checkpoint. This spread is gallery-sampling variability, not training-seed uncertainty.", "",
               "| Model | Protocol | Gallery places | T2I R@1 mean ± sampling SD | I2T R@1 mean ± sampling SD |",
               "|---|---|---:|---:|---:|"]
    for r in all_results:
        values = {d: [item["metrics"][d]["Recall@1"] * 100 for item in r["A2"]] for d in ("T2I", "I2T")}
        report.append(f"| {r['model']} | {r['protocol']} | 1000 | {statistics.mean(values['T2I']):.3f} ± {statistics.stdev(values['T2I']):.3f} | {statistics.mean(values['I2T']):.3f} ± {statistics.stdev(values['I2T']):.3f} |")
    report += ["", "## Detailed deliverables", "",
               "`metrics.csv` contains every R@1/5/10 measurement and distinguishes training seeds from gallery seeds. `results/*.json` retains paired query hit vectors and paired bootstrap differences for A1/A4. `specs/*.json` contains the exact queries, provenance, quality pairs, and nested gallery identities.", "",
               "Human audit counts: " + json.dumps(read(root / "input_audit.json")["human_review"]["counts"]), "",
               "This report updates incrementally. Consult `status.json` and `completion.json` for completion status."]
    (root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=True)
    from place_retrieval.io import run_lock
    with run_lock(root):
        immutable(root / "configuration_freeze.json", {"version": VERSION, "plan_sha256": sha(args.plan),
                  "code_sha256": sha(__file__), "release_manifest_sha256": sha(root / "code/release_files.json"),
                  "test_hyperparameter_search": False})
        try:
            write(root / "status.json", {"stage": "preparing", "time": time.time(), "pid": os.getpid()})
            specs = prepare(plan, root)
            if args.prepare_only:
                write(root / "status.json", {"stage": "prepared", "time": time.time(), "pid": os.getpid()})
                return
            import torch
            torch.set_num_threads(plan["cpu_threads"])
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            for model in plan["models"]:
                for protocol in ([model["protocol"]] if model.get("protocol") else ("P0", "P1")):
                    write(root / "status.json", {"stage": "encoding_or_evaluating", "model": model["id"],
                          "protocol": protocol, "time": time.time(), "pid": os.getpid()})
                    bank = feature_bank(model, protocol, specs[protocol], plan, root)
                    evaluate_model(model, protocol, specs[protocol], bank, plan, root)
                    export_results(root, plan)
            write(root / "completion.json", {"status": "automatic_diagnostics_complete", "time": time.time(),
                  "plan_sha256": sha(args.plan), "human_quality_evaluation": "pending independent expanded human annotations"})
            write(root / "status.json", read(root / "completion.json"))
        except BaseException as error:
            write(root / "status.json", {"stage": "failed", "time": time.time(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
