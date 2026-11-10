"""Historical sample-ranked, place-ID-positive retrieval metrics."""
import torch


def exact_topk(query, gallery, count, block=256):
    """Block query rows, retaining the complete gallery for exact retrieval."""
    return torch.cat([(query[start:start + block] @ gallery.T).topk(min(count, len(gallery)), dim=1).indices
                      for start in range(0, len(query), block)])


def coverage(features, rows):
    """Evaluate the full split gallery and the shortlist oracle."""
    places = sorted({r["place_id"] for values in rows.values() for r in values})
    ids = {place: i for i, place in enumerate(places)}
    labels = {m: torch.tensor([ids[r["place_id"]] for r in values], device=features["text"].device)
              for m, values in rows.items()}
    result = {}
    for direction, query, gallery in (
        ("T2I", "text", "image"), ("I2T", "image", "text")
    ):
        order = exact_topk(features[query], features[gallery], 200)
        hits = labels[gallery][order] == labels[query][:, None]
        result[direction] = {
            f"Recall@{k}": hits[:, :k].any(dim=1).float().mean().item()
            for k in (1, 5, 10, 20, 50, 100, 200)
        }
    return result


def shortlist(features, rows, count):
    if count < 10:
        raise ValueError("At least 10 candidates are needed for Recall@10")
    t2i = exact_topk(features["text"], features["image"], count)
    i2t = exact_topk(features["image"], features["text"], count)
    codes = torch.cat([
        (torch.arange(len(rows["text"]))[:, None] * len(rows["image"]) + t2i).flatten(),
        (i2t * len(rows["image"]) + torch.arange(len(rows["image"]))[:, None]).flatten(),
    ]).unique(sorted=True)
    pairs = torch.stack([codes // len(rows["image"]), codes % len(rows["image"])], dim=1)
    return {
        "records": rows, "pairs": pairs, "t2i": t2i, "i2t": i2t,
        "coarse_scores": torch.cat([
            (features["text"][part[:, 0]] * features["image"][part[:, 1]]).sum(-1)
            for part in pairs.split(4096)
        ]),
        "t2i_offsets": torch.searchsorted(codes, (
            torch.arange(len(rows["text"]))[:, None] * len(rows["image"]) + t2i)),
        "i2t_offsets": torch.searchsorted(codes, (
            i2t * len(rows["image"]) + torch.arange(len(rows["image"]))[:, None])),
    }


def recall(rankings, queries, gallery):
    hits = []
    for query, indices in zip(queries, rankings.tolist()):
        hits.append([gallery[index]["place_id"] == query["place_id"] for index in indices])
    return {f"Recall@{k}": sum(any(row[:k]) for row in hits) / len(hits) for k in (1, 5, 10)}


def evaluate(candidates, itm, weight):
    if len(itm) != len(candidates["pairs"]) or not torch.isfinite(itm).all():
        raise ValueError("The pair scores are incomplete or non-finite")
    scores = itm.float() + weight * candidates["coarse_scores"]
    rows = candidates["records"]
    result = {}
    for direction, key, query, gallery in (
        ("T2I", "t2i", "text", "image"), ("I2T", "i2t", "image", "text")
    ):
        values = scores[candidates[key + "_offsets"]]
        order = values.topk(min(10, values.shape[1]), dim=1).indices
        ranking = candidates[key].gather(1, order)
        result[direction] = recall(ranking, rows[query], rows[gallery])
    return result


def select_fusion(candidates, scores, weights, split):
    if split != "val":
        raise ValueError("Fusion selection is only permitted on validation")
    if not weights or any(not torch.isfinite(torch.tensor(float(w))) or w < 0 for w in weights):
        raise ValueError("Fusion weights must be a nonempty list of finite nonnegative numbers")
    reports = [dict(evaluate(candidates, scores, weight), global_weight=weight) for weight in weights]
    return max(reports, key=lambda r: min(r[d]["Recall@1"] for d in ("T2I", "I2T"))), reports
