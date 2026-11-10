"""Compare this evaluator against an existing archived report without model inference."""
import argparse
import json
import math
from pathlib import Path

import torch

from place_retrieval.metrics import evaluate, shortlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    candidates = torch.load(args.candidates, map_location="cpu", weights_only=True)
    scores = torch.load(args.scores, map_location="cpu", weights_only=True)
    if isinstance(scores, dict):
        scores = scores["itm"]
    archived = json.loads(args.report.read_text(encoding="utf-8"))
    weight = archived["selection"]["global_weight"]
    reports = {
        "selected_model": evaluate(candidates, scores, weight),
        "coarse_only": evaluate(candidates, candidates["coarse_scores"], 0),
    }
    for variant, report in reports.items():
        for direction in ("T2I", "I2T"):
            for metric, value in report[direction].items():
                expected = archived[variant][direction][metric]
                if not math.isclose(value, expected, abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"Archived metric mismatch: {variant}/{direction}/{metric}")
    if args.bank:
        bank = torch.load(args.bank, map_location="cpu", weights_only=True)
        regenerated = shortlist(bank["features"], bank["records"], candidates["t2i"].shape[1])
        for key in ("pairs", "t2i", "i2t", "coarse_scores", "t2i_offsets", "i2t_offsets"):
            if not torch.equal(regenerated[key], candidates[key]):
                raise ValueError(f"Archived shortlist mismatch: {key}")
    print(json.dumps({"status": "passed", "metrics_compared": 12,
                      "shortlist_compared": args.bank is not None, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
