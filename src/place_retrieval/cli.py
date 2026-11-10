"""Command-line entry points for the complete training and evaluation workflow."""
import argparse
from pathlib import Path

import torch

from .io import load_config, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-data", "train", "validate", "test", "run"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--config", default="configs/gsv_full.yaml")
        if name in ("train", "run"):
            subparser.add_argument("--resume", action="store_true")
        if name == "validate":
            subparser.add_argument("--checkpoint", required=True)
        if name == "test":
            subparser.add_argument("--selection", required=True)
        if name == "verify-data":
            subparser.add_argument("--check-images", action="store_true",
                                   help="Also verify SHA256 of all target images (slower)")
    args = parser.parse_args()
    c = load_config(args.config)
    torch.set_num_threads(c.get("cpu_threads", 8))
    if args.command == "verify-data":
        from .data import image_path, records, verify_formal
        from .io import sha256
        manifest = verify_formal(c["data"])
        all_places = set()
        for split in ("train", "val", "test"):
            rows = records(c["data"], split)
            expected = manifest["splits"][split]
            places = {r["place_id"] for r in rows["image"]}
            if (len(rows["image"]) != expected["gallery_images"]
                    or len(rows["text"]) != expected["texts"]
                    or len(places) != expected["places"]
                    or places != {r["place_id"] for r in rows["text"]}
                    or places & all_places):
                raise ValueError(f"Unexpected sample counts in {split}")
            all_places.update(places)
            if args.check_images:
                for row in rows["image"]:
                    path = image_path(row["path"], c["data"])
                    if not row.get("sha256") or sha256(path) != row["sha256"]:
                        raise ValueError(f"Image hash mismatch: {path}")
        log({"fingerprint": manifest["fingerprint"], "splits": manifest["splits"]})
        return
    if args.command in ("train", "run"):
        from .training import train
        checkpoint = train(c, args.resume)
        if args.command == "train":
            log({"best_checkpoint": str(checkpoint)})
            return
    from .evaluation import test, validate
    if args.command == "validate":
        validate(c, Path(args.checkpoint))
    elif args.command == "test":
        test(c, Path(args.selection))
    else:
        selection = validate(c, checkpoint)
        test(c, selection)
