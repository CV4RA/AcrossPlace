"""Download pinned public model snapshots and prepare verified BLIP safetensors."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
import torch


SPECS = {
    "siglip": ("google/siglip2-base-patch16-384", "f775b65a79762255128c981547af89addcfe0ff88"),
    "blip": ("Salesforce/blip-itm-base-coco", "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"),
    "clip16": ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "clip32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert_blip(source, destination):
    """Tensor-only load and exact conversion; retain the original file."""
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only upstream state dictionary")
    state = {k: v.detach().clone().contiguous() for k, v in state.items()}
    temporary = Path(str(destination) + ".tmp")
    save_file(state, str(temporary), metadata={"format": "pt"})
    restored = load_file(str(temporary))
    if set(state) != set(restored) or any(not torch.equal(v, restored[k]) for k, v in state.items()):
        raise ValueError("BLIP safetensors conversion changed tensors")
    temporary.replace(destination)
    return len(state)


def download(kind, output):
    repo, revision = SPECS[kind]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Upstream revision identity differs")
    patterns = ["*.json", "*.txt", "*.model", "*.safetensors"]
    if kind != "siglip":
        patterns.append("pytorch_model.bin")
    root = output / repo.split("/")[-1]
    snapshot_download(repo, revision=revision, local_dir=root, allow_patterns=patterns)
    verification = {"repo_id": repo, "revision": revision}
    if kind != "siglip":
        entry = next(s for s in info.siblings if s.rfilename == "pytorch_model.bin")
        expected = entry.lfs.sha256 if hasattr(entry.lfs, "sha256") else entry.lfs["sha256"]
        source = root / "pytorch_model.bin"
        if sha256(source) != expected:
            raise ValueError("Upstream BLIP weight SHA256 mismatch")
        verification.update(source_sha256=expected,
                            converted_tensors=convert_blip(source, root / "model.safetensors"),
                            safetensors_sha256=sha256(root / "model.safetensors"))
    if not list(root.glob("*.safetensors")):
        raise ValueError("No safetensors weights were prepared")
    (root / "UPSTREAM_VERIFIED.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(root), **verification}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=("all", *SPECS), default="all")
    args = parser.parse_args()
    for kind in SPECS if args.model == "all" else [args.model]:
        download(kind, args.output)


if __name__ == "__main__":
    main()
