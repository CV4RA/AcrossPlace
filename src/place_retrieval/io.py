"""Configuration, content identities, and atomic experiment files."""
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def log(event, path=None):
    line = json.dumps(event)
    print(line, flush=True)
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_identity(directory):
    directory = Path(directory)
    files = [p for p in directory.iterdir()
             if p.is_file() and p.suffix in (".json", ".txt", ".model", ".safetensors")]
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"Expected safetensors weights in {directory}")
    return {p.name: sha256(p) for p in sorted(files)}


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
