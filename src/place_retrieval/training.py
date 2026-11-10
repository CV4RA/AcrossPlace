"""Place-supervised LoRA training with epoch-boundary continuation."""
import math
import time
from pathlib import Path

import torch
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader

from .data import ImageDataset, records, training_batches, verify_formal
from .encoding import encode_split
from .io import digest, log, model_identity, read_json, run_lock, save_torch, seed_all, write_json
from .metrics import coverage
from .models import adapter_state, context_length, load_siglip, raw_features, restore_coarse


def gps_unit(points, device):
    gps = torch.tensor([p if p is not None else [float("nan"), float("nan")] for p in points],
                       dtype=torch.float64, device=device) * math.pi / 180
    return torch.stack([torch.cos(gps[:, 0]) * torch.cos(gps[:, 1]),
                        torch.cos(gps[:, 0]) * torch.sin(gps[:, 1]), torch.sin(gps[:, 0])], dim=1)


def place_loss(images, texts, image_rows, text_rows, temperature, tolerance_m):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = images.device
    positives = torch.tensor([[a["place_id"] == b["place_id"] for b in text_rows]
                              for a in image_rows], device=device)
    if not positives.any(0).all() or not positives.any(1).all():
        raise ValueError("Each sample needs a same-place counterpart")
    a = gps_unit([r.get("gps") for r in image_rows], device)
    b = gps_unit([r.get("gps") for r in text_rows], device)
    chord = (a[:, None] - b[None, :]).square().sum(-1).sqrt().clamp(0, 2)
    distance = 6371000 * 2 * torch.asin(chord / 2)
    allowed = ~(distance <= tolerance_m) | positives
    logits = (images.float() @ texts.float().T) / temperature
    logits = logits.masked_fill(~allowed, -torch.inf)

    def direction(scores, mask):
        probabilities = scores.log_softmax(dim=1)
        return -(probabilities.masked_fill(~mask, 0).sum(1) / mask.sum(1)).mean()

    return 0.5 * (direction(logits, positives) + direction(logits.T, positives.T))


class StepCosine:
    def __init__(self, steps, warmup):
        self.steps = steps
        self.warmup = int(steps * warmup)

    def __call__(self, step):
        if step < self.warmup:
            return (step + 1) / max(1, self.warmup)
        progress = min(1, (step - self.warmup) / max(1, self.steps - self.warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))


def selection_key(report):
    return (min(report[d]["Recall@100"] for d in ("T2I", "I2T")),
            min(report[d]["Recall@1"] for d in ("T2I", "I2T")))


def objective(images, texts, image_rows, text_rows, t):
    if t.get("loss", "place") == "place":
        return place_loss(images, texts, image_rows, text_rows, t["temperature"], t["tolerance_m"])
    if t["loss"] != "ms":
        raise ValueError("Unknown contrastive objective")
    from pytorch_metric_learning.distances import DotProductSimilarity
    from pytorch_metric_learning.losses import MultiSimilarityLoss
    from pytorch_metric_learning.miners import MultiSimilarityMiner
    ids = sorted({r["place_id"] for r in image_rows + text_rows})
    lookup = {value: index for index, value in enumerate(ids)}
    image_ids = torch.tensor([lookup[r["place_id"]] for r in image_rows], device=images.device)
    text_ids = torch.tensor([lookup[r["place_id"]] for r in text_rows], device=images.device)
    miner = MultiSimilarityMiner(epsilon=0.1, distance=DotProductSimilarity())
    loss = MultiSimilarityLoss(alpha=2, beta=40, base=0.5, distance=DotProductSimilarity())
    def direction(a, b, a_ids, b_ids):
        pairs = miner(a, a_ids, ref_emb=b, ref_labels=b_ids)
        return loss(a, a_ids, indices_tuple=pairs, ref_emb=b, ref_labels=b_ids)
    i2t = direction(images.float(), texts.float(), image_ids, text_ids)
    if t.get("ms_direction", "symmetric") == "i2t":
        return i2t
    return 0.5 * (i2t + direction(texts.float(), images.float(), text_ids, image_ids))


def train(c, resume=False):
    root = Path(c["output"]) / "coarse"
    with run_lock(root):
        return _train(c, root, resume)


def _train(c, root, resume):
    manifest = verify_formal(c["data"])
    t = c["training"]
    if t["epochs"] < 1 or t["places_per_batch"] < 1:
        raise ValueError("Training epochs and batch size must be positive")
    device = torch.device(c["device"])
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires CUDA BF16 support")
    seed_all(c["seed"])
    rows = records(c["data"], "train")
    held_out = records(c["data"], "val")
    identity = {"formal_fingerprint": manifest["fingerprint"],
                "config": digest(c), "siglip": model_identity(c["models"]["siglip"])}
    last = root / "last.pt"
    if resume != last.is_file():
        raise ValueError("Use a fresh output, or --resume with an existing coarse/last.pt")
    if (root / "identity.json").exists() and read_json(root / "identity.json") != identity:
        raise ValueError("The run configuration, pretrained model, or data changed")
    write_json(root / "identity.json", identity)
    write_json(root / "config.json", c)
    backbone, processor, heads = load_siglip(c["models"]["siglip"], t, device)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batches = [training_batches(rows, t, c["seed"] + epoch) for epoch in range(t["epochs"])]
    if any(len(batch) != len(batches[0]) for batch in batches):
        raise ValueError("The epoch lengths differ")
    groups = [group for group in (
        {"params": list(heads.parameters()), "lr": t["head_lr"]},
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": t["lora_lr"]}
    ) if group["params"]]
    if not groups:
        raise ValueError("The training variant has no trainable parameters")
    optimizer = (torch.optim.SGD(groups, momentum=0.9, weight_decay=t["weight_decay"])
                 if t.get("optimizer") == "sgd" else torch.optim.AdamW(groups, weight_decay=t["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, StepCosine(sum(map(len, batches)), t["warmup_ratio"])
    )
    history = []
    start_epoch = 0
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("The checkpoint belongs to another run")
        restore_coarse(backbone, heads, state)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        history = state["history"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    for epoch in range(start_epoch, t["epochs"]):
        backbone.train()
        heads.train()
        loader = DataLoader(ImageDataset(rows["image"], processor, c["data"]),
                            batch_sampler=[batch["image"] for batch in batches[epoch]],
                            num_workers=t["workers"], pin_memory=device.type == "cuda",
                            **({"multiprocessing_context": "spawn"} if t["workers"] else {}))
        total = 0
        started = time.monotonic()
        length = context_length(backbone, t)
        for step, (batch, pixels) in enumerate(zip(batches[epoch], loader), start=1):
            image_rows = [rows["image"][i] for i in batch["image"]]
            text_rows = [rows["text"][i] for i in batch["text"]]
            tokens = processor(text=[r["text"] for r in text_rows], padding="max_length",
                               max_length=length, truncation=True, return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                image_raw = raw_features(backbone, pixels.to(device), "image")
                text_raw = raw_features(backbone, tokens, "text")
            images = heads.encode(image_raw, "image")
            texts = heads.encode(text_raw, "text")
            loss = objective(images, texts, image_rows, text_rows, t)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [p for group in groups for p in group["params"]]
            norm = torch.nn.utils.clip_grad_norm_(parameters, t["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite training gradient")
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if step == 1 or step % 100 == 0 or step == len(loader):
                log({"epoch": epoch + 1, "step": step, "steps": len(loader),
                     "loss": loss.item(), "mean_loss": total / step,
                     "lr": [group["lr"] for group in optimizer.param_groups]}, root / "events.jsonl")
        features = encode_split(backbone, processor, heads, held_out, c, device)
        validation = coverage({m: v.to(device) for m, v in features.items()}, held_out)
        history.append({"epoch": epoch + 1, "loss": total / len(loader),
                        "seconds": time.monotonic() - started, "validation": validation})
        state = {"format": "siglip2-place-lora-v1", "identity": identity, "epoch": epoch + 1,
                 "lora": adapter_state(backbone), "heads": heads.state_dict(),
                 "training": t, "validation": validation}
        previous = history[:-1]
        # Keep the first selected epoch when scores tie.
        if not previous or selection_key(validation) > max(selection_key(row["validation"]) for row in previous):
            save_torch(root / "best.pt", state)
        save_torch(last, dict(state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             history=history, rng_cpu=torch.get_rng_state(),
                             rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        write_json(root / "history.json", history)
        log(history[-1], root / "events.jsonl")
    write_json(root / "completion.json", {"status": "complete", "identity": identity, "history": history})
    return root / "best.pt"
