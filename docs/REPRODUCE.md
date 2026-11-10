# Reproduce training, selection and testing

## Single complete baseline

Install the root project and download the pinned model revisions with `scripts/download_models.py`. Set data/model/output paths in `configs/gsv_full.yaml`. Run from the repository root so relative paths resolve consistently.

```bash
place-retrieval verify-data --config configs/gsv_full.yaml --check-images
place-retrieval train --config configs/gsv_full.yaml
place-retrieval validate --config configs/gsv_full.yaml \
  --checkpoint runs/gsv_full/coarse/best.pt
place-retrieval test --config configs/gsv_full.yaml \
  --selection runs/gsv_full/evaluation/selection.json
```

`place-retrieval run --config configs/gsv_full.yaml` performs the three stages in one command. `train --resume` resumes the same configuration from `coarse/last.pt`; use a new output directory for a changed experiment. Both language and visual encoders participate through LoRA in the default recipe; their base weights remain frozen. Residual heads are identity-initialized. Full end-to-end fine-tuning is not claimed.

The training objective is place-level symmetric multi-positive contrastive loss. Batch composition defaults to 16 places, two images and up to four texts per place. Checkpoint selection first maximizes the weaker-direction validation Recall@100, then weaker-direction Recall@1, retaining the earlier epoch on a tie. Epoch zero is a genuine pretrained initialization candidate and may remain selected. BLIP reranking evaluates a declared fusion grid only on validation; the chosen setting and checkpoint hash are checked again at test time.

## All declared methods and adaptation ablations

Prepare separate study roots. Copy P0 formal data to `runs/P0/protocol_p0/formal/`. Generate P1 from those exact inputs:

```bash
python -m place_retrieval.study.protocol \
  --source datasets/formal --destination runs/P1/protocol_p1/formal
python scripts/freeze_source.py
python -m place_retrieval.study.plan --root runs/P0 \
  --base-config configs/gsv_full.yaml --protocol P0 \
  --expected-fingerprint 0aa126e367f28be0b4f63356081c55dc9c3525c99cd70ba81a5689dd9aa06d4a
python -m place_retrieval.study.plan --root runs/P1 \
  --base-config configs/gsv_full.yaml --protocol P1 \
  --expected-fingerprint fa9e4fe583d0e364883aa4a9629caf287e31d50e4f6a52babbbe476d7e430378
```

Each plan fixes 24 adapted training runs (eight recipes × three seeds) and four pretrained references. Run roles sequentially on one GPU or assign independent roles to separate machines with identical data/source/model identities:

```bash
python -m place_retrieval.study.worker --plan runs/P0/plan.json --role comparison
python -m place_retrieval.study.worker --plan runs/P0/plan.json --role auxiliary
python -m place_retrieval.study.worker --plan runs/P0/plan.json --role ablation
python scripts/study_test.py --root runs/P0 --freeze
python scripts/study_test.py --root runs/P0 --role comparison
python scripts/study_test.py --root runs/P0 --role auxiliary
python scripts/study_test.py --root runs/P0 --role ablation
```

Repeat with `runs/P1`. Freezing requires all planned training/validation jobs to finish. The test driver hashes job configurations, checkpoints, completions and selections before evaluating the test split. On multiple physical machines, collect the role artifacts to one study root before this global freeze, or share the same filesystem; host-specific transfer and shutdown scripts are intentionally not part of the public research code.

| Job | Recipe |
|---|---|
| Z_clip16 / Z_clip32 | Pretrained CLIP, coarse only |
| Z3_siglip / Z4_blip_itc | Pretrained SigLIP2 / BLIP ITC |
| T1_clip16_place | CLIP ViT-B/16, place-supervised adaptation |
| T2_lavpr_blip_ms | BLIP ITC, all-linear rank-64 rsLoRA, I2T MS, scaled SGD |
| T3_siglip_ms | SigLIP2 with symmetric MS |
| T4_siglip_place | SigLIP2 with symmetric place loss and residual heads |
| A1 / A2 / A3 / A4 | Heads only / no residual / vision only / text only |

All adapted runs initialize independently from public pretrained weights, use seeds 42/43/44 and the common three-epoch budget. No SALAD aggregation is used. The portable planner preserves archived numerical recipes while parameterizing paths and protocol choice; it creates a new plan identity rather than impersonating the historical deployment.

## Outputs

`coarse/` contains `best.pt`, `last.pt`, configuration, history and completion. `evaluation/selection.json` stores the frozen validation choice. `evaluation/{val,test}_result.json` contains coarse and selected reranking Recall@1/5/10 in **fractions**, whereas release aggregate tables express **percentages**. Test feature/score caches remain local and can be compared using `scripts/check_archived_scores.py`.

Disk space matters: each active job needs its checkpoint pair and feature/reranking caches. Workers reserve at least 25 GiB before starting and stop below 20 GiB. They never delete unique checkpoints to make room. Set encoding batch sizes/workers to suit hardware in a new configuration; this changes configuration identity. Training needs local model snapshots, real data and CUDA; the CPU test suite proves software invariants, not GPU throughput or retrained benchmark parity.
