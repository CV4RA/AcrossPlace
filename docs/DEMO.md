# Retrieval demonstrations

The public Retrieval Lab is an **archived prediction viewer**. It compares pretrained BLIP, pretrained SigLIP2 and place-supervised SigLIP2 (P1 seed 42), using 12 documented success/failure examples. It does not silently substitute the stronger BLIP+MS baseline. Curated examples are not prevalence estimates.

T2I uses the full P1 test target-image gallery (17,014 images). The I2T diagnostic replaces the focus-place caption in a one-caption-per-place gallery (8,509 places); the website shows predictions for the first target view, while aggregate human-diagnostic results average both target images. Do not compare those I2T case values with the full multi-caption main table.

For real arbitrary-query **coarse retrieval** on your own GPU:

```bash
python scripts/retrieve.py build-index --config configs/gsv_full.yaml \
  --checkpoint runs/gsv_full/coarse/best.pt --index runs/demo-index.pt --split test
python scripts/retrieve.py query --config configs/gsv_full.yaml \
  --checkpoint runs/gsv_full/coarse/best.pt --index runs/demo-index.pt \
  --text "A brick building with a gabled roof and an attached garage." --top-k 5
python scripts/retrieve.py query --config configs/gsv_full.yaml \
  --checkpoint runs/gsv_full/coarse/best.pt --index runs/demo-index.pt \
  --image /path/to/query.jpg --top-k 5
```

The index stores normalized features, original sample records, formal fingerprint, configuration digest and checkpoint SHA256. Querying refuses incompatible data/configuration/checkpoint identities. Scores are cosine similarities, not calibrated probabilities. This demonstration does not apply the BLIP cross-encoder and is not a replacement for fixed benchmark testing. User images/texts are processed locally; no external inference service receives them.
