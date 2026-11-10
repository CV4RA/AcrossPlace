# Frozen benchmark diagnostics

`benchmark.py` implements the automated A1–A4 diagnostic suite independently of training:

1. **Paired source/target transfer:** same caption and place pool; substitute source versus independent target image.
2. **Matched candidate pools:** 1,000-place galleries, five gallery seeds per fixed checkpoint.
3. **Nested gallery scaling:** 500 fixed query places and increasing galleries, retaining natural positives.
4. **Quality analysis:** within-place valid/uncertain contrasts, optionally matched on type, source and word-count bin; separate human source/target/specificity labels.

`plan.archive.json` preserves the original machine paths and SHA256 inventory for provenance only. It is **not a runnable portable configuration**. Make a new immutable plan for your local formal files and trained checkpoints with `prepare_plan.py`; the original experiment identity is retained separately, never forged. Checkpoints and all listed model files must be present and hashed. No test-set fitting is performed.

```bash
python diagnostics/prepare_plan.py --root runs/diagnostics \
  --p0-formal datasets/formal --p1-formal runs/P1/protocol_p1/formal \
  --p0-runs runs/P0/runs --p1-runs runs/P1/runs \
  --image-root datasets/GSVCities --model-root artifacts
python diagnostics/benchmark.py --plan runs/diagnostics/plan.json --prepare-only
python diagnostics/benchmark.py --plan runs/diagnostics/plan.json
python diagnostics/make_review.py --root runs/diagnostics
# The released 240 independent human judgments can be re-evaluated:
ACROSSPLACE_DIAGNOSTICS=runs/diagnostics python diagnostics/human_retrieval.py
```

`make_review.py` creates a blinded HTML review packet; labels must be supplied by a human, not imputed from model predictions. `prepare_plan.py` installs the released 240-row review for the human retrieval analysis and the historical source-only audit for the original A4 report. It restores canonical image prefixes from dataset-relative paths while preserving text and judgments. The 240-row audit has one reviewer, 237 places and balanced rule-status sampling; it does not establish dataset-wide error prevalence. Source support and visibility on either displayed target image are different axes; the I2T aggregate averages two target queries.

Gallery seeds represent candidate-sampling variability; training seeds represent adaptation variability. The archived per-query results under `results/` and the manuscript diagnostics can be inspected without downloading models. To run this isolated suite: `PYTHONPATH=src python -m pytest -q diagnostics/test_benchmark.py`.
