# Audited experimental evidence

`P0/` and `P1/` retain the collected aggregate metrics, per-seed metrics and training histories. `evidence.json` is the manuscript evidence summary; `diagnostics/` contains measured cross-view, matched-gallery, nested-gallery, caption, city, human-quality and training diagnostics used in the draft. All public Recall values in aggregate tables are **percentages**; evaluator outputs use fractions.

`human_review.json` holds 240 annotations across 237 places from one reviewer, balanced 120/120 by automated rule status. Image paths are dataset-relative; labels and text are unchanged. `qualitative_rankings.json` stores actual top-1 predictions, with explicit gallery scope. A same-place hit is separate from factual caption support or target-view visibility.

Three training seeds are not five gallery draws: their uncertainty estimates have different meanings. P0/P1 comparisons combine several distribution and gallery changes. Human subgroups and post-hoc qualitative examples are descriptive, not randomized interventions. Source-image support, target visibility and place specificity must not be collapsed into an automatic `valid` label.

Reproduce the main table and diagnostic chart from the archived numeric release with `python scripts/plot_release.py --output runs/release-plots`. No model inference is needed for this operation.
