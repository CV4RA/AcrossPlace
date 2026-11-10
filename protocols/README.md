# Protocol records

`identity.json` pins P0/P1 formal identities and distinguishes initial image selection from final eligibility. `P0/manifest.json` is the original formal manifest; its embedded paths are historical provenance, not paths to rewrite. `P0/place_assignments.json` contains the exact 56,726-place assignment.

Generate P1 with `place_retrieval.study.protocol.prepare` from the original P0 formal files. This deterministic city allocation depends only on metadata, not model scores. P1 has 39,706/8,511/8,509 training/validation/test places. The generated manifest must match the P1 fingerprint in `identity.json`; stop on a mismatch. This repository does not include the bulk `valid.jsonl` annotation file or imagery archive.
