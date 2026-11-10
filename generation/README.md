# Data construction

This directory preserves the implemented Qwen3-VL generation and append-only expansion pipeline, independently of retrieval dependencies. Run these commands from `generation/` on Linux. Keep model/data outputs outside Git.

```bash
python -m venv .venv-generation
source .venv-generation/bin/activate
pip install -r requirements.txt
# Place the downloaded Qwen3-VL snapshot in ../artifacts/Qwen3-VL-4B-Instruct.
# Set the upstream image root and intended output in configs/dataset.yaml.
python scripts/prepare_gsv.py --config configs/dataset.yaml
python scripts/generate_place_texts.py --config configs/text_generation.yaml --help
python scripts/generate_place_texts.py --config configs/text_generation.yaml --filter
python scripts/build_crossmodal_dataset.py --config configs/retrieval_protocol.yaml
```

Generation extracts structured visual facts, retains image/fact provenance, generates detailed/concise/partial/relational descriptions and applies explicit rule filters. The source/target plan uses panorama groups and duplicate checks. `uncertain` records remain available for controlled diagnosis; only rule-valid records enter the principal formal benchmark.

For the archived high-throughput vLLM path, create a separate environment using `requirements-vllm.txt`, start `bash scripts/start_vllm_a800.sh`, then run `python scripts/run_vllm_pipeline.py --config configs/vllm_15k.yaml`. Model snapshots and vLLM/CUDA compatibility must be prepared independently. The loopback endpoint has no secret/API token. The included profile reflects the archived A800 settings, not a throughput guarantee on other hardware. No new GPU benchmark is launched by installation.

`scripts/prepare_formal.py --config <yaml>` calls the immutable formalizer. Its YAML requires `annotations`, `output`, `protocol: B` and optional duplicate/spatial-audit settings. For exact main-benchmark replication, use the released frozen formal files and verify their identities, rather than regenerating stochastic VLM output and expecting byte equality.

`prepare_gsv_extension.py` and `run_gsv_extension.py` preserve the original append-only 15k-to-full migration. They are historical workflow tools: the base 15k outputs must exist in the documented original `outputs/vllm_15k` and `outputs/retrieval_15k/data` layout. They freeze old artifacts, exclude held-out overlaps and keep validation/test bytes unchanged. They do not replace the public P0/P1 protocol. Use `--help` to supply the baseline index/config, image root and extension directory; do not run them on an empty dataset expecting a fresh build.

```bash
PYTHONPATH=. pytest -q tests
```

The tests exercise structured-output parsing, provenance, fact fusion, quality rules, vLLM batch handling and offline loading. Model weights are not downloaded by tests.
