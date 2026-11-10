#!/usr/bin/env bash
set -euo pipefail
# Run from generation/. Override for the available GPU rather than benchmarking
# the user machine automatically. Values reflect the archived A800 profile.
MODEL_PATH="${MODEL_PATH:-../artifacts/Qwen3-VL-4B-Instruct}"
exec vllm serve "$MODEL_PATH" --served-model-name qwen3-vl-4b \
  --host 127.0.0.1 --port "${PORT:-8000}" --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-16384}"
