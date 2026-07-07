#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: scripts/serve_vllm_example.sh <hf-model> [served-name]}"
SERVED_NAME="${2:-local-model}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_NAME"
