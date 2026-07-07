#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-full_matrix_$(date +%Y%m%d_%H%M%S)}"
python -m memcontam.cli run configs/full_matrix.yaml --run-id "$RUN_ID"
