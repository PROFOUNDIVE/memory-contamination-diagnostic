#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-pilot_game24_$(date +%Y%m%d_%H%M%S)}"
python -m memcontam.cli run configs/pilot_game24.yaml --run-id "$RUN_ID"
