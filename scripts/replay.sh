#!/usr/bin/env bash
set -euo pipefail

base_cfg="${1:?base_cfg required}"
idx="${2:?idx required}"
shift 2

python pipeline/replay.py \
  --base_cfg "${base_cfg}" \
  --idx "${idx}" \
  "$@"
