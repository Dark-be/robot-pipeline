#!/usr/bin/bash
base_cfg="${1:?base_cfg required}"
st_idx="${2:?st_idx required}"
shift 2

uv run pipeline/collect.py \
  --base_cfg "${base_cfg}" \
  --st_idx "${st_idx}" \
  "$@"
