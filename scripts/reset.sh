base_cfg="${1:?base_cfg required}"
shift 1

uv run pipeline/reset.py \
  --base_cfg "${base_cfg}" \
  "$@"