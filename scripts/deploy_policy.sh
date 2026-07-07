base_cfg="${1:?base_cfg required}"
shift 1

python3 pipeline/deploy_act_policy.py \
  --base_cfg "${base_cfg}" \
  "$@"