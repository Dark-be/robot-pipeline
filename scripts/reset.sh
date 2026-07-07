#!/usr/bin/env bash
set -euo pipefail

base_cfg="${1:?base_cfg required}"

python3 pipeline/reset.py --base_cfg "${base_cfg}"
