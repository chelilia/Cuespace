#!/usr/bin/env bash
# Usage: bash scripts/test.sh <dataset> <gpu> <weight> [output_dir]
# Example:
#   bash scripts/test.sh mavqa 0 ./checkpoints/mavqa.pt ./logs/mavqa

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <dataset> <gpu> <weight> [output_dir]"
  echo "  dataset: mavqa | mavqa_r | mavqa_v2 | valor32k | avqa"
  exit 1
fi

dataset=$1
gpu=$2
weight=$3
output=${4:-./result/test/${dataset}}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

extra=()
if [[ "$dataset" == "avqa" && "${MCQ:-0}" == "1" ]]; then
  extra+=(--mcq)
fi
if [[ "$dataset" == "mavqa_v2" ]]; then
  extra+=(--v2-split "${V2_SPLIT:-balance}")
fi

CUDA_VISIBLE_DEVICES="$gpu" python test.py \
  --dataset "$dataset" \
  --weight "$weight" \
  --gpu 0 \
  --output "$output" \
  "${extra[@]}"
