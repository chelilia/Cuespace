#!/usr/bin/env bash
# Quick smoke: one forward batch per dataset profile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

run_one() {
  local ds=$1
  local extra=$2
  echo ">>> smoke $ds $extra"
  CUDA_VISIBLE_DEVICES="${GPU:-4}" "$PYTHON" -c "
import torch
from box import Box
from configs.test_profiles import build_config
from src.eval.loader import get_model, get_dloaders
from src.eval.batching import get_items
from src.runtime.cli import seed_everything
extra = $extra
cfg = Box(build_config('$ds', **extra))
cfg.data.num_workers = 0
seed_everything(cfg.seed)
model = get_model(cfg, torch.device('cuda:0'))
loader = get_dloaders(cfg)['test']
model.eval()
with torch.no_grad():
    for batch in loader:
        if batch is None:
            continue
        out = model(get_items(batch, torch.device('cuda:0')))
        print('OK', '$ds', tuple(out['out'].shape))
        break
"
}

run_one mavqa '{}'
run_one mavqa_r '{}'
run_one mavqa_v2 '{"v2_split":"balance"}'
run_one valor32k '{}'
run_one avqa '{"mcq":True, "batch_size":8}'
echo "All smoke tests passed."
