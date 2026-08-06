#!/usr/bin/env bash
# Upload CueSpace published checkpoints to Hugging Face Hub (model repo).
#
# Prerequisites:
#   pip install -U huggingface_hub
#   hf auth login          # paste token from https://huggingface.co/settings/tokens
#
# Usage:
#   bash scripts/upload_weights_to_hf.sh YOUR_HF_USERNAME
#   bash scripts/upload_weights_to_hf.sh YOUR_HF_USERNAME --repo CueSpace-checkpoints
#   bash scripts/upload_weights_to_hf.sh YOUR_ORG/CueSpace   # org repo
#
# Optional env:
#   HF_REPO=CueSpace       default repo name
#   CREATE_REPO=1          create repo if missing (default 1)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf_user_or_org/repo> [--repo NAME]"
  echo "  Examples:"
  echo "    $0 alice"
  echo "    $0 alice --repo CueSpace"
  echo "    $0 my-org/CueSpace"
  exit 1
fi

HF_TARGET="$1"
shift
HF_REPO="${HF_REPO:-CueSpace}"
CREATE_REPO="${CREATE_REPO:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) HF_REPO="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$HF_TARGET" == */* ]]; then
  HF_REPO_ID="$HF_TARGET"
else
  HF_REPO_ID="${HF_TARGET}/${HF_REPO}"
fi

WEIGHTS=(
  mavqa.pt
  mavqa_r.pt
  mavqa_v2_balance.pt
  mavqa_v2_bias.pt
  valor32k_mcq.pt
  avqa_mcq.pt
)

STAGE="${ROOT}/.hf_upload_staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo ">>> Staging weights (resolve symlinks)..."
total=0
for f in "${WEIGHTS[@]}"; do
  src="${ROOT}/checkpoints/${f}"
  [[ -e "$src" ]] || { echo "Missing: $src"; exit 1; }
  cp -L "$src" "${STAGE}/${f}"
  sz=$(stat -c%s "${STAGE}/${f}")
  total=$((total + sz))
  echo "  $(numfmt --to=iec-i --suffix=B "$sz" 2>/dev/null || echo "${sz} bytes")  $f"
done
echo ">>> Staged total: $(numfmt --to=iec-i --suffix=B "$total" 2>/dev/null || echo "$total bytes")"

if ! hf auth whoami &>/dev/null; then
  echo ""
  echo "Not logged in. Run:"
  echo "  hf auth login"
  echo "Token: https://huggingface.co/settings/tokens (need Write)"
  exit 1
fi

echo ">>> Logged in as: $(hf auth whoami 2>/dev/null | head -1)"
echo ">>> Target repo: ${HF_REPO_ID}"

if [[ "$CREATE_REPO" == "1" ]]; then
  echo ">>> Creating model repo (ignore error if exists)..."
  hf repo create "$HF_REPO_ID" --type model --exist-ok 2>/dev/null || true
fi

# Model card
if [[ -f "${ROOT}/checkpoints/HF_MODEL_CARD.md" ]]; then
  cp "${ROOT}/checkpoints/HF_MODEL_CARD.md" "${STAGE}/README.md"
fi

echo ">>> Uploading (may take several minutes for ~4GB)..."
hf upload "$HF_REPO_ID" "$STAGE" . --repo-type model

echo ""
echo "Done: https://huggingface.co/${HF_REPO_ID}"
echo ""
echo "Users download with:"
echo "  hf download ${HF_REPO_ID} --local-dir ./checkpoints"
echo ""
echo "Or in CueSpace test.py:"
echo "  python test.py --dataset mavqa --weight hf://${HF_REPO_ID}/mavqa.pt --gpu 0"
