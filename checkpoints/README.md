# CueSpace published checkpoints

Place benchmark weights in this directory (copy, download, or **local symlink**). Paths are referenced as `./checkpoints/<name>.pt` from `configs/test_profiles.py`.

## Download from Hugging Face

```bash
pip install -U huggingface_hub
hf download chelili/CueSpace --local-dir ./checkpoints
```

## Files

| File | Dataset | Test accuracy (reported) |
|------|---------|--------------------------|
| `mavqa.pt` | MUSIC-AVQA | **79.22%** (7232/9129) |
| `mavqa_r.pt` | MUSIC-AVQA-R | same weights as `mavqa.pt` |
| `mavqa_v2_balance.pt` | MAVQA-v2 balance | **78.49%** / 9116 |
| `mavqa_v2_bias.pt` | MAVQA-v2 bias | **78.59%** / 9116 |
| `valor32k_mcq.pt` | Valor32k-AVQA MCQ | **63.03%** / 26088 |
| `avqa_mcq.pt` | AVQA MCQ (`--mcq`) | **91.33%** (15348/16805) |

## Usage

```bash
python test.py --dataset mavqa --gpu 0
python test.py --dataset avqa --mcq --gpu 0
```

Default `--weight` values point here; override with `--weight /path/to/custom.pt` if needed.

**Note:** `.pt` files are listed in `.gitignore`. Download or symlink them locally before running tests.
