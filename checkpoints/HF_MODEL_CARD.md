---
license: mit
tags:
- audio-visual-question-answering
- avqa
- music-avqa
library_name: cuespace
---

# CueSpace Checkpoints

Published test weights for [CueSpace](https://github.com/chelilia/Cuespace) — Question-Guided Structured Cue Modeling and Adaptive Fusion for Audio-Visual Question Answering.


## Files

| File | Dataset | `--dataset` | Reported test accuracy |
|------|---------|-------------|------------------------|
| `mavqa.pt` | MUSIC-AVQA | `mavqa` | **79.22%** (7232/9129) |
| `mavqa_r.pt` | MUSIC-AVQA-R | `mavqa_r` | same weights as `mavqa.pt` |
| `mavqa_v2_balance.pt` | MAVQA-v2 balance | `mavqa_v2 --v2-split balance` | **78.49%** |
| `mavqa_v2_bias.pt` | MAVQA-v2 bias | `mavqa_v2 --v2-split bias` | **78.59%** |
| `valor32k_mcq.pt` | Valor32k-AVQA MCQ | `valor32k` | **63.03%** |
| `avqa_mcq.pt` | AVQA MCQ | `avqa --mcq` | **91.33%** (15348/16805) |

## Download

```bash
pip install -U huggingface_hub
hf download chelili/CueSpace --local-dir ./checkpoints
```

## Usage (CueSpace repo)

```bash
git clone https://github.com/chelilia/Cuespace.git
cd CueSpace
pip install -r requirements.txt
# prepare data/ + ckpt/ (CLIP/AST) locally — see README

python test.py --dataset mavqa --weight ./checkpoints/mavqa.pt --gpu 0
python test.py --dataset valor32k --weight ./checkpoints/valor32k_mcq.pt --gpu 0
python test.py --dataset avqa --mcq --weight ./checkpoints/avqa_mcq.pt --gpu 0
```

## Citation

```bibtex
@article{cuespace2026,
  title={CueSpace: Question-Guided Structured Cue Modeling and Adaptive Fusion for Audio-Visual Question Answering},
  author={...},
  year={2026}
}
```
