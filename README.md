# CueSpace

Question-Guided Structured Cue Modeling and Adaptive Fusion for Audio-Visual Question Answering.

This repo provides inference, evaluation, and offline feature extraction for five AVQA benchmarks.

## Environment

```bash
git clone https://github.com/chelilia/Cuespace.git
cd Cuespace
pip install -r requirements.txt
```

Requires **CUDA + PyTorch 2.4**, and `ffmpeg` if you extract features from MP4.

---

## Step 1 — Download checkpoints

Published test weights are on Hugging Face: **[chelili/CueSpace](https://huggingface.co/chelili/CueSpace)**

Download all six checkpoints (~4 GB) into `./checkpoints/` (see the model card for per-file accuracy). Defaults are wired in `configs/test_profiles.py` — you can omit `--weight` when files sit there.

| File | Benchmark | `test.py --dataset` | Test accuracy |
|------|-----------|---------------------|---------------|
| `mavqa.pt` | MUSIC-AVQA | `mavqa` | **79.22%** (7232/9129) |
| `mavqa_r.pt` | MUSIC-AVQA-R | `mavqa_r` | same weights as `mavqa.pt` |
| `mavqa_v2_balance.pt` | MAVQA-v2 balance | `mavqa_v2 --v2-split balance` | **78.49%** |
| `mavqa_v2_bias.pt` | MAVQA-v2 bias | `mavqa_v2 --v2-split bias` | **78.59%** |
| `valor32k_mcq.pt` | Valor32k-AVQA MCQ | `valor32k` | **63.03%** |
| `avqa_mcq.pt` | AVQA MCQ | `avqa --mcq` | **91.33%** (15348/16805) |

---

## Step 2 — Prepare data & backbone weights

Inference reads **precomputed** visual/audio `.npy` features under `data/feats/`. To extract them from MP4, place backbone weights locally:

| Weight | Link | Local path |
|--------|------|------------|
| CLIP ViT-L/14@336px | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt) | `ckpt/ViT-L-14-336px.pt` |
| AST | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | `ckpt/ast/` |

Download each benchmark's **videos** and place **offline features** under `data/feats/` (see Step 3). **Annotations** are included in this repo under `data/annots/` (official splits only).

```
data/
├── annots/
│   ├── music_avqa/          # MUSIC-AVQA
│   ├── music_avqa_v2/       # MAVQA-v2
│   ├── valor32k_avqa/       # Valor32k-AVQA
│   └── avqa/                # AVQA
└── feats/
    ├── final/               # mavqa / mavqa_r / mavqa_v2
    ├── valor32k_avqa/
    └── avqa/
```

| Benchmark | Videos / features source | Annotations (in repo) |
|-----------|--------------------------|------------------------|
| **MUSIC-AVQA** | [gewu-lab.github.io/MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/) | `data/annots/music_avqa/` |
| **MAVQA-v2** | [MUSIC-AVQA-v2.0](https://github.com/DragonLiu1995/MUSIC-AVQA-v2.0) + MUSIC-AVQA videos | `data/annots/music_avqa_v2/` |
| **Valor32k-AVQA** | [valor32k-avqa-2](https://github.com/inesriahi/valor32k-avqa-2) | `data/annots/valor32k_avqa/` (`test.json` for eval) |
| **AVQA** | [juyil/AVQA-videos](https://huggingface.co/datasets/juyil/AVQA-videos) | `data/annots/avqa/` (`val.json` for official eval) |
| **MUSIC-AVQA-R** | same features as MUSIC-AVQA | `data/annots/music_avqa_r/` |

Annotation filenames must match `configs/test_profiles.py`. **AVQA** has no separate test split in the official release — benchmark evaluation uses **`val.json`** (16805 QA pairs). **Valor32k** uses **`test.json`** for held-out evaluation.

---

## Step 3 — Extract offline features

From the repo root. Output goes under `data/feats/<benchmark>/`.

**MUSIC-AVQA / MAVQA-v2** (60-frame cap):

```bash
bash scripts/extract_features.sh all \
  --video-dir /path/to/mavqa_mp4s \
  --output ./data/feats/final \
  --gpu 0 --gpu1 1
```

**Valor32k / AVQA** (variable length — add `--use-all-frames`):

```bash
bash scripts/extract_features.sh all \
  --video-dir /path/to/valor32k_mp4s \
  --output ./data/feats/valor32k_avqa \
  --gpu 0 --gpu1 1 --use-all-frames

bash scripts/extract_features.sh all \
  --video-dir /path/to/avqa_videos/videos \
  --output ./data/feats/avqa \
  --gpu 0 --gpu1 1 --use-all-frames
```

More options: [`scripts/feature_extraction/README.md`](scripts/feature_extraction/README.md). AVQA MCQ also needs option text features at `data/feats/avqa/mcq_text_clip/`.

---

## Step 4 — Run inference

```bash
python test.py --dataset mavqa --gpu 0
python test.py --dataset mavqa_r --gpu 0
python test.py --dataset mavqa_v2 --v2-split balance --gpu 0
python test.py --dataset mavqa_v2 --v2-split bias --gpu 0
python test.py --dataset valor32k --gpu 0
python test.py --dataset avqa --mcq --gpu 0
```

```bash
# optional
python test.py --dataset mavqa --gpu 0 --batch-size 16
bash scripts/smoke_test.sh
torchrun --nproc_per_node=4 test.py --dataset valor32k --distributed
```

---

## More

- [`checkpoints/README.md`](checkpoints/README.md) · [`docs/architecture.md`](docs/architecture.md)
