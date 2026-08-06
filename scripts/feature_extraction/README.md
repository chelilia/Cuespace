# Offline Feature Extraction for CueSpace

CueSpace inference expects **precomputed multi-level inputs** (paper Sec. *Multi-Level Input Representation*). This folder documents how to build them from raw MP4 videos.

## What gets extracted

| Paper role | Output directory | Shape per video | Used by |
|------------|------------------|-----------------|---------|
| Visual frame stream **F_v** | `frame_ViT-L14@336px/` | `[T, 768]` | QMSFE / QMDCR |
| Visual fine-grained tokens **P_v** | `visual_tome14/` | `[T, 14, 1024]` | FCR (VFCR) |
| Audio frame stream **F_a** | `ast/audio_ast_cls/` | `[T, 768]` | QMSFE / QMDCR |
| Audio fine-grained tokens **P_a** | `ast/audio_ast_patch_last_pooled/` | `[T, 12, 768]` | FCR (AFCR) |

- **T** = number of temporal segments (1 fps; default max **60** for MUSIC-AVQA-style benchmarks).
- For Valor32k / AVQA, pass `--use-all-frames` on the visual stage so **T** matches each video length.
- **Question text** is encoded online at test time (`quest_feat=None`); no offline question features are required.

## Prerequisites

```bash
# System
ffmpeg

# Python (same env as CueSpace test)
pip install -r requirements.txt

# Weights (local ckpt/ directory or CKPT_DIR)
ckpt/ViT-L-14-336px.pt
ckpt/ast/                       # HuggingFace AST (see below)
```

Download AST once:

```python
from transformers import ASTModel, AutoFeatureExtractor

path = "./ckpt/ast"
ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").save_pretrained(path)
AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").save_pretrained(path)
```

Visual and audio extraction use modules under `src/cuespace/layers/` (CLIP, ToMe, AST via transformers). Weights go in `ckpt/` (see below).

## Quick start

From the CueSpace root:

```bash
# Full pipeline: mp4 → frames → visual + audio features
bash scripts/extract_features.sh all \
  --video-dir /path/to/videos \
  --output ./data/feats/my_benchmark \
  --gpu 0 --gpu1 1

# Smoke test (first 5 videos)
bash scripts/extract_features.sh all \
  --video-dir /path/to/videos \
  --output ./data/feats/smoke \
  --gpu 0 --gpu1 1 \
  --limit 5
```

The visual stage runs two workers by default (`--gpu` and `--gpu1`). Pass `--single-gpu` to use one device only.

### Stage-by-stage

```bash
# 1) JPEG frames only
bash scripts/extract_features.sh frames \
  --video-dir /path/to/videos \
  --output ./data/feats/my_benchmark

# 2) Visual features (requires frames under output/frames or --frames)
bash scripts/extract_features.sh visual \
  --frames ./data/feats/my_benchmark/frames \
  --output ./data/feats/my_benchmark \
  --gpu 0 --gpu1 1

# 3) Audio features (reads MP4 directly)
bash scripts/extract_features.sh audio \
  --video-dir /path/to/videos \
  --output ./data/feats/my_benchmark \
  --gpu 0
```

### Variable-length videos (Valor32k / AVQA)

```bash
bash scripts/extract_features.sh visual \
  --frames ./data/raw_frames \
  --output ./data/feats/valor32k_avqa \
  --gpu 0 --gpu1 1 \
  --use-all-frames
```

## Output layout

```
<output>/
├── frames/                          # optional intermediate JPEGs
├── frame_ViT-L14@336px/
│   └── {video_id}.npy               # [T, 768]
├── visual_tome14/
│   └── {video_id}.npy               # [T, 14, 1024]
├── ast/
│   ├── audio_ast_cls/
│   │   └── {video_id}.npy           # [T, 768]
│   └── audio_ast_patch_last_pooled/
│       └── {video_id}.npy           # [T, 12, 768]
└── logs/
```

**Indexing rule:** `{video_id}.npy` basename must match the `video_id` field in the annotation JSON.

## Wire into test config

Edit `configs/test_profiles.py` (or your profile’s `data` block):

```python
video_feat="./feats/my_benchmark/frame_ViT-L14@336px",
patch_feat="./feats/my_benchmark/visual_tome14",
audio_feat="./feats/my_benchmark/ast/audio_ast_cls",
audio_patch_feat="./feats/my_benchmark/ast/audio_ast_patch_last_pooled",
quest_feat=None,   # online tokenization at test time
```

Paths are relative to the repo root (`data/` is a local directory — copy or symlink annotations and features there).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CKPT_DIR` | `./ckpt` | CLIP + AST weights |
| `PYTHON` | `python3` | Python interpreter |
| `CLIP_MODEL_PATH` | — | Optional override for ViT-L/14@336px weights |

## Files in this directory

| File | Role |
|------|------|
| `extract_frames.py` | MP4 → JPEG (1 fps, max 60) |
| `extract_visual.py` | Visual frame + patch features |
| `extract_audio.py` | AST audio frame + pooled patch features |
| `pool_ast_patch.py` | AST patch frequency pooling |
| `build_video_list.py` | MP4 path list for audio extractor |
| `../extract_features.sh` | Main entry (frames / visual / audio / all) |
