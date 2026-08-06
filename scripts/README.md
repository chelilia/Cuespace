# Scripts

Entry points for **offline feature extraction** and **quick validation**. Full benchmark evaluation uses root-level `test.py`.

| Script | Purpose |
|--------|---------|
| [`extract_features.sh`](extract_features.sh) | MP4 → frames → visual + audio multi-level features |
| [`smoke_test.sh`](smoke_test.sh) | One forward batch per test profile (needs GPU + checkpoints) |
| [`test.sh`](test.sh) | Thin shell wrapper around `test.py` |

Feature extraction helpers live in [`feature_extraction/`](feature_extraction/):

| File | Role |
|------|------|
| `extract_frames.py` | MP4 → JPEG frames (1 fps) |
| `extract_visual.py` | Frame stream + fine-grained visual tokens |
| `extract_audio.py` | AST frame + pooled patch tokens |
| `pool_ast_patch.py` | Frequency pooling for audio patches |
| `build_video_list.py` | MP4 path list for audio stage |

See [`feature_extraction/README.md`](feature_extraction/README.md) for weights, output layout, and wiring into `configs/test_profiles.py`.

```bash
# Extract features (self-contained; weights under ckpt/)
bash scripts/extract_features.sh all \
  --video-dir /path/to/mp4s \
  --output ./data/feats/my_benchmark \
  --gpu 0 --gpu1 1

# Smoke test inference
GPU=4 bash scripts/smoke_test.sh
```
