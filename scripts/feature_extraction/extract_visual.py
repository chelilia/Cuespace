"""Visual multi-level feature extraction: frame stream + fine-grained tokens."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cuespace.layers.clip.clip import load  # noqa: E402
from src.cuespace.layers.tome.patch import clip as clip_tome  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_visual_encoder(
    ckpt_dir: str | None = None,
    layers: int = 23,
    r_per_layer: int = 25,
):
    ckpt_dir = ckpt_dir or os.environ.get("CKPT_DIR", str(ROOT / "ckpt"))
    weight_path = os.path.join(ckpt_dir, "ViT-L-14-336px.pt")
    if not os.path.isfile(weight_path):
        weight_path = os.environ.get("CLIP_MODEL_PATH", weight_path)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"CLIP weights not found: {weight_path}")

    print(f"Loading visual encoder: {weight_path}")
    model, preprocess = load(weight_path, q_aware_N=-1, device=device)
    model = model[0] if isinstance(model, tuple) else model
    visual = model.visual

    clip_tome.apply_patch(visual)
    clip_tome.configure_tome_r(visual, layers=layers, r_per_layer=r_per_layer)

    dummy = torch.randn(1, 3, visual.input_resolution, visual.input_resolution, device=device)
    with torch.no_grad():
        frame, patch = visual(dummy)
    if patch.shape[1] != 14 or patch.shape[2] != 1024:
        raise RuntimeError(f"Unexpected patch shape: {tuple(patch.shape)}")
    print(f"Visual encoder ready: frame {tuple(frame.shape)}, patch {tuple(patch.shape)}")
    return model, preprocess, visual


def extract_visual_features(
    frames_base_dir: str,
    frame_output_dir: str,
    patch_output_dir: str,
    video_list_file: str | None = None,
    batch_size: int = 128,
    num_frames: int = 60,
    use_all_frames: bool = False,
    gpu_id: str = "0",
    layers: int = 23,
    r_per_layer: int = 25,
    ckpt_dir: str | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _, preprocess, visual = load_visual_encoder(ckpt_dir, layers, r_per_layer)
    visual.eval()

    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs(patch_output_dir, exist_ok=True)

    if video_list_file and os.path.exists(video_list_file):
        with open(video_list_file) as f:
            video_frame_dirs = [ln.strip() for ln in f if ln.strip()]
        print(f"[GPU {gpu_id}] listed videos: {len(video_frame_dirs)}")
    else:
        video_frame_dirs = []
        for subdir in sorted(os.listdir(frames_base_dir)):
            subdir_path = os.path.join(frames_base_dir, subdir)
            if os.path.isdir(subdir_path) and (
                glob.glob(os.path.join(subdir_path, "*.jpg"))
                or glob.glob(os.path.join(subdir_path, "*.png"))
            ):
                video_frame_dirs.append(subdir_path)
        print(f"[GPU {gpu_id}] scanned videos: {len(video_frame_dirs)}")

    all_frames: list[str] = []
    frame_to_video: list[int] = []
    video_info: list[tuple] = []
    skipped = 0

    for video_idx, video_frame_dir in enumerate(video_frame_dirs):
        video_name = os.path.basename(video_frame_dir.rstrip("/"))
        frame_file = os.path.join(frame_output_dir, video_name + ".npy")
        patch_file = os.path.join(patch_output_dir, video_name + ".npy")
        if os.path.exists(frame_file) and os.path.exists(patch_file):
            skipped += 1
            continue
        if not os.path.isdir(video_frame_dir):
            continue

        img_list = sorted(glob.glob(os.path.join(video_frame_dir, "*.jpg")))
        if not img_list:
            img_list = sorted(glob.glob(os.path.join(video_frame_dir, "*.png")))
        if not img_list:
            continue

        if not use_all_frames and len(img_list) >= num_frames:
            idx = np.round(np.linspace(0, len(img_list) - 1, num_frames)).astype(int)
            img_list = [img_list[i] for i in idx]

        start = len(all_frames)
        n = len(img_list)
        video_info.append((video_name, video_idx, start, n, frame_file, patch_file))
        all_frames.extend(img_list)
        frame_to_video.extend([video_idx] * n)

    total = len(all_frames)
    print(f"[GPU {gpu_id}] skipped done: {skipped}, pending videos: {len(video_info)}, frames: {total}")
    if total == 0:
        return

    info_map = {vi[1]: vi for vi in video_info}
    frame_dict: dict[int, dict[int, np.ndarray]] = {vi[1]: {} for vi in video_info}
    patch_dict: dict[int, dict[int, np.ndarray]] = {vi[1]: {} for vi in video_info}
    frame_count: dict[int, int] = {vi[1]: 0 for vi in video_info}

    def flush_done():
        for vid, (name, _, _, n, ff, pf) in info_map.items():
            if frame_count[vid] < n or not frame_dict[vid]:
                continue
            order = sorted(frame_dict[vid])
            frames = np.stack([frame_dict[vid][i] for i in order], axis=0)
            patches = np.stack([patch_dict[vid][i] for i in order], axis=0)
            if not use_all_frames and frames.shape[0] > num_frames:
                frames = frames[:num_frames]
                patches = patches[:num_frames]
            np.save(ff, frames.astype(np.float32))
            np.save(pf, patches.astype(np.float32))
            print(f"[GPU {gpu_id}] saved {name} frame {frames.shape} patch {patches.shape}")
            frame_dict[vid].clear()
            patch_dict[vid].clear()

    bs = batch_size
    pos = 0
    while pos < total:
        end = min(pos + bs, total)
        paths = all_frames[pos:end]
        tensors = []
        indices = []
        for i, p in enumerate(paths):
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                tensors.append(img)
                indices.append(pos + i)
            except Exception as e:
                print(f"[GPU {gpu_id}] skip {p}: {e}")

        if not tensors:
            pos = end
            continue

        batch_t = torch.stack(tensors).to(device)
        try:
            with torch.no_grad():
                frame_feat, patch_feat = visual(batch_t)
            frame_cpu = frame_feat.cpu().numpy()
            patch_cpu = patch_feat.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 8:
                raise
            bs = max(bs // 2, 8)
            print(f"[GPU {gpu_id}] OOM, batch_size -> {bs}")
            continue

        del batch_t, frame_feat, patch_feat
        torch.cuda.empty_cache()

        for j, gidx in enumerate(indices):
            vid = frame_to_video[gidx]
            if vid not in info_map:
                continue
            _, _, start, n, _, _ = info_map[vid]
            rel = gidx - start
            if 0 <= rel < n:
                frame_dict[vid][rel] = frame_cpu[j]
                patch_dict[vid][rel] = patch_cpu[j]
                frame_count[vid] += 1

        flush_done()
        pos = end

    flush_done()
    print(f"[GPU {gpu_id}] visual extraction finished")


def main():
    parser = argparse.ArgumentParser(description="Extract visual frame + patch features")
    parser.add_argument("--frames_base_dir", required=True)
    parser.add_argument("--frame_output_dir", required=True)
    parser.add_argument("--patch_output_dir", required=True)
    parser.add_argument("--video_list_file", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--use_all_frames", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--layers", type=int, default=23)
    parser.add_argument("--r_per_layer", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    args = parser.parse_args()

    extract_visual_features(
        frames_base_dir=args.frames_base_dir,
        frame_output_dir=args.frame_output_dir,
        patch_output_dir=args.patch_output_dir,
        video_list_file=args.video_list_file,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        use_all_frames=args.use_all_frames,
        gpu_id=args.gpu_id,
        layers=args.layers,
        r_per_layer=args.r_per_layer,
        ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
