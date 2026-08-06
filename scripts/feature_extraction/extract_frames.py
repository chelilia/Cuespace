"""Extract JPEG frames from MP4 videos (1 fps, up to 60 frames).

Output layout: ``<output_dir>/<video_stem>/frame_*.jpg``
Feature filenames must match JSON ``video_id`` (directory basename).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, fps: float = 1.0, max_frames: int = 60) -> int:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(video_output_dir.glob("frame_*.jpg"))
    if len(existing) >= max_frames:
        return len(existing)

    temp_pattern = str(video_output_dir / "temp_frame_%06d.jpg")
    try:
        cmd = ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", "-y", temp_pattern]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        temp_frames = sorted(video_output_dir.glob("temp_frame_*.jpg"))
        if len(temp_frames) > max_frames:
            indices = np.round(np.linspace(0, len(temp_frames) - 1, max_frames)).astype(int)
            temp_frames = [temp_frames[i] for i in indices]
        for idx, temp_frame in enumerate(temp_frames, 1):
            temp_frame.rename(video_output_dir / f"frame_{idx:06d}.jpg")
        for temp_frame in video_output_dir.glob("temp_frame_*.jpg"):
            temp_frame.unlink()
        return len(list(video_output_dir.glob("frame_*.jpg")))
    except subprocess.TimeoutExpired:
        print(f"  timeout: {video_path.stem}")
        return 0
    except Exception as exc:
        print(f"  error {video_path.stem}: {exc}")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from MP4 for CueSpace")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory of *.mp4 files")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for frame folders")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos (smoke test)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(video_dir.glob("*.mp4"))
    if args.limit:
        mp4_files = mp4_files[: args.limit]

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print("ffmpeg is required (apt install ffmpeg / conda install ffmpeg)")
        sys.exit(1)

    ok = 0
    for video_file in tqdm(mp4_files, desc="extract frames"):
        if extract_frames(video_file, output_dir, fps=args.fps, max_frames=args.max_frames) > 0:
            ok += 1
    print(f"Done: {ok}/{len(mp4_files)} videos → {output_dir}")


if __name__ == "__main__":
    main()
