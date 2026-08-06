"""Write one video path per line for AST audio extraction."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    root = Path(args.video_dir)
    pattern = "**/*.mp4" if args.recursive else "*.mp4"
    paths = sorted(root.glob(pattern))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p.resolve()) for p in paths) + ("\n" if paths else ""), encoding="utf-8")
    print(f"Wrote {len(paths)} paths → {out}")


if __name__ == "__main__":
    main()
