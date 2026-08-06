"""CLI argument parsing and runtime configuration."""
from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from box import Box

from src.runtime.cuda_preflight import (
    narrow_cuda_visible_devices_for_torchrun_worker,
    narrow_cvd_per_worker_enabled,
)

narrow_cuda_visible_devices_for_torchrun_worker()


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CueSpace AVQA test-only inference")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("mavqa", "mavqa_r", "mavqa_v2", "valor32k", "avqa"),
    )
    parser.add_argument("--mcq", action="store_true")
    parser.add_argument(
        "--v2-split",
        type=str,
        default="balance",
        choices=("balance", "bias"),
    )
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", dest="output_path", type=str, default="")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=5678)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def setting(args: argparse.Namespace) -> Tuple[Box, torch.device, int]:
    from configs.test_profiles import build_config

    raw = build_config(
        args.dataset,
        weight=args.weight or None,
        mcq=args.mcq,
        v2_split=args.v2_split,
        gpu=args.gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    conf = Box(raw)
    conf.mode = "test"
    conf.output_path = args.output_path

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        conf.cur_rank = local_rank
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed test requires CUDA")
        narrow_cvd = narrow_cvd_per_worker_enabled()
        n_vis = torch.cuda.device_count()
        cuda_idx = 0 if narrow_cvd and n_vis == 1 else local_rank
        if cuda_idx >= n_vis:
            raise RuntimeError(f"LOCAL_RANK={local_rank} >= device_count={n_vis}")
        torch.cuda.set_device(cuda_idx)
        device = torch.device("cuda", cuda_idx)
        dist.init_process_group(backend="nccl", init_method="env://")
        seed_everything(conf.seed)
        return conf, device, conf.cur_rank

    if not torch.cuda.is_available():
        conf.cur_rank = 0
        device = torch.device("cpu")
    else:
        n = torch.cuda.device_count()
        try:
            want = int(str(conf.hyper_params.gpus).split(",")[0].strip())
        except ValueError:
            want = 0
        cuda_idx = want if 0 <= want < n else 0
        torch.cuda.set_device(cuda_idx)
        conf.cur_rank = cuda_idx
        device = torch.device("cuda", cuda_idx)
    seed_everything(conf.seed)
    return conf, device, conf.cur_rank
