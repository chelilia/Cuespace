"""
torchrun 多进程下的 GPU 可见性策略（由 NARROW_CVD_PER_WORKER 控制）。

NARROW_CVD_PER_WORKER=1（默认，防御型）：
  父环境 CUDA_VISIBLE_DEVICES=0,1,... 时，按 LOCAL_RANK 收窄为单卡；
  进程内始终 cuda:0，避免多进程争用可见设备上下文 → busy/unavailable。

NARROW_CVD_PER_WORKER=0（标准 torchrun）：
  不修改 CUDA_VISIBLE_DEVICES；各 worker 均见 0,1,...，由 setting() 按 LOCAL_RANK 选 cuda:N。

须在 import torch / torch.distributed 之前调用 narrow_cuda_visible_devices_for_torchrun_worker()。
"""
from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def narrow_cvd_per_worker_enabled() -> bool:
    return os.environ.get("NARROW_CVD_PER_WORKER", "1").lower() in _TRUTHY


def narrow_cuda_visible_devices_for_torchrun_worker() -> None:
    if not narrow_cvd_per_worker_enabled():
        return

    lr_s = os.environ.get("LOCAL_RANK", "").strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not lr_s or not vis or "," not in vis:
        return
    parts = [p.strip() for p in vis.split(",") if p.strip()]
    lr = int(lr_s)
    if lr < 0 or lr >= len(parts):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = parts[lr]
