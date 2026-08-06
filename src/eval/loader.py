"""Model and dataloader construction for test."""
from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.cuespace.checkpoint import load_published_weights
from src.cuespace.model import CueSpace
from src.data.dataset import AVQA_dataset
from src.eval.batching import get_items  # noqa: F401 — re-export
from src.runtime.logging import get_logger


def collate_fn_filter_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    from torch.utils.data._utils.collate import default_collate

    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    valid_indices = set(range(len(batch)))
    for key in all_keys:
        values = [batch[i].get(key) for i in valid_indices]
        if any(v is None for v in values) and any(v is not None for v in values):
            valid_indices = {i for i in valid_indices if batch[i].get(key) is not None}
    valid_indices = sorted(valid_indices)
    if len(valid_indices) == 0:
        return None

    result = {}
    for key in all_keys:
        values = [batch[i].get(key) for i in valid_indices]
        if all(v is None for v in values):
            result[key] = None
        else:
            result[key] = default_collate(values)
    return result


def get_model(cfg: dict, device: torch.device) -> nn.Module:
    logger = get_logger()
    md = dict(cfg.hyper_params.model)
    model = CueSpace(**md)
    model = model.to(device)

    if cfg.weight:
        msg = load_published_weights(model, cfg.weight, device)
        logger.info('Missing keys: %s', msg.missing_keys)
        logger.info('Unexpected keys: %s', msg.unexpected_keys)
        logger.info("=> loaded successfully '%s'", cfg.weight)

    if dist.is_initialized():
        find_unused = os.environ.get('DDP_FIND_UNUSED_PARAMETERS', 'True').lower() in ('1', 'true', 'yes')
        model = DDP(model, device_ids=[device.index], find_unused_parameters=find_unused)
    elif device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


def get_dataloader(cfg: dict) -> DataLoader:
    dataset = AVQA_dataset(cfg, mode=cfg.mode)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
        b_size = cfg.data.eval_batch_size // dist.get_world_size()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        sampler = None
        b_size = cfg.data.eval_batch_size
        seed_worker = None

    return DataLoader(
        dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn_filter_none,
    )


def get_dloaders(cfg: dict) -> Dict[str, DataLoader]:
    return {cfg.mode: get_dataloader(cfg)}
