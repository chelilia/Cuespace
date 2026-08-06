"""Test loop and accuracy reporting."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.taxonomy import (
    get_qtype_taxonomy,
    get_qtype_taxonomy_name,
    parse_batched_qtype_pairs,
    taxonomy_gather_idx,
    taxonomy_size,
)
from src.eval.batching import get_items
from src.runtime.logging import get_logger

_LOG_INTERVAL = 100


def sync_processes():
    if dist.is_initialized():
        dist.barrier()


def _log_per_qtype_accuracy(taxonomy, correct_tensor, tot_tensor, *, prefix='Test'):
    logger = get_logger()
    for modality, sub_map in taxonomy.items():
        modality_corr = 0
        modality_tot = 0
        for qst_type in sub_map:
            gather_idx = taxonomy_gather_idx(taxonomy, modality, qst_type)
            corr = int(correct_tensor[gather_idx].item())
            tot = int(tot_tensor[gather_idx].item())
            modality_corr += corr
            modality_tot += tot
            key = f'{modality}/{qst_type}'
            if tot > 0:
                value = corr / tot * 100.0
                logger.info(f'{prefix} - {key:>28} accuracy: {value:.2f}({corr}/{tot})')
            else:
                logger.info(f'{prefix} - {key:>28} accuracy: N/A(0/0)')
        if modality_tot > 0:
            modality_acc = modality_corr / modality_tot * 100.0
            logger.info(f'{prefix} - {modality:>28} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
        else:
            logger.info(f'{prefix} - {modality:>28} accuracy: N/A(0/0)')


def run_test(cfg, device, test_loader: DataLoader, model: nn.Module) -> float:
    logger = get_logger()
    model.eval()

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(
            f"[GPU {rank}/{world_size}] DataLoader: dataset={len(test_loader.dataset)}, "
            f"batches={len(test_loader)}"
        )
    else:
        logger.info(
            f"[single-GPU] DataLoader: dataset={len(test_loader.dataset)}, "
            f"batches={len(test_loader)}"
        )

    total, correct = 0, 0
    taxonomy = get_qtype_taxonomy(cfg)
    taxonomy_name = get_qtype_taxonomy_name(cfg)
    n_qtypes = taxonomy_size(taxonomy)
    tot_tensor = torch.zeros(n_qtypes, dtype=torch.long, device=device)
    correct_tensor = torch.zeros(n_qtypes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            if sample is None:
                continue
            reshaped_data = get_items(sample, device)
            target = reshaped_data['label']
            output = model(reshaped_data)
            _, predicted = torch.max(output['out'].data, 1)

            bs = predicted.size(0)
            total += bs
            correct += (predicted == target).sum().item()

            qst_types = sample['type']
            for idx, (modal_type, qst_type) in enumerate(
                parse_batched_qtype_pairs(taxonomy_name, qst_types)
            ):
                gather_idx = taxonomy_gather_idx(taxonomy, modal_type, qst_type)
                if gather_idx is None:
                    continue
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()

            if batch_idx % _LOG_INTERVAL == 0 or batch_idx == len(test_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(test_loader) - 1}')

    sync_processes()
    if dist.is_initialized():
        correct_t = torch.tensor(correct, device=device)
        total_t = torch.tensor(total, device=device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        correct, total = correct_t.item(), total_t.item()
        for idx in range(n_qtypes):
            dist.all_reduce(tot_tensor[idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor[idx], op=dist.ReduceOp.SUM)

    acc = correct / total * 100.0 if total else 0.0
    _log_per_qtype_accuracy(taxonomy, correct_tensor, tot_tensor, prefix='Test')
    logger.info(f'Test {"Total avg":>28} accuracy: {acc:.2f}({correct}/{total})')
    return acc
