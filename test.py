"""CueSpace test-only entry point."""
from __future__ import print_function

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.runtime.cuda_preflight import narrow_cuda_visible_devices_for_torchrun_worker

narrow_cuda_visible_devices_for_torchrun_worker()

import torch.distributed as dist

from src.runtime.cli import arg_parse, seed_everything, setting
from src.runtime.logging import get_logger, set_logger, logging_config
from src.eval.loader import get_model, get_dloaders
from src.eval.runner import run_test, sync_processes


def main():
    args = arg_parse()
    cfg, device, _ = setting(args)
    set_logger(cfg)
    logger = get_logger()

    logging_config(cfg)
    seed_everything(cfg.seed)

    model = get_model(cfg, device)
    d_loaders = get_dloaders(cfg)

    sync_processes()
    if isinstance(cfg.data.test_annots, (list, tuple)):
        for test_annot in cfg.data.test_annots:
            cfg.data.test_annot = test_annot
            loader = get_dloaders(cfg)["test"]
            total_samples = len(loader.dataset)
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                samples_per_gpu = (
                    len(loader.sampler)
                    if hasattr(loader, "sampler") and loader.sampler is not None
                    else total_samples
                )
                logger.info(
                    f"\n-------------- evaluating {cfg.data.test_annot} --------------"
                )
                logger.info(
                    f"total={total_samples}, gpus={world_size}, rank={rank} samples={samples_per_gpu}"
                )
            else:
                logger.info(
                    f"\n-------------- evaluating {cfg.data.test_annot} --------------"
                )
                logger.info(f"total samples: {total_samples}")
            run_test(cfg, device, loader, model)
    else:
        loader = d_loaders["test"]
        total_samples = len(loader.dataset)
        logger.info(
            f"\n-------------- evaluating {cfg.data.test_annot} --------------"
        )
        logger.info(f"total samples: {total_samples}")
        run_test(cfg, device, loader, model)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
