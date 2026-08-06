"""Logging helpers for test runs."""
from __future__ import annotations

import json
import logging
import platform
import warnings
from logging import getLogger
from pathlib import Path

import distro
import torch
import torch.distributed as dist


def get_logger():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            getLogger("AVQA").setLevel(logging.INFO)
            return getLogger("AVQA")
        getLogger("AVQA").setLevel(logging.WARNING)
        return getLogger("AVQA")
    getLogger("AVQA").setLevel(logging.INFO)
    return getLogger("AVQA")


def set_logger(cfg) -> None:
    warnings.filterwarnings("ignore")
    if cfg.output_path:
        logging_path = Path(cfg.output_path)
        if not logging_path.exists():
            logging_path.mkdir(parents=True, exist_ok=True)
        logging_path = logging_path / (str(Path(cfg.weight).stem) + "_result.txt")
    else:
        logging_path = Path(str(cfg.weight).replace(".pt", "_result.txt"))

    if dist.is_available() and (
        (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()
    ):
        logger = logging.getLogger(name="AVQA")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        file_handler = logging.FileHandler(logging_path, mode="w")
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def logging_config(config) -> None:
    os_version = f"{distro.name()} {distro.version()}"
    kernel_version = platform.platform()
    logger = get_logger()
    logger.info("\n-------------- config --------------")
    logger.info(json.dumps(config, indent=4, default=str))
    logger.info("\n-------------- environment --------------")
    logger.info(f"OS version: {os_version}")
    logger.info(f"Kernel version: {kernel_version}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"cuda version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        cid = torch.cuda.current_device()
        name = torch.cuda.get_device_name(cid)
        mem = round(torch.cuda.get_device_properties(cid).total_memory / 1024**3, 1)
        logger.info(f"gpu device: {cid} {name} - {mem}GB")
