"""AST patch frequency pooling: [T, 1212, D] -> [T, 12, D]."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_FREQ_BANDS = 12
NUM_TIME_TOKENS = 101


def pool_patch_numpy(patch: np.ndarray) -> np.ndarray:
    T, P, D = patch.shape
    assert P == NUM_FREQ_BANDS * NUM_TIME_TOKENS, f"expected P=1212, got {P}"
    x = patch.reshape(T, NUM_FREQ_BANDS, NUM_TIME_TOKENS, D)
    return x.mean(axis=2).astype(np.float32)


def is_valid_pooled(path: Path) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        return (
            arr.ndim == 3
            and arr.shape[1] == NUM_FREQ_BANDS
            and arr.shape[2] == 768
        )
    except Exception:
        return False
