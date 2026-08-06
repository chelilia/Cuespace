"""Runtime overrides for published CueSpace test checkpoints."""
from __future__ import annotations

from typing import Any, Dict

# Structural hyperparameters are fixed in sub-modules (input_proj, qmdcr, qcaf).
# Only dataset-dependent fields are passed from build_config.
_OVERRIDABLE = frozenset({
    'num_labels',
    'answer_mode',
    'mcq_forward_mode',
})

# Kept for apply_test_defaults merge with InputProj dims (from checkpoint era).
_ARCH_DIMS = dict(
    video_dim=768,
    patch_dim=1024,
    audio_dim=768,
    audio_patch_dim=768,
    encoder_type='ViT-L/14@336px',
    use_quest_norm=True,
    align_dim=512,
)


def apply_test_defaults(overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(_ARCH_DIMS)
    cfg.setdefault('num_labels', 42)
    cfg.setdefault('answer_mode', 'open_vocab')
    cfg.setdefault('mcq_forward_mode', 'shared_stacked')
    for key in _OVERRIDABLE:
        if key in overrides and overrides[key] is not None:
            cfg[key] = overrides[key]
    return cfg
