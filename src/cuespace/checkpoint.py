"""Remap published checkpoint keys to paper-aligned module names."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor

# Old top-level prefix → new top-level prefix (sub-module param names unchanged).
CHECKPOINT_PREFIX_RENAME: Tuple[Tuple[str, str], ...] = (
    ('quest_encoder.', 'input_proj.text_encoder.'),
    ('audio_proj.', 'input_proj.audio_proj.'),
    ('video_proj.', 'input_proj.video_proj.'),
    ('patch_proj.', 'input_proj.patch_proj.'),
    ('audio_patch_proj.', 'input_proj.audio_patch_proj.'),
    ('words_proj.', 'input_proj.words_proj.'),
    ('quest_proj.', 'input_proj.quest_proj.'),
    ('quest_norm.', 'input_proj.quest_norm.'),
    ('ot_align_audio.', 'qmsfe.align_audio.'),
    ('ot_align_video.', 'qmsfe.align_video.'),
    ('crs_attn.', 'qmsfe.crs_attn.'),
    ('patch_selecter.', 'qmdcr.vfcr.'),
    ('audio_patch_selecter.', 'qmdcr.afcr.'),
    ('patch_selecter_av.', 'qmdcr.vfcr_av.'),
    ('audio_patch_selecter_av.', 'qmdcr.afcr_av.'),
    ('at_aggregator.', 'qmdcr.qcf_moe_audio.'),
    ('vt_aggregator.', 'qmdcr.qcf_moe_video.'),
    ('at_aggregator_av.', 'qmdcr.qcf_moe_av_audio.'),
    ('vt_aggregator_av.', 'qmdcr.qcf_moe_av_video.'),
    ('shared_expert_proj_av.', 'qmdcr.collab_fusion_av.'),
    ('ple_comp_mem_head.', 'qcaf.head.'),
)

# Test inference does not use these; drop before load to reduce noise.
CHECKPOINT_IGNORE_PREFIXES: Tuple[str, ...] = (
    'head.',
    'video_evidence_grounding.',
    'video_evidence_aux_head.',
    'shared_expert_proj.',
    'qmdcr.collab_fusion.',
    'qmdcr.qcf_moe_audio.gauss_pred.',
    'qmdcr.qcf_moe_video.gauss_pred.',
    'qmdcr.qcf_moe_av_audio.gauss_pred.',
    'qmdcr.qcf_moe_av_video.gauss_pred.',
    'at_aggregator.gauss_pred.',
    'vt_aggregator.gauss_pred.',
    'at_aggregator_av.gauss_pred.',
    'vt_aggregator_av.gauss_pred.',
)


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith('module.') else key


def rename_state_dict_keys(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Map legacy checkpoint keys onto CueSpace paper-named modules."""
    out: Dict[str, Tensor] = {}
    for raw_key, value in state_dict.items():
        key = _strip_module_prefix(raw_key)
        if any(key.startswith(p) for p in CHECKPOINT_IGNORE_PREFIXES):
            continue
        new_key = key
        for old_prefix, new_prefix in CHECKPOINT_PREFIX_RENAME:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        out[new_key] = value
    return out


def load_published_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and 'state_dict' in raw:
        raw = raw['state_dict']
    if not isinstance(raw, dict):
        raise TypeError(f'checkpoint must be a dict, got {type(raw)}')
    remapped = rename_state_dict_keys(raw)
    msg = model.load_state_dict(remapped, strict=False)
    return msg
