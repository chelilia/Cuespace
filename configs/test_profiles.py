"""Test-only profiles: dataset paths + inference knobs (batch size, GPU, weight).

Model architecture is hardcoded in src/cuespace/defaults.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

_ROOT = Path(__file__).resolve().parents[1]

DatasetName = Literal["mavqa", "mavqa_r", "mavqa_v2", "valor32k", "avqa"]
V2Split = Literal["balance", "bias"]

_DATA_ROOT = "./data"
_MAVQA_FEATS = "./feats/final"
_VALOR_FEATS = "./feats/valor32k_avqa"
_AVQA_FEATS = "./feats/avqa"

DEFAULT_WEIGHTS: Dict[str, str] = {
    "mavqa": "./checkpoints/mavqa.pt",
    "mavqa_r": "./checkpoints/mavqa_r.pt",
    "mavqa_v2_balance": "./checkpoints/mavqa_v2_balance.pt",
    "mavqa_v2_bias": "./checkpoints/mavqa_v2_bias.pt",
    "valor32k": "./checkpoints/valor32k_mcq.pt",
    "avqa_mcq": "./checkpoints/avqa_mcq.pt",
}

# Per-benchmark: annotation paths, feature dirs, label space, optional taxonomy.
_PROFILES: Dict[str, Dict[str, Any]] = {
    "mavqa": {
        "test_annots": ["./annots/music_avqa/music_avqa_test.json"],
        "ans_quelen": "./annots/music_avqa/answer2idx.json",
        "video_feat": f"{_MAVQA_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_MAVQA_FEATS}/visual_tome14",
        "audio_feat": f"{_MAVQA_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_MAVQA_FEATS}/ast/audio_ast_patch_last_pooled",
        "target_frames": 60,
        "num_labels": 42,
        "answer_mode": "open_vocab",
        "weight_key": "mavqa",
    },
    "mavqa_r": {
        "test_annots": [
            "./annots/music_avqa_r/avqa-test-head.json",
            "./annots/music_avqa_r/avqa-test-headtail.json",
            "./annots/music_avqa_r/avqa-test-tail.json",
        ],
        "ans_quelen": "./annots/music_avqa/answer2idx.json",
        "video_feat": f"{_MAVQA_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_MAVQA_FEATS}/visual_tome14",
        "audio_feat": f"{_MAVQA_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_MAVQA_FEATS}/ast/audio_ast_patch_last_pooled",
        "target_frames": 60,
        "num_labels": 42,
        "answer_mode": "open_vocab",
        "weight_key": "mavqa_r",
    },
    "mavqa_v2_balance": {
        "test_annots": ["./annots/music_avqa_v2/test_balance.json"],
        "ans_quelen": "./annots/music_avqa_v2/answer2idx.json",
        "video_feat": f"{_MAVQA_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_MAVQA_FEATS}/visual_tome14",
        "audio_feat": f"{_MAVQA_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_MAVQA_FEATS}/ast/audio_ast_patch_last_pooled",
        "target_frames": 60,
        "num_labels": 42,
        "answer_mode": "open_vocab",
        "weight_key": "mavqa_v2_balance",
    },
    "mavqa_v2_bias": {
        "test_annots": ["./annots/music_avqa_v2/test_bias.json"],
        "ans_quelen": "./annots/music_avqa_v2/answer2idx.json",
        "video_feat": f"{_MAVQA_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_MAVQA_FEATS}/visual_tome14",
        "audio_feat": f"{_MAVQA_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_MAVQA_FEATS}/ast/audio_ast_patch_last_pooled",
        "target_frames": 60,
        "num_labels": 42,
        "answer_mode": "open_vocab",
        "weight_key": "mavqa_v2_bias",
    },
    "valor32k": {
        "test_annots": ["./annots/valor32k_avqa/test.json"],
        "ans_quelen": "./annots/valor32k_avqa/answer2idx.json",
        "video_feat": f"{_VALOR_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_VALOR_FEATS}/visual_tome14",
        "audio_feat": f"{_VALOR_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_VALOR_FEATS}/ast/audio_ast_patch_last_pooled",
        "target_frames": 12,
        "qtype_taxonomy": "valor32k",
        "num_labels": 4,
        "answer_mode": "mcq",
        "mcq_forward_mode": "per_option_full",
        "weight_key": "valor32k",
    },
    "avqa_mcq": {
        "test_annots": ["./annots/avqa/val.json"],
        "ans_quelen": "./annots/avqa/answer2idx.json",
        "video_feat": f"{_AVQA_FEATS}/frame_ViT-L14@336px",
        "patch_feat": f"{_AVQA_FEATS}/visual_tome14",
        "audio_feat": f"{_AVQA_FEATS}/ast/audio_ast_cls",
        "audio_patch_feat": f"{_AVQA_FEATS}/ast/audio_ast_patch_last_pooled",
        "mcq_text_feat": "./feats/avqa/mcq_text_clip",
        "target_frames": 12,
        "qtype_taxonomy": "avqa",
        "num_labels": 4,
        "answer_mode": "mcq",
        "mcq_forward_mode": "shared_stacked",
        "weight_key": "avqa_mcq",
    },
}


def build_config(
    dataset: DatasetName,
    *,
    weight: Optional[str] = None,
    mcq: bool = False,
    v2_split: V2Split = "balance",
    gpu: str = "0",
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = "./result/test",
    seed: int = 5678,
) -> Dict[str, Any]:
    if dataset == "mavqa_v2":
        profile_key = f"mavqa_v2_{v2_split}"
    elif dataset == "avqa":
        if not mcq:
            raise ValueError("avqa is MCQ-only in this release; pass --mcq")
        profile_key = "avqa_mcq"
    elif dataset == "valor32k":
        if mcq:
            raise ValueError("valor32k is MCQ-only; do not pass --mcq")
        profile_key = "valor32k"
    else:
        profile_key = dataset

    if profile_key not in _PROFILES:
        raise ValueError(f"unknown dataset: {dataset}")

    p = _PROFILES[profile_key]
    test_annots = list(p["test_annots"])
    weight_key = p["weight_key"]
    resolved_weight = weight or DEFAULT_WEIGHTS.get(weight_key, "")
    if resolved_weight and not Path(resolved_weight).is_absolute():
        resolved_weight = str((_ROOT / resolved_weight).resolve())

    data: Dict[str, Any] = {
        "root": _DATA_ROOT,
        "test_annot": test_annots[0],
        "test_annots": test_annots,
        "ans_quelen": p["ans_quelen"],
        "video_feat": p["video_feat"],
        "patch_feat": p["patch_feat"],
        "audio_feat": p["audio_feat"],
        "audio_patch_feat": p["audio_patch_feat"],
        "target_frames": p["target_frames"],
        "eval_batch_size": batch_size,
        "num_workers": num_workers,
        "answer_mode": p["answer_mode"],
        "mcq_forward_mode": p.get("mcq_forward_mode", "shared_stacked"),
    }
    if "qtype_taxonomy" in p:
        data["qtype_taxonomy"] = p["qtype_taxonomy"]
    if "mcq_text_feat" in p:
        data["mcq_text_feat"] = p["mcq_text_feat"]

    model_overrides = {
        "num_labels": p["num_labels"],
        "answer_mode": p["answer_mode"],
        "mcq_forward_mode": p.get("mcq_forward_mode", "shared_stacked"),
    }

    return {
        "seed": seed,
        "mode": "test",
        "output_dir": output_dir,
        "weight": resolved_weight,
        "data": data,
        "hyper_params": {
            "gpus": gpu,
            "model": model_overrides,
        },
    }
