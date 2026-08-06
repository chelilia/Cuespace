"""QMSFE — Question-Guided Modality-Specific Feature Enhancement."""
from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.cuespace.layers.modules import AVQSelfAttn


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class _ModalityAlign(nn.Module):
    """LayerNorm + Linear projection (legacy checkpoint: align_audio.norm / align_audio.align)."""

    def __init__(self, d_model: int, align_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.align = nn.Linear(d_model, align_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.align(self.norm(x))


class QMSFE(nn.Module):
    def __init__(self, d_model: int = 512, align_dim: int = 512, nhead: int = 8):
        super().__init__()
        self.align_audio = _ModalityAlign(d_model, align_dim)
        self.align_video = _ModalityAlign(d_model, align_dim)
        self.crs_attn = AVQSelfAttn(d_model, nhead)
        _init_linear(self)

    def enhance(
        self,
        audio: Tensor,
        video: Tensor,
        words: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Align, then optionally crs_attn. Returns (audio, video, audio_pre_crs, video_pre_crs)."""
        audio = self.align_audio(audio)
        video = self.align_video(video)
        audio_pre = audio.clone()
        video_pre = video.clone()
        if words is not None:
            audio, video = self.crs_attn(audio, video, words)
        return audio, video, audio_pre, video_pre

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
        words: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        audio, video, _, _ = self.enhance(audio, video, words)
        return audio, video
