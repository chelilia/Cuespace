"""Multi-level input projection + CLIP question encoding."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.cuespace.layers.encoder import CLIP_TEncoder
from src.cuespace.layers.modules import Projection


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class InputProj(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 512,
        video_dim: int = 768,
        patch_dim: int = 1024,
        audio_dim: int = 768,
        audio_patch_dim: int = 768,
        encoder_type: str = 'ViT-L/14@336px',
        use_quest_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_quest_norm = use_quest_norm
        self.text_encoder = CLIP_TEncoder(encoder_type)
        self.text_encoder.freeze()
        self.audio_proj = Projection(audio_dim, d_model)
        self.video_proj = Projection(video_dim, d_model)
        self.patch_proj = Projection(patch_dim, d_model)
        self.audio_patch_proj = Projection(audio_patch_dim, d_model)
        self.words_proj = Projection(video_dim, d_model)
        self.quest_proj = Projection(video_dim, d_model)
        self.quest_norm = nn.LayerNorm(d_model) if use_quest_norm else None
        _init_linear(self)

    def encode_text(
        self,
        quest: Tensor,
        quest_words: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if quest.dtype in (torch.float32, torch.float64):
            q = quest.squeeze(1) if quest.dim() > 1 else quest
            return q, quest_words
        qvec, words = self.text_encoder(quest)
        return qvec.squeeze(1), words

    def project_modalities(
        self,
        *,
        audio: Tensor,
        video: Tensor,
        patch: Optional[Tensor],
        audio_patch: Optional[Tensor],
        words: Optional[Tensor],
        quest: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        audio = self.audio_proj(audio)
        video = self.video_proj(video)
        words_proj = self.words_proj(words) if words is not None else None
        quest = self.quest_proj(quest)
        if self.use_quest_norm and self.quest_norm is not None:
            quest = self.quest_norm(quest)
        patch = self.patch_proj(patch) if patch is not None else None
        audio_patch = self.audio_patch_proj(audio_patch) if audio_patch is not None else None
        return quest, words_proj, audio, video, patch, audio_patch

    def project_candidate_quests(self, cand_quest: Tensor) -> Tensor:
        """MCQ shared_stacked: [B,K,*] → [B,K,D]."""
        if cand_quest.dtype in (torch.float32, torch.float64) and cand_quest.shape[-1] != 77:
            b, k, d = cand_quest.shape
            qvec = cand_quest.reshape(b * k, d)
        else:
            b, k, seq_len = cand_quest.shape
            flat = cand_quest.reshape(b * k, seq_len)
            qvec, _ = self.text_encoder(flat)
            if qvec.dim() > 2:
                qvec = qvec.squeeze(1)
        qproj = self.quest_proj(qvec)
        if self.use_quest_norm and self.quest_norm is not None:
            qproj = self.quest_norm(qproj)
        return qproj.view(b, k, -1)
