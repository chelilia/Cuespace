"""QMDCR — Question-Guided Multi-Level Discriminative Cue Representation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.cuespace.layers.modules import FCR, QCFMoE

D_MODEL = 512
TOP_K = 7
N_EXPERTS = 7
NHEAD = 8
DROPOUT = 0.1


def _init_mlp(module: nn.Sequential) -> None:
    for m in module:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class QMDCR(nn.Module):
    """FCR + QCF-MoE + collaborative cue S (published v6 + quest-free shared branch)."""

    def __init__(self):
        super().__init__()
        self.vfcr = FCR(D_MODEL, NHEAD)
        self.afcr = FCR(D_MODEL, NHEAD)
        self.vfcr_av = FCR(D_MODEL, NHEAD)
        self.afcr_av = FCR(D_MODEL, NHEAD)

        self.qcf_moe_audio = QCFMoE(D_MODEL, NHEAD, topK=TOP_K, n_experts=N_EXPERTS)
        self.qcf_moe_video = QCFMoE(D_MODEL, NHEAD, topK=TOP_K, n_experts=N_EXPERTS)
        self.qcf_moe_av_audio = QCFMoE(D_MODEL, NHEAD, topK=TOP_K, n_experts=N_EXPERTS)
        self.qcf_moe_av_video = QCFMoE(D_MODEL, NHEAD, topK=TOP_K, n_experts=N_EXPERTS)

        self.collab_fusion_av = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL * 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 4, D_MODEL),
        )
        _init_mlp(self.collab_fusion_av)

    @staticmethod
    def _moe_sequence_output(outputs: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = outputs
        return (a + b) / 2

    def _run_fcr(
        self,
        fcr: FCR,
        *,
        patch: Tensor,
        frame: Tensor,
        modal_type: str,
    ) -> Tensor:
        return fcr(patch, frame=frame, modal_type=modal_type)

    def _quest_free_shared(
        self,
        audio: Tensor,
        video: Tensor,
        audio_patch: Optional[Tensor],
        patch: Optional[Tensor],
    ) -> Tensor:
        dummy_q = torch.zeros(audio.size(0), audio.size(-1), device=audio.device, dtype=audio.dtype)
        audio_patch_av = None
        video_patch_av = None
        if audio_patch is not None:
            audio_patch_av = self._run_fcr(
                self.afcr_av,
                patch=audio_patch,
                frame=audio,
                modal_type='audio',
            )
        if patch is not None:
            video_patch_av = self._run_fcr(
                self.vfcr_av,
                patch=patch,
                frame=video,
                modal_type='video',
            )

        video_sub = [video_patch_av, video_patch_av] if video_patch_av is not None else None
        at_out = self.qcf_moe_av_audio(dummy_q, audio, video_sub, quest_free_router=True)
        av_audio = self._moe_sequence_output(at_out) + audio

        audio_sub = [audio_patch_av, audio_patch_av] if audio_patch_av is not None else None
        vt_out = self.qcf_moe_av_video(dummy_q, video, audio_sub, quest_free_router=True)
        av_video = self._moe_sequence_output(vt_out) + video

        shared_input = torch.cat([av_audio, av_video], dim=-1)
        b, t, _ = shared_input.shape
        return self.collab_fusion_av(shared_input.reshape(b * t, -1)).reshape(b, t, -1)

    def forward(
        self,
        *,
        quest_proj: Tensor,
        audio: Tensor,
        video: Tensor,
        audio_shared: Tensor,
        video_shared: Tensor,
        patch: Optional[Tensor],
        audio_patch: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return audio-primary A, visual-primary V, collaborative S — each [B,T,D]."""
        audio_patch_out = None
        video_patch_out = None
        if audio_patch is not None:
            audio_patch_out = self._run_fcr(
                self.afcr,
                patch=audio_patch,
                frame=audio,
                modal_type='audio',
            )
        if patch is not None:
            video_patch_out = self._run_fcr(
                self.vfcr,
                patch=patch,
                frame=video,
                modal_type='video',
            )

        audio_sub = [audio_patch_out, audio_patch_out] if audio_patch_out is not None else None
        at_out = self.qcf_moe_audio(quest_proj, audio, audio_sub)
        audio_expert = self._moe_sequence_output(at_out) + audio

        video_sub = [video_patch_out, video_patch_out] if video_patch_out is not None else None
        vt_out = self.qcf_moe_video(quest_proj, video, video_sub)
        video_expert = self._moe_sequence_output(vt_out) + video

        shared_expert = self._quest_free_shared(
            audio_shared, video_shared, audio_patch, patch,
        )
        return audio_expert, video_expert, shared_expert
