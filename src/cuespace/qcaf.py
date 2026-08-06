"""QCAF — Question-Guided Cue Adaptive Fusion (CompositionalMemoryHead mode B)."""
from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from src.cuespace.layers.memory_head import CompositionalMemoryHead

D_MODEL = 512
NHEAD = 8
DROPOUT = 0.1


class QCAF(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.head = CompositionalMemoryHead(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_labels=num_labels,
            dropout=DROPOUT,
        )

    def forward(
        self,
        a_seq: Tensor,
        v_seq: Tensor,
        av_seq: Tensor,
        *,
        quest: Tensor,
        words: Optional[Tensor],
        answer_mode: str = 'open_vocab',
        mcq_forward_mode: str = 'shared_stacked',
        cand_quest_proj: Optional[Tensor] = None,
        original_batch: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if answer_mode == 'mcq' and mcq_forward_mode == 'per_option_full':
            if original_batch is None:
                raise ValueError('per_option_full requires original_batch')
            return self.head.forward_mcq_per_option_full(
                a_seq, v_seq, av_seq,
                quest=quest,
                words=words,
                original_batch=original_batch,
            )
        if answer_mode == 'mcq':
            if cand_quest_proj is None:
                raise RuntimeError('MCQ shared_stacked requires cand_quest_proj')
            return self.head.forward_mcq(
                a_seq, v_seq, av_seq,
                quest=quest,
                words=words,
                cand_quest_proj=cand_quest_proj,
            )
        return self.head(a_seq, v_seq, av_seq, quest=quest, words=words)
