"""QCAF readout: compositional memory head (mode B, inference-only)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# (group name, branch keys into {a, v, av})
_COMP_MEM_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('a_av', ('a', 'av')),
    ('av_v', ('av', 'v')),
    ('a_v_av', ('a', 'v', 'av')),
)


def _ca_residual(
    slots: Tensor,
    seq: Tensor,
    attn: nn.MultiheadAttention,
    norm: nn.LayerNorm,
) -> Tensor:
    q = slots.transpose(0, 1)
    kv = seq.transpose(0, 1)
    delta = attn(q, kv, kv, need_weights=False)[0]
    return norm(slots + delta.transpose(0, 1))


class _QuestStackReadout(nn.Module):
    """Single-query cross-attention readout: quest attends over stacked slots."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, quest: Tensor, tokens: Tensor) -> Tensor:
        q = quest.unsqueeze(0)
        kv = tokens.transpose(0, 1)
        attn_out = self.attn(q, kv, kv, need_weights=False)[0].squeeze(0)
        evidence = self.dropout(self.mlp(attn_out))
        return self.norm(quest + evidence)


class _CompMemGroup(nn.Module):
    """Mode B: initialize slots from words, then cross-attend modality sequences."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        branch_keys: Tuple[str, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branch_keys = branch_keys
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
            for _ in branch_keys
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in branch_keys])

    def forward(self, seqs: Dict[str, Tensor], words: Tensor) -> Tensor:
        slots = words
        for key, attn, norm in zip(self.branch_keys, self.cross_attns, self.norms):
            slots = _ca_residual(slots, seqs[key], attn, norm)
        return slots


class CompositionalMemoryHead(nn.Module):
    """Mode B only: words-as-memory slots + quest-guided readout."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_labels: int = 42,
        dropout: float = 0.1,
        **_unused,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.groups = nn.ModuleList([
            _CompMemGroup(d_model, nhead, keys, dropout)
            for _, keys in _COMP_MEM_GROUPS
        ])
        self.fusion_readout = _QuestStackReadout(d_model, nhead, dropout=dropout)
        self.head_act = nn.ReLU()
        self.head = nn.Linear(d_model, num_labels)
        self.mcq_score = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_stacked(
        self,
        a_seq: Tensor,
        v_seq: Tensor,
        av_seq: Tensor,
        words: Tensor,
    ) -> Tensor:
        if words is None:
            raise ValueError('CompositionalMemoryHead requires words [B,L,D]')
        seqs = {'a': a_seq, 'v': v_seq, 'av': av_seq}
        group_slots = [g(seqs, words) for g in self.groups]
        return torch.cat(group_slots, dim=1)

    def readout_mcq_logits(self, cand_quest: Tensor, stacked: Tensor) -> Tensor:
        b, k, d = cand_quest.shape
        l = stacked.shape[1]
        stacked_rep = stacked.unsqueeze(1).expand(b, k, l, d).reshape(b * k, l, d)
        q = cand_quest.reshape(b * k, d)
        h = self.fusion_readout(q, stacked_rep)
        return self.mcq_score(self.head_act(h)).view(b, k)

    def forward_mcq(
        self,
        a_seq: Tensor,
        v_seq: Tensor,
        av_seq: Tensor,
        quest: Tensor,
        words: Tensor,
        cand_quest_proj: Tensor,
    ) -> Dict[str, Tensor]:
        del quest
        stacked = self.compute_stacked(a_seq, v_seq, av_seq, words)
        return {'out': self.readout_mcq_logits(cand_quest_proj, stacked)}

    def forward_mcq_per_option_full(
        self,
        a_seq: Tensor,
        v_seq: Tensor,
        av_seq: Tensor,
        quest: Tensor,
        words: Tensor,
        original_batch: int,
    ) -> Dict[str, Tensor]:
        stacked = self.compute_stacked(a_seq, v_seq, av_seq, words)
        h = self.fusion_readout(quest, stacked)
        scores = self.mcq_score(self.head_act(h)).view(original_batch, -1)
        return {'out': scores}

    def forward(
        self,
        a_seq: Tensor,
        v_seq: Tensor,
        av_seq: Tensor,
        quest: Tensor,
        words: Tensor,
    ) -> Dict[str, Tensor]:
        stacked = self.compute_stacked(a_seq, v_seq, av_seq, words)
        h = self.fusion_readout(quest, stacked)
        return {'out': self.head(self.head_act(h))}
