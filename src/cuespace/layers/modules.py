import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
sys.path.append(ROOT.as_posix())

from torch import Tensor
from typing import List, Optional, Tuple
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self,
                 inp_dim: int = 512,
                 d_model: int = 512,
    ):
        super(Projection, self).__init__()
        self.proj = nn.Linear(inp_dim, d_model)
        
    def forward(self, inp: Tensor) -> Tensor:
        return self.proj(inp)


class AVQSelfAttn(nn.Module):
    """
    最佳基线：Audio-Video Self Attention with Query（无audio-video交叉注意力）
    保留：self attention 和 query attention
    移除：audio-video 之间的 cross attention
    """
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 dropout: float = 0.1,
    ):
        super(AVQSelfAttn, self).__init__()
        self.d_model = d_model

        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 不包含 crs_attn（audio-video 之间的交叉注意力）
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)
    
    def sub_forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Tensor,
                visualize: bool = False
    ) -> Tensor:
        src_q_perm = src_q.permute(1, 0, 2)  # [T, B, D]
        src_v_perm = src_v.permute(1, 0, 2)
        query_perm = query.permute(1, 0, 2)

        qst_attn, weight = self.qst_attn(src_q_perm, query_perm, query_perm)
        slf_attn = self.slf_attn(src_q_perm, src_q_perm, src_q_perm)[0]
        # 残差连接（注意：slf_attn和qst_attn的shape是[T, B, D]）
        src_q_perm = src_q_perm + \
                self.dropout(slf_attn) + \
                self.dropout(qst_attn)
        src_q_perm = self.norm1(src_q_perm)
        
        # FFN
        src_q_perm = src_q_perm + \
                self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q_perm)))))
        src_q_perm = self.norm2(src_q_perm)
        
        return src_q_perm.permute(1, 0, 2), weight  # [B, T, D]
    
    def forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Tensor,
                visualize: bool = False
    ) -> List[Tensor]:

        src1, a_weight = self.sub_forward(src_q, src_v, query, visualize)
        src2, v_weight = self.sub_forward(src_v, src_q, query, visualize)
        
        if visualize:
            return src1, src2, [a_weight, v_weight]
        return src1, src2


class QCFMoE(nn.Module):
    """Question-guided cue fusion MoE: dual A/V branch, top-k expert sequence output."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        topK: int = 5,
        n_experts: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.topK = topK
        self.n_experts = n_experts
        self.anorm = nn.LayerNorm(d_model)
        self.vnorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.router = nn.Sequential(nn.Linear(d_model, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model),
            ])
            for _ in range(n_experts)
        ])
        self.experts.apply(self._init_weights)
        self.router.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _aggregate_sequence(
        self,
        experts_logits: Tensor,
        topk_inds: Tensor,
        topk_probs: Tensor,
        shape: tuple,
        actual_topK: int,
    ) -> Tensor:
        B, T, C = shape
        experts_logits = experts_logits.permute(1, 0, 2, 3)  # [B, T, N_EXPERTS, C]
        topk_experts = torch.gather(
            experts_logits, 2,
            topk_inds.unsqueeze(1).unsqueeze(-1).expand(B, T, actual_topK, C),
        )
        topk_probs_expanded = topk_probs.unsqueeze(1).unsqueeze(-1)
        return (topk_probs_expanded * topk_experts).sum(dim=2)

    def forward(
        self,
        qst: Tensor,
        data: Tensor,
        sub_data: Optional[list] = None,
        *,
        quest_free_router: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        B, T, C = data.size()
        data = data.permute(1, 0, 2)
        if quest_free_router:
            temp_w = data.mean(dim=0)
        else:
            temp_w = self.qst_attn(qst.unsqueeze(0), data, data)[0].squeeze(0)

        router_probs = F.softmax(self.router(temp_w), dim=-1)
        actual_topK = min(self.topK, self.n_experts)
        topk_probs, topk_inds = torch.topk(router_probs, actual_topK, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        if sub_data is not None:
            a_patch = sub_data[0].permute(1, 0, 2)
            v_patch = sub_data[1].permute(1, 0, 2) if len(sub_data) > 1 else a_patch
            a_outs = torch.stack([e(data + a_patch) for e in self.experts], dim=2)
            v_outs = torch.stack([e(data + v_patch) for e in self.experts], dim=2)
        else:
            main_outs = torch.stack([e(data) for e in self.experts], dim=2)
            a_outs = v_outs = main_outs

        shape = (B, T, C)
        a_seq = self._aggregate_sequence(a_outs, topk_inds, topk_probs, shape, actual_topK)
        v_seq = self._aggregate_sequence(v_outs, topk_inds, topk_probs, shape, actual_topK)
        return self.anorm(a_seq), self.vnorm(v_seq)


class FCR(nn.Module):
    """Fine-grained Cue Retrieval: patch self-attn + frame cross-attn."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.anorm = nn.LayerNorm(d_model)
        self.vnorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, patch: Tensor, *, frame: Tensor, modal_type: str) -> Tensor:
        if modal_type not in ('audio', 'video'):
            raise ValueError(f"FCR modal_type must be 'audio' or 'video', got {modal_type!r}")

        B, T, P, D = patch.size()
        patch = patch.reshape(B * T, P, D).permute(1, 0, 2)
        patch = patch + self.slf_attn(patch, patch, patch)[0]

        query = frame.reshape(B * T, 1, D).permute(1, 0, 2)
        attn = self.crs_attn(query, patch, patch)[0].permute(1, 0, 2)
        attn = self.mlp(self.dropout(attn)).reshape(B, T, D)
        return self.anorm(attn) if modal_type == 'audio' else self.vnorm(attn)
