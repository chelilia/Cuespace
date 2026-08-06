"""Apply ToMe token merging to CLIP ViT visual encoder (ViT-L/14@336px)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.cuespace.layers.clip.base.clip_base import ResidualAttentionBlock, VisionTransformer
from src.cuespace.layers.tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from src.cuespace.layers.tome.utils import parse_r


class ToMeClipBlock(ResidualAttentionBlock):
    """ResidualAttentionBlock with ToMe between attention and MLP."""

    def _key_metric(self, x_lnd: torch.Tensor) -> torch.Tensor:
        x_nld = x_lnd.permute(1, 0, 2)
        B, L, C = x_nld.shape
        num_heads = self.attn.num_heads
        head_dim = C // num_heads
        qkv = F.linear(x_nld, self.attn.in_proj_weight, self.attn.in_proj_bias)
        k = qkv.chunk(3, dim=-1)[1]
        k = k.reshape(B, L, num_heads, head_dim).transpose(1, 2)
        return k.mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out

        r = self._tome_info["r"].pop(0)
        if r > 0:
            x_nld = x.permute(1, 0, 2)
            metric = self._key_metric(x_norm)
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x_nld, self._tome_info["source"]
                )
            x_nld, self._tome_info["size"] = merge_wavg(
                merge, x_nld, self._tome_info["size"]
            )
            x = x_nld.permute(1, 0, 2)

        x_nld = x.permute(1, 0, 2)
        x_nld = x_nld + self.mlp(self.ln_2(x_nld))
        return x_nld.permute(1, 0, 2)


def _make_tome_visual_class(vt_class):
    class ToMeClipVisionTransformer(vt_class):
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if hasattr(self, "_tome_info") and self._tome_info is not None:
                n_blocks = len(self.transformer.resblocks)
                self._tome_info["r"] = parse_r(n_blocks, self.r)
                self._tome_info["size"] = None
                self._tome_info["source"] = None

            x = x.type(self.conv1.weight.dtype)
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat(
                [
                    self.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)

            if self.proj is not None:
                cls_token = x[:, 0, :] @ self.proj
                return cls_token, x
            return x[:, 0, :], x[:, 1:, :]

    return ToMeClipVisionTransformer


def apply_patch(
    visual: VisionTransformer,
    trace_source: bool = False,
) -> VisionTransformer:
    ToMeVT = _make_tome_visual_class(visual.__class__)
    visual.__class__ = ToMeVT

    visual.r = 0
    visual._tome_info = {
        "r": visual.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": False,
        "class_token": True,
        "distill_token": False,
    }

    for block in visual.transformer.resblocks:
        if isinstance(block, ResidualAttentionBlock):
            block.__class__ = ToMeClipBlock
            block._tome_info = visual._tome_info

    return visual


def configure_tome_r(
    visual: VisionTransformer,
    tokens: int = 14,
    layers: int = 23,
    r_per_layer: int = 25,
) -> None:
    num_blocks = len(visual.transformer.resblocks)
    visual.r = [r_per_layer] * min(layers, num_blocks - 1)


__all__ = ["apply_patch", "configure_tome_r", "ToMeClipBlock"]
