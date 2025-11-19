"""Thin wrapper around PyTorch's ring attention implementation.

The upstream implementation lives in ``specforge.context_parallel._torch_ring_attention``
(which is a verbatim copy of the PyTorch 2.9.1 experimental backend). This file
just exposes a stable function that SpecForge modules can call without pulling in
PyTorch's context manager stack.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist

from specforge.distributed import get_tp_device_mesh, get_tp_group

from . import _torch_ring_attention as torch_ring


class RingAttentionBackend(str, Enum):
    FLASH = "flash"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"


def _tp_mesh():
    mesh = get_tp_device_mesh()
    if mesh is None:
        raise RuntimeError("Tensor-parallel mesh is not initialized")
    return mesh


def _tp_world_size() -> int:
    tp_group = get_tp_group()
    if tp_group is None:
        return 1
    return dist.get_world_size(tp_group)


class _RingFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        dropout_p: float,
        scale: Optional[float],
    ) -> torch.Tensor:
        mesh = _tp_mesh()
        out, logsumexp = torch_ring._scaled_dot_product_ring_flash_attention(
            mesh,
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        ctx.save_for_backward(query, key, value, out, logsumexp)
        ctx.mesh = mesh
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.max_q = query.size(2)
        ctx.max_k = key.size(2)
        ctx.cum_seq_q = None
        ctx.cum_seq_k = None
        ctx.philox_seed = torch.zeros(2, dtype=torch.int64, device=query.device)
        ctx.philox_offset = torch.zeros(2, dtype=torch.int64, device=query.device)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        query, key, value, out, logsumexp = ctx.saved_tensors
        grads = torch_ring._scaled_dot_product_ring_flash_attention_backward(
            ctx.mesh,
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            ctx.cum_seq_q,
            ctx.cum_seq_k,
            ctx.max_q,
            ctx.max_k,
            ctx.dropout_p,
            ctx.is_causal,
            ctx.philox_seed,
            ctx.philox_offset,
            scale=ctx.scale,
        )
        grad_query, grad_key, grad_value = grads[:3]
        return grad_query, grad_key, grad_value, None, None, None


def ring_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: RingAttentionBackend = RingAttentionBackend.FLASH,
) -> torch.Tensor:
    if attn_mask is not None:
        raise NotImplementedError(
            "Ring attention backend does not support attn_mask; rely on is_causal"
        )

    if _tp_world_size() == 1:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    if backend != RingAttentionBackend.FLASH:
        raise NotImplementedError("Only flash ring attention is currently supported")

    return _RingFlashAttention.apply(query, key, value, is_causal, dropout_p, scale)


__all__ = [
    "RingAttentionBackend",
    "ring_scaled_dot_product_attention",
]
