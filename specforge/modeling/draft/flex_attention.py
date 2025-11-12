from collections import OrderedDict
from typing import Callable, ClassVar, List, Optional, Tuple

import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

dynamo.config.recompile_limit = 64


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _cache_size: ClassVar[int] = 32
    _compiled_flex_attention: ClassVar[
        OrderedDict[Tuple[str, torch.dtype, int, int, int, int], Callable]
    ] = OrderedDict()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        # Nothing to initialize eagerly – cache is populated on demand.
        pass

    def _get_compiled(
        self,
        compile_key: Tuple[str, torch.dtype, int, int, int, int],
    ):
        if compile_key in self._compiled_flex_attention:
            # Move to the end to preserve LRU order
            self._compiled_flex_attention.move_to_end(compile_key)
            return self._compiled_flex_attention[compile_key]

        compiled_fn = torch.compile(
            flex_attention,
            dynamic=True,
        )
        self._compiled_flex_attention[compile_key] = compiled_fn
        # Trim cache if we exceed capacity
        if len(self._compiled_flex_attention) > self._cache_size:
            self._compiled_flex_attention.popitem(last=False)
        return compiled_fn

    def __call__(self, query, key, value, **kwargs):
        compile_key = (
            query.device.type,
            query.dtype,
            query.shape[1],  # num attention heads
            key.shape[1],  # num kv heads (for GQA)
            query.shape[-2],  # query length
            key.shape[-2],  # kv length
        )
        compiled_fn = self._get_compiled(compile_key)
        try:
            return compiled_fn(query, key, value, **kwargs)
        except RuntimeError as exc:
            # Flex attention occasionally triggers CUDA illegal memory access when
            # recompilation is required. In that case we fall back to eager mode and
            # drop the cached compiled graph so it can be rebuilt safely.
            if "illegal memory" in str(exc).lower():
                self._compiled_flex_attention.pop(compile_key, None)
                return flex_attention(query, key, value, **kwargs)
            raise


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    if is_torchdynamo_compiling():
        return flex_attention(query, key, value, **kwargs)
    flex_attention_compiled = WrappedFlexAttention()
    return flex_attention_compiled(
        query=query,
        key=key,
        value=value,
        **kwargs,
    )


class WrappedCreateBlockMask:
    _instance = None
    _cache_size: ClassVar[int] = 32
    _compiled_create_block_mask: ClassVar[
        OrderedDict[Tuple[str, int, int, int, int, str], Callable]
    ] = OrderedDict()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        pass

    def _get_compiled(
        self,
        compile_key: Tuple[str, int, int, int, int, str],
    ):
        if compile_key in self._compiled_create_block_mask:
            self._compiled_create_block_mask.move_to_end(compile_key)
            return self._compiled_create_block_mask[compile_key]
        compiled_fn = torch.compile(
            create_block_mask,
            dynamic=True,
        )
        self._compiled_create_block_mask[compile_key] = compiled_fn
        if len(self._compiled_create_block_mask) > self._cache_size:
            self._compiled_create_block_mask.popitem(last=False)
        return compiled_fn

    def __call__(self, mask_mod, B, H, Q_LEN, KV_LEN, device):
        mask_mod_name = getattr(mask_mod, "__name__", str(id(mask_mod)))
        compile_key = (
            device.type,
            B,
            H,
            Q_LEN,
            KV_LEN,
            mask_mod_name,
        )
        compiled_fn = self._get_compiled(compile_key)
        try:
            return compiled_fn(mask_mod, B, H, Q_LEN, KV_LEN, device)
        except RuntimeError as exc:
            if "illegal memory" in str(exc).lower():
                self._compiled_create_block_mask.pop(compile_key, None)
                return create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
            raise


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    create_block_mask_compiled = (
        WrappedCreateBlockMask()
        if not is_torchdynamo_compiling()
        else None
    )
    if create_block_mask_compiled is None:
        return create_block_mask(
            mask_mod,
            B,
            H,
            Q_LEN,
            KV_LEN,
            device,
        )
    return create_block_mask_compiled(
        mask_mod=mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod
