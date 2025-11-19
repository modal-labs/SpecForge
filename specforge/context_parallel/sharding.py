"""Utilities for sharding Eagle3 inputs across tensor-parallel ranks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from specforge.distributed import get_tp_group, shard_tensor
from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput


def _shard_dim(tensor: torch.Tensor, dim: int, process_group: dist.ProcessGroup) -> torch.Tensor:
    if tensor is None:
        return tensor
    return shard_tensor(tensor, process_group=process_group, dim=dim)


def shard_eagle3_output_for_context_parallel(
    output: Eagle3TargetOutput,
    seq_dim: int = 1,
) -> Eagle3TargetOutput:
    """Shard sequence dimensions across the tensor-parallel group.

    Args:
        output: ``Eagle3TargetOutput`` containing tensors of shape ``[B, S, ...]``.
        seq_dim: The dimension along which the sequence lives (defaults to 1).

    Returns:
        The same ``Eagle3TargetOutput`` instance with tensors replaced by
        per-rank shards and ``sequence_offset``/``global_seq_length`` updated.
    """

    tp_group = get_tp_group()
    if tp_group is None:
        raise RuntimeError("Tensor-parallel group is not initialized")

    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)

    seq_len = output.hidden_states.size(seq_dim)
    if seq_len % world_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by tp_size={world_size} for context parallelism"
        )

    shard_len = seq_len // world_size
    seq_offset = shard_len * rank

    output.hidden_states = _shard_dim(output.hidden_states, seq_dim, tp_group)
    output.target = _shard_dim(output.target, seq_dim, tp_group)
    output.loss_mask = _shard_dim(output.loss_mask, seq_dim, tp_group)
    output.input_ids = _shard_dim(output.input_ids, seq_dim, tp_group)
    output.attention_mask = _shard_dim(output.attention_mask, seq_dim, tp_group)
    if output.last_hidden_states is not None:
        output.last_hidden_states = _shard_dim(
            output.last_hidden_states, seq_dim, tp_group
        )

    output.sequence_offset = seq_offset
    output.global_seq_length = seq_len
    return output


__all__ = ["shard_eagle3_output_for_context_parallel"]
