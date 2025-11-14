from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import torch
import torch.distributed as dist

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None


@dataclass
class SequenceShardMetadata:
    global_length: int
    shard_start: int
    shard_end: int
    tp_rank: int
    tp_size: int

    @property
    def local_length(self) -> int:
        return max(self.shard_end - self.shard_start, 0)


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH


def init_distributed(timeout: int = 10, tp_size: int = 1):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=["dp", "tp"]
    )
    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")
    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _DP_GROUP = dp_group
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gather_tensor = torch.cat(obj_list, dim=dim)
    return gather_tensor


def _compute_sequence_shard_bounds(
    seq_len: int, tp_size: int, tp_rank: int
) -> (int, int):
    base = seq_len // tp_size
    remainder = seq_len % tp_size
    shard_start = tp_rank * base + min(tp_rank, remainder)
    shard_length = base + (1 if tp_rank < remainder else 0)
    shard_end = shard_start + shard_length
    return shard_start, shard_end


def compute_sequence_shard_metadata(
    seq_len: int, process_group: Optional[dist.ProcessGroup] = None
) -> SequenceShardMetadata:
    process_group = process_group or get_tp_group()
    tp_size = dist.get_world_size(process_group)
    tp_rank = dist.get_rank(process_group)
    shard_start, shard_end = _compute_sequence_shard_bounds(seq_len, tp_size, tp_rank)
    return SequenceShardMetadata(
        global_length=seq_len,
        shard_start=shard_start,
        shard_end=shard_end,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def shard_sequence_tensor(
    tensor: torch.Tensor,
    metadata: SequenceShardMetadata,
    dim: int = 1,
) -> torch.Tensor:
    if tensor.size(dim) != metadata.global_length:
        raise ValueError(
            "Tensor sequence dimension does not match metadata."
            f" Expected {metadata.global_length}, got {tensor.size(dim)}"
        )

    if metadata.local_length == metadata.global_length:
        return tensor

    slicer = [slice(None)] * tensor.ndim
    slicer[dim] = slice(metadata.shard_start, metadata.shard_end)
    return tensor[tuple(slicer)].contiguous()


def gather_sequence_tensor(
    tensor: torch.Tensor,
    metadata: SequenceShardMetadata,
    dim: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    process_group = process_group or get_tp_group()
    tp_size = dist.get_world_size(process_group)

    local_length = torch.tensor(
        [metadata.local_length], device=tensor.device, dtype=torch.long
    )
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(tp_size)]
    dist.all_gather(gathered_lengths, local_length, group=process_group)
    shard_lengths = [int(length.item()) for length in gathered_lengths]
    max_length = max(shard_lengths) if shard_lengths else 0

    padded = tensor.contiguous()
    if metadata.local_length < max_length:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_length - metadata.local_length
        padded = torch.cat([padded, tensor.new_zeros(pad_shape)], dim=dim)

    gather_list = [torch.empty_like(padded) for _ in range(tp_size)]
    dist.all_gather(gather_list, padded, group=process_group)

    collected: List[torch.Tensor] = []
    for chunk, length in zip(gather_list, shard_lengths):
        if length == 0:
            continue
        slicer = [slice(None)] * chunk.ndim
        slicer[dim] = slice(0, length)
        collected.append(chunk[tuple(slicer)])

    if not collected:
        output_shape = list(tensor.shape)
        output_shape[dim] = 0
        return tensor.new_zeros(output_shape)

    gathered = torch.cat(collected, dim=dim)
    total_length = sum(shard_lengths)
    if metadata.global_length >= 0 and total_length != metadata.global_length:
        raise RuntimeError(
            "Sequence gather produced mismatched length: "
            f"{total_length} vs expected {metadata.global_length}"
        )
    return gathered


class SequenceParallelCoordinator:
    def __init__(
        self,
        seq_dim: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.seq_dim = seq_dim
        self.process_group = process_group or get_tp_group()

    def build_metadata(self, seq_len: int) -> SequenceShardMetadata:
        return compute_sequence_shard_metadata(seq_len, self.process_group)

    def shard(self, tensor: torch.Tensor, metadata: SequenceShardMetadata) -> torch.Tensor:
        return shard_sequence_tensor(tensor, metadata, dim=self.seq_dim)

    def gather(self, tensor: torch.Tensor, metadata: SequenceShardMetadata) -> torch.Tensor:
        return gather_sequence_tensor(
            tensor, metadata, dim=self.seq_dim, process_group=self.process_group
        )


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
