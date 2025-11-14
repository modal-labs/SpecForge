import torch
import pytest

from specforge.distributed import (
    SequenceShardMetadata,
    _compute_sequence_shard_bounds,
)


def _reference_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    weighted = (target * log_probs).sum(dim=-1)
    masked = mask.squeeze(-1) * weighted
    return -masked.sum() / (logits.shape[0] * logits.shape[1])


def _reference_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    position_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    correct = (
        (logits.argmax(dim=-1) == target.argmax(dim=-1))
        * position_mask.squeeze(-1)
    ).sum()
    total = loss_mask.sum().clamp_min(1e-6)
    return correct / total


def _build_metadata(seq_len: int, tp_size: int, tp_rank: int) -> SequenceShardMetadata:
    start, end = _compute_sequence_shard_bounds(seq_len, tp_size, tp_rank)
    return SequenceShardMetadata(
        global_length=seq_len,
        shard_start=start,
        shard_end=end,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


@pytest.mark.parametrize("seq_len", [2048, 8192])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_sequence_parallel_math_matches_baseline(seq_len: int, tp_size: int):
    torch.manual_seed(0)
    batch_size = 2
    vocab = 512

    logits = torch.randn(batch_size, seq_len, vocab)
    target = torch.softmax(torch.randn(batch_size, seq_len, vocab), dim=-1)
    loss_mask = torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    position_mask = loss_mask.clone()

    baseline_loss = _reference_loss(logits, target, position_mask)
    baseline_acc = _reference_accuracy(logits, target, position_mask, loss_mask)

    aggregated_loss = 0.0
    aggregated_correct = torch.zeros(1)
    aggregated_total = torch.zeros(1)

    for rank in range(tp_size):
        metadata = _build_metadata(seq_len, tp_size, rank)
        shard = slice(metadata.shard_start, metadata.shard_end)

        local_logits = logits[:, shard]
        local_target = target[:, shard]
        local_position_mask = position_mask[:, shard]
        local_loss_mask = loss_mask[:, shard]

        local_loss = _reference_loss(local_logits, local_target, local_position_mask)
        length_scale = local_logits.shape[1] / max(metadata.global_length, 1)
        aggregated_loss += local_loss * length_scale

        local_correct = (
            (local_logits.argmax(dim=-1) == local_target.argmax(dim=-1))
            * local_position_mask.squeeze(-1)
        ).sum()
        local_total = local_loss_mask.sum().clamp_min(1e-6)
        aggregated_correct += local_correct
        aggregated_total += local_total

    aggregated_acc = (aggregated_correct / aggregated_total).squeeze(0)

    assert torch.allclose(aggregated_loss, baseline_loss, atol=1e-6)
    assert torch.allclose(aggregated_acc, baseline_acc, atol=1e-6)
