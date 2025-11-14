#!/usr/bin/env python3
import argparse
import torch
import torch.distributed as dist

import train_eagle3_online as trainer
from specforge.distributed import destroy_distributed, init_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare sequence-parallel and baseline losses")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output-dir", default="./tmp/sequence_parallel_equiv")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    parser.add_argument("--chat-template", default="llama3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--build-dataset-num-proc", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--is-vlm", action="store_true")
    return parser.parse_args()


def _prepare_batch(dataloader):
    batch = next(iter(dataloader))
    return {
        key: value.cuda()
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "loss_mask"}
    }


def _compute_loss(args, eagle3_model, target_model, data, sequence_parallel):
    args.sequence_parallel = sequence_parallel
    eagle3_data = target_model.generate_eagle3_data(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        loss_mask=data["loss_mask"],
    )

    if sequence_parallel:
        trainer._shard_eagle3_data_along_sequence(eagle3_data)
    else:
        eagle3_data.input_ids = trainer.get_dp_data_shard_from_tp(eagle3_data.input_ids)
        eagle3_data.attention_mask = trainer.get_dp_data_shard_from_tp(eagle3_data.attention_mask)
        eagle3_data.loss_mask = trainer.get_dp_data_shard_from_tp(eagle3_data.loss_mask)
        eagle3_data.target = trainer.get_dp_data_shard_from_tp(eagle3_data.target)
        eagle3_data.hidden_states = trainer.get_dp_data_shard_from_tp(eagle3_data.hidden_states)

    plosses, _, _ = eagle3_model(
        input_ids=eagle3_data.input_ids,
        attention_mask=eagle3_data.attention_mask,
        loss_mask=eagle3_data.loss_mask,
        target=eagle3_data.target,
        hidden_states=eagle3_data.hidden_states,
        sequence_metadata=getattr(eagle3_data, "sequence_metadata", None),
    )
    ploss = sum(plosses) / len(plosses)
    dist.all_reduce(ploss, op=dist.ReduceOp.AVG)
    return ploss.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    init_distributed(tp_size=args.tp_size)
    trainer.sanity_check(args)

    draft_model_config, draft_model = trainer.build_draft_model(args)
    target_model, processor = trainer.build_target_model(args, draft_model_config)
    dataloader, _, _ = trainer.build_dataloaders(args, draft_model_config, processor)
    batch = _prepare_batch(dataloader)

    eagle3_model = trainer.OnlineEagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    )

    baseline_loss = _compute_loss(args, eagle3_model, target_model, batch, False)
    sp_loss = _compute_loss(args, eagle3_model, target_model, batch, True)

    if dist.get_rank() == 0:
        diff = torch.abs(baseline_loss - sp_loss).item()
        print(
            f"baseline={baseline_loss.item():.8f}, sequence_parallel={sp_loss.item():.8f}, diff={diff:.3e}"
        )
        if diff > args.tolerance:
            raise SystemExit(1)
    destroy_distributed()


if __name__ == "__main__":
    main()
