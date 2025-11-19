# SpecForge Context Parallel Integration Status

_Last updated: 2025-11-19_

## Summary
We ported the PyTorch 2.9.1 ring attention backend into SpecForge and wired it into the Eagle3 online training stack. The new backend can be selected via `--attention-backend ring_context_parallel` in `scripts/train_eagle3_online.py`. When enabled, the training script splits every Eagle3 data tensor along the sequence dimension across the tensor-parallel (TP) group, passes shard offsets into `OnlineEagle3Model`, and runs `ring_scaled_dot_product_attention` inside the Llama draft model instead of standard SDPA.

## Completed Work
1. **Backend Port**
   - Copied `torch.distributed.tensor.experimental._attention` into `specforge/context_parallel/_torch_ring_attention.py`.
   - Added `specforge/context_parallel/ring.py`, exposing `ring_scaled_dot_product_attention` + `RingAttentionBackend` as a TP-aware autograd bridge.
   - New package init exports the backend symbols.

2. **Model Integration**
   - `LlamaAttention` now accepts `attention_backend` and calls the ring kernel when set to `"ring_context_parallel"` (with sanity checks on masks).
   - `LlamaDecoderLayer` and `OnlineEagle3Model` propagate the backend choice; CLI scripts support the new flag value.
   - `OnlineEagle3Model.forward` takes a `sequence_offset`, builds absolute `position_ids`, nulls masks for the ring backend, and disables the TTT cache path for now.
   - Accuracy computation reduces per-token numerators/denominators across all ranks so metrics remain global after sharding.

3. **Data Sharding**
   - `specforge/context_parallel/sharding.py` shards all Eagle3 tensors along the sequence dimension using the TP group.
   - `Eagle3TargetOutput` now records `sequence_offset` and `global_seq_length` metadata so downstream modules know where each shard belongs.
   - `scripts/train_eagle3_online.py` invokes the sharder when the ring backend is selected and forwards the offset into the model.

## Known Limitations / TODOs
1. **Cache-aware attention**
   - `cache_hidden` is disabled for `ring_context_parallel`, so every TTT step recomputes the local shard. Need to teach the ring backend (and possibly KV caches) to operate on shard-local histories so recurrent TTT usage is efficient.

2. **Masking & Padding**
   - The ring kernel currently forbids `attention_mask`; we simply set it to `None`. This means padding/two-tower masks are ignored. We must either trim sequences per shard before attention or extend the backend to accept attn_bias-like masks while maintaining causal semantics.

3. **Sequence divisibility**
   - `shard_eagle3_output_for_context_parallel` requires `seq_len % tp_size == 0`. Implement fallback padding (or ragged tail handling) so arbitrary lengths work.

4. **DP > 1 and Offline/SGL flows**
   - Only the online trainer calls the sharder. Offline and SGL scripts still replicate full sequences. Once the online path is stable, replicate the sharding logic there.

5. **Testing & Validation**
   - No automated tests cover the new backend. Add a distributed unit test that compares ring vs SDPA on short contexts (`tp=2`, `seq=512`) and a smoke test for `tp=8` shards.

6. **Documentation & Monitoring**
   - Need docs describing how to enable the backend, preconditions (tp>1, divisible sequence), and recommended monitoring (per-rank offsets, accuracy reductions). Also add runtime logging to confirm shard ranges during training.

---
**Next Steps**: address TODOs in the order above, starting with cache support and mask handling so the backend can run multi-TTT steps on padded sequences.
