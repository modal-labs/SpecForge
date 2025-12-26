# Running SpecForge EAGLE-3 OPD on Modal

Run EAGLE-3 On-Policy Distillation training on Modal's cloud infrastructure with easy sweep capabilities.

## Prerequisites

1. Install Modal: `pip install modal`
2. Authenticate: `modal setup`
3. Create required secrets:
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=<your-key>
   modal secret create huggingface-secret HF_TOKEN=<your-token>
   ```

## Quick Start

```bash
# Test mode (small model, 1 GPU, 1 epoch, synthetic dataset)
modal run modal_eagle3_opd.py --test

# Full sweep: 4 ablations × 3 LRs = 12 parallel runs (8x H100 each)
# Note: Requires dataset to be prepared first (see below)
modal run modal_eagle3_opd.py --train-data sharegpt_train.jsonl

# Single ablation (sweeps 3 LRs)
modal run modal_eagle3_opd.py --ablations full-opd

# Skip LR sweep, use middle LR only (4 runs)
modal run modal_eagle3_opd.py --no-lr-sweep
```

## Usage

### Main Sweep Entrypoint

```bash
modal run modal_eagle3_opd.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--target-model` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `--ablations` | `all` | Comma-separated list: `baseline,rkl-only,mixed,full-opd` |
| `--train-data` | `sharegpt_train.jsonl` | Dataset filename in volume (uses `<dataset>_train.jsonl` naming) |
| `--num-epochs` | `10` | Training epochs |
| `--output-prefix` | `eagle3-opd` | Output directory prefix |
| `--test` | `false` | Enable test mode (small model, 1 GPU, synthetic data) |
| `--sequential` | `false` | Run ablations sequentially instead of parallel |
| `--skip-download` | `false` | Skip model download step |
| `--single-gpu` | `false` | Use 1x H100 instead of 8x H100 |
| `--no-lr-sweep` | `false` | Use middle LR only (skip LR sweep) |

**Note:** Multi-GPU runs are fixed at 8x H100. Use `--single-gpu` for 1x H100.

### Ablation Configurations

Each ablation has per-tuned learning rates for speculative decoding training on ~200k samples with Qwen3-8B:

| Name | λ_ce | λ_rkl | β_hinge | Learning Rates | Description |
|------|------|-------|---------|----------------|-------------|
| `baseline` | 1.0 | 0.0 | 0.0 | 5e-5, 1e-4, 2e-4 | Standard EAGLE-3 (CE only) |
| `rkl-only` | 0.0 | 1.0 | 0.0 | 3e-5, 5e-5, 1e-4 | Pure OPD (reverse-KL only) |
| `mixed` | 0.5 | 0.5 | 0.0 | 5e-5, 1e-4, 2e-4 | Mixed CE + reverse-KL |
| `full-opd` | 0.4 | 0.5 | 0.1 | 3e-5, 5e-5, 1e-4 | Full OPD (CE + RKL + hinge) |

**LR rationale:** Reverse-KL loss is more sensitive to learning rate, so `rkl-only` and `full-opd` use more conservative LRs.

### Utility Functions (call remote functions directly)

```bash
# Download a model to the cache volume
modal run modal_eagle3_opd.py::download_model --model-id Qwen/Qwen3-8B

# Prepare a supported dataset (uses scripts/prepare_data.py)
# Supported: ultrachat, sharegpt, eaglechat, perfectblend, magpie-qwen2.5-pro-1m-v0.1, sharegpt4v, allava4v, opc
modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt

# Prepare dataset with sample limit
modal run modal_eagle3_opd.py::prep_dataset --dataset ultrachat --sample-size 10000

# Prepare dataset with train/eval split
modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt --split-eval

# Create synthetic test dataset
modal run modal_eagle3_opd.py::prep_test_dataset

# Check if a dataset exists
modal run modal_eagle3_opd.py::check_dataset_exists --filename sharegpt_train.jsonl
```

### Dataset Regeneration (Recommended)

For optimal draft model performance, regenerate the dataset using your target model.
This replaces assistant responses with the target model's actual outputs, improving
draft-target alignment and acceptance rates.

```bash
# 8B model: use 8 servers with TP=1 for maximum throughput
modal run modal_eagle3_opd.py::regenerate_dataset \
    --model Qwen/Qwen3-8B \
    --input-file sharegpt_train.jsonl

# 70B model: use 1 server with TP=8
modal run modal_eagle3_opd.py::regenerate_dataset \
    --model Qwen/Qwen3-70B \
    --input-file sharegpt_train.jsonl \
    --tp-size 8 --num-servers 1

# 32B model: use 4 servers with TP=2
modal run modal_eagle3_opd.py::regenerate_dataset \
    --model Qwen/QwQ-32B \
    --input-file sharegpt_train.jsonl \
    --tp-size 2 --num-servers 4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | HuggingFace model ID |
| `--input-file` | (required) | Input dataset filename |
| `--output-file` | `<input>_regenerated.jsonl` | Output filename |
| `--tp-size` | `1` | Tensor parallelism per server (1 for ≤8B, 2-4 for 8-32B, 8 for 70B+) |
| `--num-servers` | `8` | Number of sglang servers (must satisfy: tp_size × num_servers ≤ 8) |
| `--concurrency` | `64` | Concurrent requests per server |
| `--num-samples` | all | Number of samples to process |
| `--temperature` | `0.7` | Sampling temperature (0 for greedy) |

### Dataset Preprocessing (Optional)

Pre-tokenize datasets to speed up training startup:

```bash
modal run modal_eagle3_opd.py::preprocess_dataset \
    --dataset-file sharegpt_train.jsonl \
    --target-model Qwen/Qwen3-8B \
    --chat-template qwen \
    --max-length 2048
```

## Examples

### Run a Quick Test

```bash
# Uses Qwen3-0.6B, 1 GPU, 1 epoch, synthetic dataset, baseline + full-opd ablations
modal run modal_eagle3_opd.py --test
```

### Run Full Experiment

```bash
# Step 1: Download model
modal run modal_eagle3_opd.py::download_model --model-id Qwen/Qwen3-8B

# Step 2: Prepare your dataset
modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt

# Step 3 (Recommended): Regenerate with target model for better alignment
modal run modal_eagle3_opd.py::regenerate_dataset \
    --model Qwen/Qwen3-8B \
    --input-file sharegpt_train.jsonl

# Step 4: Run all ablations × LRs in parallel (12 runs × 8 H100s each)
modal run modal_eagle3_opd.py \
    --target-model Qwen/Qwen3-8B \
    --train-data sharegpt_train_regenerated.jsonl \
    --num-epochs 10

# Or skip LR sweep for faster iteration (4 runs)
modal run modal_eagle3_opd.py \
    --target-model Qwen/Qwen3-8B \
    --train-data sharegpt_train_regenerated.jsonl \
    --no-lr-sweep
```

### Custom Ablation Subset

```bash
# Run only baseline and full-opd
modal run modal_eagle3_opd.py --ablations baseline,full-opd
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Local Machine                           │
│  modal run modal_eagle3_opd.py                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Modal Cloud                            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Volumes    │  │  Volumes    │  │  Volumes    │         │
│  │ hf-cache    │  │  datasets   │  │ checkpoints │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Training Runs (Parallel)                 │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────┐ │  │
│  │  │ baseline   │ │ rkl-only   │ │ mixed      │ │full│ │  │
│  │  │ 8× H100    │ │ 8× H100    │ │ 8× H100    │ │opd │ │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Volumes

Volumes are mounted at standard cache locations that tools expect:

| Volume | Mount Path | Purpose |
|--------|------------|---------|
| `specforge-hf-cache` | `/root/.cache/huggingface` | HuggingFace model cache (HF_HOME default) |
| `specforge-datasets` | `/root/.cache/specforge/datasets` | Training datasets |
| `specforge-checkpoints` | `/root/.cache/specforge/checkpoints` | Model checkpoints and logs |

## Dataset Format

Training data must be in JSONL format with conversations:

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Troubleshooting

### Dataset not found error

If you see `Dataset 'xxx' not found in volume`, prepare the dataset first:

```bash
# Use one of the supported datasets
modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt
# Supported: ultrachat, sharegpt, eaglechat, perfectblend, magpie-qwen2.5-pro-1m-v0.1, sharegpt4v, allava4v, opc
```

### Check volume contents

```bash
modal volume ls specforge-checkpoints
modal volume ls specforge-datasets
modal volume ls specforge-hf-cache
```

### View training logs

Training logs are available in the Modal dashboard or via W&B if configured.

### Resume from checkpoint

The training script automatically resumes from the last checkpoint when `--resume` is enabled (default). Just re-run the same command.
