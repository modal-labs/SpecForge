# On-Policy Distillation (OPD) for EAGLE-3

Scripts for training EAGLE-3 with OPD to improve acceptance rates.

## Quick Start

```bash
# Run baseline (standard EAGLE-3)
./run_qwen3_baseline.sh 8

# Run full OPD
./run_qwen3_full_opd.sh 8

# Run all ablations
./run_all_ablations.sh 8
```

## Ablations

| Script | λ_ce | λ_rkl | β_hinge | Description |
|--------|------|-------|---------|-------------|
| `run_qwen3_baseline.sh` | 1.0 | 0.0 | 0.0 | Standard EAGLE-3 (CE only) |
| `run_qwen3_rkl_only.sh` | 0.0 | 1.0 | 0.0 | Pure OPD (reverse-KL only) |
| `run_qwen3_mixed.sh` | 0.5 | 0.5 | 0.0 | Mixed CE + reverse-KL |
| `run_qwen3_full_opd.sh` | 0.4 | 0.5 | 0.1 | Full OPD (CE + RKL + hinge) |

## Customization

Set environment variables before running:

```bash
export TARGET_MODEL="Qwen/Qwen3-8B"
export TRAIN_DATA="/path/to/train.jsonl"
export OUTPUT_DIR="/path/to/outputs"
./run_qwen3_full_opd.sh 8
```
