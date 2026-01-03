# Claude Code Instructions for Modal Development

The `examples/modal/` directory is a Python package. Run Modal commands from the repo root using the `-m` flag.

## Running Tests

```bash
uv run modal run -m examples.modal.test_data::test_prep
uv run modal run -m examples.modal.test_data::test_regen
uv run modal run -m examples.modal.test_data::test_all

uv run modal run -m examples.modal.test_train::test_train
uv run modal run -m examples.modal.test_train::test_sweep
uv run modal run -m examples.modal.test_train::test_all
```

## Data Pipeline

```bash
# Prepare dataset
uv run modal run -m examples.modal.modal_data::prep_dataset --dataset sharegpt

# Regenerate with target model
uv run modal run -m examples.modal.modal_data::regenerate_dataset --model Qwen/Qwen3-8B --input-file sharegpt_train.jsonl

# Preprocess for training
uv run modal run -m examples.modal.modal_data::preprocess_dataset --input-file sharegpt_train.jsonl --target-model Qwen/Qwen3-8B
```

## Training

```bash
# Single training run
uv run modal run -m examples.modal.modal_train::train --target-model Qwen/Qwen3-8B --preprocessed sharegpt_qwen3

# Sweep
uv run modal run -m examples.modal.modal_train::sweep --ablations baseline,full-opd
```
