"""
Modal training app for SpecForge EAGLE-3 OPD.

This module provides:
- train_run(): Single training run with flat kwargs
- train_run_single_gpu(): Single GPU variant for quick iterations
- sweep(): Parallel sweep over ablations and learning rates
- make_sweep_configs(): Config generation helper

Usage (from repo root):
    # Single training run
    uv run modal run -m examples.modal.modal_train::train \
        --target-model Qwen/Qwen3-8B \
        --preprocessed sharegpt_qwen3

    # Sweep over ablations
    uv run modal run -m examples.modal.modal_train::sweep \
        --ablations baseline,full-opd \
        --preprocessed sharegpt_qwen3
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import modal

from .common import (
    CKPT_PATH,
    DATASET_PATH,
    HF_CACHE_PATH,
    ckpt_vol,
    dataset_vol,
    hf_cache_vol,
    hf_secret,
    train_image,
    wandb_secret,
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App("specforge-train")

# =============================================================================
# Sweep Configuration
# =============================================================================

# Base config (defaults for all runs in a sweep)
BASE_CONFIG: dict[str, Any] = {
    "target_model": "Qwen/Qwen3-8B",
    "preprocessed": "sharegpt_qwen3",
    "num_epochs": 10,
    "max_length": 2048,
    "batch_size": 1,
    "ttt_length": 7,
}

# Named ablations (override specific params)
ABLATIONS: dict[str, dict[str, float]] = {
    "baseline": {"lambda_ce": 1.0, "lambda_rkl": 0.0, "beta_hinge": 0.0},
    "rkl-only": {"lambda_ce": 0.0, "lambda_rkl": 1.0, "beta_hinge": 0.0},
    "mixed": {"lambda_ce": 0.5, "lambda_rkl": 0.5, "beta_hinge": 0.0},
    "full-opd": {"lambda_ce": 0.4, "lambda_rkl": 0.5, "beta_hinge": 0.1},
}

# Per-ablation LR grids
LR_GRIDS: dict[str, list[float]] = {
    "baseline": [5e-5, 1e-4, 2e-4],
    "rkl-only": [3e-5, 5e-5, 1e-4],
    "mixed": [5e-5, 1e-4, 2e-4],
    "full-opd": [3e-5, 5e-5, 1e-4],
}


def make_sweep_configs(
    ablations: list[str],
    base_overrides: dict | None = None,
    lr_sweep: bool = True,
    target_model: str | None = None,
    preprocessed: str | None = None,
    output_prefix: str | None = None,
) -> list[dict]:
    """Generate configs for all (ablation, lr) combinations.

    Args:
        ablations: List of ablation names to run
        base_overrides: Dict of overrides to apply to all configs
        lr_sweep: If True, sweep all LRs; if False, use middle LR only
        target_model: Override target model for all configs
        preprocessed: Override preprocessed dataset name
        output_prefix: Prefix for output directories

    Returns:
        List of config dicts ready for train_run()
    """
    configs = []
    for name in ablations:
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {name}. Valid: {list(ABLATIONS.keys())}")

        ablation = ABLATIONS[name]
        lrs = LR_GRIDS[name] if lr_sweep else [LR_GRIDS[name][1]]  # middle LR

        for lr in lrs:
            config = {**BASE_CONFIG, **ablation, "learning_rate": lr}

            # Apply overrides
            if base_overrides:
                config.update(base_overrides)
            if target_model:
                config["target_model"] = target_model
            if preprocessed:
                config["preprocessed"] = preprocessed

            # Auto-generate output_dir and wandb_name
            prefix = output_prefix or "checkpoints"
            config["output_dir"] = f"{prefix}/{name}_lr{lr}"
            config["wandb_name"] = f"{name}_lr{lr}"

            configs.append(config)

    return configs


# =============================================================================
# Train Run (Multi-GPU)
# =============================================================================


@app.function(
    image=train_image,
    gpu="H100:8",
    volumes={
        DATASET_PATH: dataset_vol,
        HF_CACHE_PATH: hf_cache_vol,
        CKPT_PATH: ckpt_vol,
    },
    secrets=[hf_secret, wandb_secret],
    timeout=86400,  # 24h
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=1.0,
        initial_delay=10.0,
    ),
)
def train_run(
    # Required
    target_model: str,
    preprocessed: str,
    output_dir: str,
    # OPD loss weights
    lambda_ce: float = 1.0,
    lambda_rkl: float = 0.0,
    beta_hinge: float = 0.0,
    opd_temperature: float = 1.0,
    # Training hyperparams
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_length: int = 2048,
    ttt_length: int = 7,
    batch_size: int = 1,
    max_num_steps: int | None = None,
    save_interval: int | None = None,
    # Optional
    eval_preprocessed: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    resume: bool = True,
    seed: int = 0,
    **extra_kwargs,
) -> str:
    """Run a single EAGLE-3 OPD training job on 8x H100.

    Expects preprocessed data from preprocess_dataset():
    - {preprocessed}_dataset.pkl
    - {preprocessed}_vocab_mapping.pt

    Returns:
        Output directory path containing checkpoints
    """
    # Reload volumes to get latest data
    dataset_vol.reload()
    ckpt_vol.reload()
    hf_cache_vol.reload()

    # Build paths - expects JSONL from prep/regen, vocab mapping cached by preprocess
    dataset_file = f"{DATASET_PATH}/{preprocessed}_train.jsonl"
    full_output_dir = f"{CKPT_PATH}/{output_dir}"

    # Verify data exists
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    # Build training command
    # Vocab mapping is found via cache (preprocess_dataset pre-generates it)
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "/root/specforge/scripts/train_eagle3.py",
        "--target-model-path", target_model,
        "--train-data-path", dataset_file,
        "--output-dir", full_output_dir,
        "--model-download-dir", HF_CACHE_PATH,
        "--cache-dir", DATASET_PATH,  # Use dataset volume for cache
        # Loss weights
        "--lambda-ce", str(lambda_ce),
        "--lambda-rkl", str(lambda_rkl),
        "--beta-hinge", str(beta_hinge),
        "--opd-temperature", str(opd_temperature),
        # Training hyperparams
        "--num-epochs", str(num_epochs),
        "--learning-rate", str(learning_rate),
        "--max-length", str(max_length),
        "--ttt-length", str(ttt_length),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]

    # Optional args
    if max_num_steps is not None:
        cmd.extend(["--max-num-steps", str(max_num_steps)])

    if save_interval is not None:
        cmd.extend(["--save-interval", str(save_interval)])

    if resume:
        cmd.append("--resume")

    if wandb_project:
        cmd.extend(["--tracker", "wandb", "--wandb-project", wandb_project])
        if wandb_name:
            cmd.extend(["--wandb-name", wandb_name])

    if eval_preprocessed:
        eval_dataset_file = f"{DATASET_PATH}/{eval_preprocessed}_dataset.pkl"
        cmd.extend(["--eval-data-path", eval_dataset_file])

    # Add extra kwargs as command-line args
    for key, value in extra_kwargs.items():
        arg_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_name}")
        else:
            cmd.extend([f"--{arg_name}", str(value)])

    print(f"Running training command:\n{' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, check=True)

    # Commit checkpoint volume
    ckpt_vol.commit()

    return full_output_dir


# =============================================================================
# Train Run (Single GPU)
# =============================================================================


@app.function(
    image=train_image,
    gpu="H100",
    volumes={
        DATASET_PATH: dataset_vol,
        HF_CACHE_PATH: hf_cache_vol,
        CKPT_PATH: ckpt_vol,
    },
    secrets=[hf_secret, wandb_secret],
    timeout=86400,  # 24h
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=1.0,
        initial_delay=10.0,
    ),
)
def train_run_single_gpu(
    # Required
    target_model: str,
    preprocessed: str,
    output_dir: str,
    # OPD loss weights
    lambda_ce: float = 1.0,
    lambda_rkl: float = 0.0,
    beta_hinge: float = 0.0,
    opd_temperature: float = 1.0,
    # Training hyperparams
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_length: int = 2048,
    ttt_length: int = 7,
    batch_size: int = 1,
    max_num_steps: int | None = None,
    save_interval: int | None = None,
    # Optional
    eval_preprocessed: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    resume: bool = True,
    seed: int = 0,
    **extra_kwargs,
) -> str:
    """Run a single EAGLE-3 OPD training job on 1x H100.

    Same as train_run() but for single GPU - useful for quick iterations.
    """
    # Reload volumes
    dataset_vol.reload()
    ckpt_vol.reload()
    hf_cache_vol.reload()

    # Build paths - expects JSONL from prep/regen, vocab mapping cached by preprocess
    dataset_file = f"{DATASET_PATH}/{preprocessed}_train.jsonl"
    full_output_dir = f"{CKPT_PATH}/{output_dir}"

    # Verify data exists
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    # Build training command (single GPU via torch.distributed.run with 1 process)
    # Vocab mapping is found via cache (preprocess_dataset pre-generates it)
    cmd = [
        "python", "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "/root/specforge/scripts/train_eagle3.py",
        "--target-model-path", target_model,
        "--train-data-path", dataset_file,
        "--output-dir", full_output_dir,
        "--model-download-dir", HF_CACHE_PATH,
        "--cache-dir", DATASET_PATH,  # Use dataset volume for cache
        # Loss weights
        "--lambda-ce", str(lambda_ce),
        "--lambda-rkl", str(lambda_rkl),
        "--beta-hinge", str(beta_hinge),
        "--opd-temperature", str(opd_temperature),
        # Training hyperparams
        "--num-epochs", str(num_epochs),
        "--learning-rate", str(learning_rate),
        "--max-length", str(max_length),
        "--ttt-length", str(ttt_length),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]

    # Optional args
    if max_num_steps is not None:
        cmd.extend(["--max-num-steps", str(max_num_steps)])

    if save_interval is not None:
        cmd.extend(["--save-interval", str(save_interval)])

    if resume:
        cmd.append("--resume")

    if wandb_project:
        cmd.extend(["--tracker", "wandb", "--wandb-project", wandb_project])
        if wandb_name:
            cmd.extend(["--wandb-name", wandb_name])

    if eval_preprocessed:
        eval_dataset_file = f"{DATASET_PATH}/{eval_preprocessed}_dataset.pkl"
        cmd.extend(["--eval-data-path", eval_dataset_file])

    # Add extra kwargs
    for key, value in extra_kwargs.items():
        arg_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_name}")
        else:
            cmd.extend([f"--{arg_name}", str(value)])

    print(f"Running training command:\n{' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, check=True)

    # Commit checkpoint volume
    ckpt_vol.commit()

    return full_output_dir


# =============================================================================
# Sweep
# =============================================================================


@app.function(
    image=train_image,
    volumes={
        DATASET_PATH: dataset_vol,
        HF_CACHE_PATH: hf_cache_vol,
        CKPT_PATH: ckpt_vol,
    },
    secrets=[hf_secret, wandb_secret],
    timeout=3600,  # 1h for orchestration
)
def sweep(
    # Ablation selection
    ablations: str = "all",
    lr_sweep: bool = True,
    ablation_lrs: str | None = None,
    # Base overrides
    target_model: str | None = None,
    preprocessed: str | None = None,
    num_epochs: int | None = None,
    max_length: int | None = None,
    max_num_steps: int | None = None,
    # Output config
    output_prefix: str | None = None,
    wandb_project: str | None = None,
    # Execution mode
    sequential: bool = False,
) -> list[str]:
    """Run ablation sweep with flexible configuration.

    Examples:
        # All ablations, all LRs (full grid)
        modal run -m examples.modal.modal_train::sweep

        # Single ablation, all LRs
        modal run -m examples.modal.modal_train::sweep --ablations baseline

        # Multiple ablations, no LR sweep (use default LR each)
        modal run -m examples.modal.modal_train::sweep --ablations baseline,full-opd --no-lr-sweep

        # Explicit ablation:LR pairs
        modal run -m examples.modal.modal_train::sweep --ablation-lrs "baseline:1e-4,full-opd:5e-5"

    Returns:
        List of output directory paths for each completed run
    """
    # Build configs
    if ablation_lrs:
        # Explicit mode: parse "ablation:lr,ablation:lr,..."
        configs = []
        for pair in ablation_lrs.split(","):
            name, lr = pair.split(":")
            if name not in ABLATIONS:
                raise ValueError(f"Unknown ablation: {name}")
            config = {
                **BASE_CONFIG,
                **ABLATIONS[name],
                "learning_rate": float(lr),
            }
            prefix = output_prefix or "checkpoints"
            config["output_dir"] = f"{prefix}/{name}_lr{lr}"
            config["wandb_name"] = f"{name}_lr{lr}"
            configs.append(config)
    else:
        # Grid mode: ablations × LRs
        ablation_names = list(ABLATIONS.keys()) if ablations == "all" else ablations.split(",")
        configs = make_sweep_configs(
            ablation_names,
            lr_sweep=lr_sweep,
            target_model=target_model,
            preprocessed=preprocessed,
            output_prefix=output_prefix,
        )

    # Apply base overrides to all configs
    for cfg in configs:
        if target_model:
            cfg["target_model"] = target_model
        if preprocessed:
            cfg["preprocessed"] = preprocessed
        if num_epochs is not None:
            cfg["num_epochs"] = num_epochs
        if max_length is not None:
            cfg["max_length"] = max_length
        if max_num_steps is not None:
            cfg["max_num_steps"] = max_num_steps
        if wandb_project:
            cfg["wandb_project"] = wandb_project

    print(f"Running sweep with {len(configs)} configurations:")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}] {cfg['output_dir']} (lr={cfg['learning_rate']})")

    if sequential:
        # Run sequentially
        results = []
        for cfg in configs:
            result = train_run.remote(**cfg)
            results.append(result)
    else:
        # Run in parallel using spawn + gather
        handles = [train_run.spawn(**cfg) for cfg in configs]
        results = [h.get() for h in handles]

    return results


# =============================================================================
# CLI Entrypoints
# =============================================================================


@app.local_entrypoint()
def train(
    target_model: str,
    preprocessed: str,
    output_dir: str | None = None,
    # OPD loss weights
    lambda_ce: float = 1.0,
    lambda_rkl: float = 0.0,
    beta_hinge: float = 0.0,
    # Training hyperparams
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_length: int = 2048,
    batch_size: int = 1,
    max_num_steps: int | None = None,
    # Optional
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    resume: bool = True,
    single_gpu: bool = False,
):
    """Run a single training job.

    Examples:
        modal run -m examples.modal.modal_train::train \\
            --target-model Qwen/Qwen3-8B \\
            --preprocessed sharegpt_qwen3

        modal run -m examples.modal.modal_train::train \\
            --target-model Qwen/Qwen3-8B \\
            --preprocessed sharegpt_qwen3 \\
            --single-gpu
    """
    # Auto-generate output_dir if not specified
    if output_dir is None:
        model_short = target_model.split("/")[-1].lower()
        output_dir = f"{model_short}_{preprocessed}"

    kwargs = {
        "target_model": target_model,
        "preprocessed": preprocessed,
        "output_dir": output_dir,
        "lambda_ce": lambda_ce,
        "lambda_rkl": lambda_rkl,
        "beta_hinge": beta_hinge,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "batch_size": batch_size,
        "resume": resume,
    }

    if max_num_steps is not None:
        kwargs["max_num_steps"] = max_num_steps
    if wandb_project:
        kwargs["wandb_project"] = wandb_project
    if wandb_name:
        kwargs["wandb_name"] = wandb_name

    if single_gpu:
        result = train_run_single_gpu.remote(**kwargs)
    else:
        result = train_run.remote(**kwargs)

    print(f"\nTraining complete!")
    print(f"Output directory: {result}")


@app.local_entrypoint()
def sweep_cli(
    ablations: str = "all",
    lr_sweep: bool = True,
    ablation_lrs: str | None = None,
    target_model: str | None = None,
    preprocessed: str | None = None,
    num_epochs: int | None = None,
    max_length: int | None = None,
    max_num_steps: int | None = None,
    output_prefix: str | None = None,
    wandb_project: str | None = None,
    sequential: bool = False,
):
    """Run ablation sweep.

    Examples:
        # All ablations with default preprocessed data
        modal run -m examples.modal.modal_train::sweep_cli

        # Specific ablations
        modal run -m examples.modal.modal_train::sweep_cli \\
            --ablations baseline,full-opd \\
            --preprocessed sharegpt_qwen3

        # Quick test (single step)
        modal run -m examples.modal.modal_train::sweep_cli \\
            --ablations baseline \\
            --max-num-steps 1 \\
            --no-lr-sweep
    """
    results = sweep.remote(
        ablations=ablations,
        lr_sweep=lr_sweep,
        ablation_lrs=ablation_lrs,
        target_model=target_model,
        preprocessed=preprocessed,
        num_epochs=num_epochs,
        max_length=max_length,
        max_num_steps=max_num_steps,
        output_prefix=output_prefix,
        wandb_project=wandb_project,
        sequential=sequential,
    )

    print(f"\nSweep complete!")
    print(f"Results ({len(results)} runs):")
    for result in results:
        print(f"  - {result}")
