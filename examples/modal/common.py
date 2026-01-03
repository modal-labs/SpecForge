"""
Shared infrastructure for SpecForge Modal training apps.

This module contains volumes, images, secrets, and utilities shared across
modal_data.py and modal_train.py.

Usage (from repo root):
    modal run -m examples.modal.modal_data::prep_dataset --dataset sharegpt
    modal run -m examples.modal.modal_train::train --target-model Qwen/Qwen3-8B
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# =============================================================================
# Repository Root
# =============================================================================

# Repository root (relative to this file: examples/modal/common.py -> repo root)
REPO_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# Volumes
# =============================================================================

# Persistent storage volumes (mounted at standard cache locations)
hf_cache_vol = modal.Volume.from_name("specforge-hf-cache", create_if_missing=True)
dataset_vol = modal.Volume.from_name("specforge-datasets", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("specforge-checkpoints", create_if_missing=True)

# Volume mount paths - use standard locations that tools expect
HF_CACHE_PATH = "/root/.cache/huggingface"  # HF_HOME default
DATASET_PATH = "/root/.cache/specforge/datasets"
CKPT_PATH = "/root/.cache/specforge/checkpoints"

# =============================================================================
# Secrets
# =============================================================================

hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")

# =============================================================================
# Images
# =============================================================================

# Lightweight image for downloading models
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Full training image with specforge installed from local source
# Uses sglang base image for proper CUDA/sgl-kernel support
train_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime")
    .entrypoint([])
    .apt_install("git", "curl")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/specforge",
        copy=True,
        ignore=[
            ".git",
            ".jj",
            "__pycache__",
            "*.pyc",
            ".venv",
            "venv",
            "*.egg-info",
            "cache",
            "outputs",
            ".devcontainer",
            "examples/modal",  # Avoid rebuilds when editing modal files
        ],
    )
    # Install specforge with its pinned dependencies (torch==2.8.0, etc.)
    # Only add packages not already in requirements.txt
    .uv_pip_install(
        "/root/specforge",
        "tensorboard",
    )
    .env(
        {"TORCHINDUCTOR_CACHE_DIR": "/root/.cache/specforge/checkpoints/torchinductor"}
    )
)

# Image for sglang inference (dataset regeneration)
# Use official sglang runtime image to avoid dependency conflicts with specforge
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime")
    .entrypoint([])
    .add_local_file(
        str(REPO_ROOT / "scripts" / "regenerate_train_data.py"),
        remote_path="/root/regenerate_train_data.py",
        copy=True,
    )
    .uv_pip_install("huggingface_hub[hf_transfer]", "openai")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": HF_CACHE_PATH,
        }
    )
)

# =============================================================================
# Utility Functions
# =============================================================================


def load_jsonl(path: str) -> list[dict]:
    """Load data from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, data: list[dict]) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_json(path: str) -> dict:
    """Load data from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# =============================================================================
# Modal App
# =============================================================================

app = modal.App("specforge-common")


@app.function(image=train_image)
def debug_stub():
    """Debug stub function for testing the training image."""
    import sys
    import torch

    print("=== Debug Stub ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    # Check if specforge is importable
    try:
        import specforge

        print(f"SpecForge imported successfully from: {specforge.__file__}")
    except ImportError as e:
        print(f"Failed to import specforge: {e}")

    print("=== End Debug Stub ===")
    return {"status": "ok"}
