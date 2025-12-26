"""
Modal app for running SpecForge EAGLE-3 OPD training with sweep capabilities.

Prerequisites:
    1. Create Modal secrets:
       - modal secret create wandb-secret WANDB_API_KEY=<your-key>
       - modal secret create huggingface-secret HF_TOKEN=<your-token>

    2. Prepare training dataset:
       modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt

    3. (Recommended) Regenerate dataset with target model for better alignment:
       modal run modal_eagle3_opd.py::regenerate_dataset --model Qwen/Qwen3-8B --input-file sharegpt_train.jsonl

    4. (Optional) Pre-tokenize dataset to speed up training:
       modal run modal_eagle3_opd.py::preprocess_dataset --dataset-file sharegpt_train.jsonl --target-model Qwen/Qwen3-8B

Usage:
    # Test mode (auto-creates synthetic dataset)
    modal run modal_eagle3_opd.py --test

    # Full sweep with all 4 ablations in parallel
    modal run modal_eagle3_opd.py --train-data sharegpt_train.jsonl

    # Custom configuration
    modal run modal_eagle3_opd.py --target-model Qwen/Qwen3-8B

    # Single ablation run
    modal run modal_eagle3_opd.py --ablations full-opd

    # Download model only
    modal run modal_eagle3_opd.py::download_model --model-id Qwen/Qwen3-8B

Best Practice:
    For optimal draft model performance, regenerate the dataset using your target model.
    This aligns the training data with the target model's output distribution, improving
    acceptance rates during speculative decoding. See:
    https://docs.sglang.ai/SpecForge/basic_usage/data_preparation.html#regenerate-datasets
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import modal

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App("specforge-eagle3-opd")

# Volumes for persistent storage (mounted at standard cache locations)
hf_cache_vol = modal.Volume.from_name("specforge-hf-cache", create_if_missing=True)
dataset_vol = modal.Volume.from_name("specforge-datasets", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("specforge-checkpoints", create_if_missing=True)

# Volume mount paths - use standard locations that tools expect
HF_CACHE_PATH = "/root/.cache/huggingface"  # HF_HOME default
DATASET_PATH = "/root/.cache/specforge/datasets"
CKPT_PATH = "/root/.cache/specforge/checkpoints"

# Repository root (relative to this file)
REPO_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# Image Definitions
# =============================================================================

# Lightweight image for downloading models
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Full training image with specforge installed from local source
train_image = (
    modal.Image.debian_slim(python_version="3.11")
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
            "examples/modal",  # Avoid rebuilds when editing this file
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

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    lambda_ce: float
    lambda_rkl: float
    beta_hinge: float
    learning_rates: tuple[float, ...]  # LRs to sweep over for this ablation
    default_lr: float  # Used when --no-lr-sweep is set
    opd_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.learning_rates:
            raise ValueError(f"{self.name}: learning_rates must be non-empty")

    def __str__(self) -> str:
        return f"{self.name}(ce={self.lambda_ce}, rkl={self.lambda_rkl}, hinge={self.beta_hinge})"


# Predefined ablation configurations with per-ablation learning rate sweeps.
# LRs are tuned for speculative decoding training on ~200k samples with Qwen3-8B:
# - baseline/mixed: standard distillation, can tolerate higher LRs
# - rkl-only/full-opd: reverse-KL is more sensitive, use conservative LRs
ABLATIONS = {
    "baseline": AblationConfig(
        name="baseline",
        lambda_ce=1.0,
        lambda_rkl=0.0,
        beta_hinge=0.0,
        learning_rates=(5e-5, 1e-4, 2e-4),
        default_lr=1e-4,
    ),
    "rkl-only": AblationConfig(
        name="rkl-only",
        lambda_ce=0.0,
        lambda_rkl=1.0,
        beta_hinge=0.0,
        learning_rates=(3e-5, 5e-5, 1e-4),
        default_lr=5e-5,
    ),
    "mixed": AblationConfig(
        name="mixed",
        lambda_ce=0.5,
        lambda_rkl=0.5,
        beta_hinge=0.0,
        learning_rates=(5e-5, 1e-4, 2e-4),
        default_lr=1e-4,
    ),
    "full-opd": AblationConfig(
        name="full-opd",
        lambda_ce=0.4,
        lambda_rkl=0.5,
        beta_hinge=0.1,
        learning_rates=(3e-5, 5e-5, 1e-4),
        default_lr=5e-5,
    ),
}

# Quick test ablations (subset for faster iteration)
TEST_ABLATIONS = ["baseline", "full-opd"]


@dataclass
class Eagle3TrainConfig:
    """Full training configuration for EAGLE-3 OPD."""

    # Model and data
    target_model_path: str = "Qwen/Qwen3-8B"
    train_data_path: str = "sharegpt.jsonl"  # relative to DATASET_PATH
    eval_data_path: Optional[str] = None
    chat_template: str = "qwen"
    is_vlm: bool = False

    # Training hyperparameters
    num_epochs: int = 10
    learning_rate: float = 1e-4
    max_length: int = 2048
    warmup_ratio: float = 0.015
    max_grad_norm: float = 0.5
    seed: int = 0

    # OPD loss weights
    lambda_ce: float = 1.0
    lambda_rkl: float = 0.0
    beta_hinge: float = 0.0
    opd_temperature: float = 1.0
    ttt_length: int = 7

    # Distributed training
    num_gpus: int = 8
    tp_size: int = 8
    draft_global_batch_size: int = 8
    draft_micro_batch_size: int = 1

    # Logging
    report_to: str = "wandb"
    wandb_project: str = "specforge-eagle3-opd"
    wandb_name: Optional[str] = None
    log_steps: int = 50

    # Output
    output_subdir: str = "eagle3-opd"  # relative to CKPT_PATH
    embedding_key: str = "model.embed_tokens.weight"

    # Misc
    dist_timeout_min: int = 30
    resume: bool = True

    def to_cli_args(self) -> list[str]:
        """Convert config to CLI arguments for train_eagle3_online.py."""
        train_data_full = f"{DATASET_PATH}/{self.train_data_path}"
        output_dir_full = f"{CKPT_PATH}/{self.output_subdir}"

        args = [
            "--target-model-path",
            self.target_model_path,
            "--train-data-path",
            train_data_full,
            "--output-dir",
            output_dir_full,
            "--num-epochs",
            str(self.num_epochs),
            "--learning-rate",
            str(self.learning_rate),
            "--max-length",
            str(self.max_length),
            "--warmup-ratio",
            str(self.warmup_ratio),
            "--max-grad-norm",
            str(self.max_grad_norm),
            "--seed",
            str(self.seed),
            "--chat-template",
            self.chat_template,
            "--cache-dir",
            HF_CACHE_PATH,
            "--embedding-key",
            self.embedding_key,
            "--tp-size",
            str(self.tp_size),
            "--draft-global-batch-size",
            str(self.draft_global_batch_size),
            "--draft-micro-batch-size",
            str(self.draft_micro_batch_size),
            "--ttt-length",
            str(self.ttt_length),
            "--lambda-ce",
            str(self.lambda_ce),
            "--lambda-rkl",
            str(self.lambda_rkl),
            "--beta-hinge",
            str(self.beta_hinge),
            "--opd-temperature",
            str(self.opd_temperature),
            "--dist-timeout",
            str(self.dist_timeout_min),
            "--log-steps",
            str(self.log_steps),
            "--report-to",
            self.report_to,
        ]

        if self.report_to == "wandb":
            args += ["--wandb-project", self.wandb_project]
            if self.wandb_name:
                args += ["--wandb-name", self.wandb_name]

        if self.eval_data_path:
            args += ["--eval-data-path", f"{DATASET_PATH}/{self.eval_data_path}"]

        if self.is_vlm:
            args += ["--is-vlm"]

        if self.resume:
            args += ["--resume"]

        return args


def make_config_for_ablation(
    base: Eagle3TrainConfig,
    ablation: AblationConfig,
    learning_rate: float,
) -> Eagle3TrainConfig:
    """Create a training config for a specific ablation and learning rate."""
    lr_str = f"{learning_rate:.0e}".replace("-0", "-")  # e.g., "1e-4"
    run_name = f"{ablation.name}-lr{lr_str}"

    cfg_dict = asdict(base)
    cfg_dict.update(
        lambda_ce=ablation.lambda_ce,
        lambda_rkl=ablation.lambda_rkl,
        beta_hinge=ablation.beta_hinge,
        opd_temperature=ablation.opd_temperature,
        learning_rate=learning_rate,
        output_subdir=f"{base.output_subdir}-{run_name}",
        wandb_name=f"{base.wandb_name or base.output_subdir}-{run_name}",
    )
    return Eagle3TrainConfig(**cfg_dict)


# =============================================================================
# Modal Functions
# =============================================================================


@app.function(
    image=download_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=60 * 60,  # 1 hour
)
def download_model(model_id: str = "Qwen/Qwen3-8B"):
    """Download a model from HuggingFace to the cache volume."""
    from huggingface_hub import snapshot_download

    hf_cache_vol.reload()

    print(f"Downloading model: {model_id}")
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=HF_CACHE_PATH,
    )
    print(f"Model downloaded to: {local_dir}")

    hf_cache_vol.commit()
    return local_dir


SUPPORTED_DATASETS = [
    "ultrachat",
    "sharegpt",
    "eaglechat",
    "perfectblend",
    "magpie-qwen2.5-pro-1m-v0.1",
    "sharegpt4v",
    "allava4v",
    "opc",
]


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol},
    secrets=[hf_secret],
    timeout=60 * 60,  # 1 hour
)
def prep_dataset(
    dataset: str = "sharegpt",
    sample_size: Optional[int] = None,
    split_eval: bool = False,
    data_path: Optional[str] = None,
):
    """
    Prepare training dataset using scripts/prepare_data.py.

    Args:
        dataset: Dataset name (ultrachat, sharegpt, eaglechat, perfectblend,
                 magpie-qwen2.5-pro-1m-v0.1, sharegpt4v, allava4v, opc)
        sample_size: Number of samples to process (None = all)
        split_eval: Whether to split into train/eval sets
        data_path: Custom data path for sharegpt format datasets

    Output files are saved to /vol/datasets/<dataset>_train.jsonl (and _test.jsonl if split_eval).
    """
    dataset_vol.reload()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported: {SUPPORTED_DATASETS}"
        )

    cmd = [
        "python",
        "/root/specforge/scripts/prepare_data.py",
        "--dataset",
        dataset,
        "--output-path",
        DATASET_PATH,
    ]

    if sample_size is not None:
        cmd.extend(["--sample-size", str(sample_size)])

    if split_eval:
        cmd.append("--split-eval")

    if data_path is not None:
        cmd.extend(["--data-path", data_path])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Dataset preparation failed with return code {result.returncode}"
        )

    dataset_vol.commit()

    output_file = f"{DATASET_PATH}/{dataset}_train.jsonl"
    print(f"Dataset prepared: {output_file}")
    return output_file


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol},
    timeout=60 * 10,
)
def prep_test_dataset(num_samples: int = 100):
    """
    Create a small synthetic test dataset for validation.

    Uses the format from scripts/prepare_data.py:
    {"id": str, "conversations": [{"role": "user/assistant", "content": "..."}]}

    Output: /vol/datasets/test_train.jsonl
    """
    import json

    dataset_vol.reload()

    dest_path = f"{DATASET_PATH}/test_train.jsonl"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"Creating synthetic test dataset with {num_samples} samples...")

    with open(dest_path, "w") as f:
        for i in range(num_samples):
            sample = {
                "id": f"test-{i}",
                "conversations": [
                    {"role": "user", "content": f"What is {i} + {i}?"},
                    {
                        "role": "assistant",
                        "content": f"The answer is {i + i}. Let me explain: when you add {i} to itself, you get {i * 2}.",
                    },
                ],
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Test dataset saved to: {dest_path}")
    dataset_vol.commit()
    return dest_path


@app.function(
    image=sglang_image,
    gpu="H100:8",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DATASET_PATH: dataset_vol,
    },
    secrets=[hf_secret],
    timeout=24 * 60 * 60,  # 24 hours for large datasets
)
def regenerate_dataset(
    model: str,
    input_file: str,
    output_file: Optional[str] = None,
    tp_size: int = 1,
    num_servers: int = 8,
    concurrency: int = 64,
    num_samples: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """
    Regenerate dataset by replacing assistant responses with target model outputs.

    This improves draft-target alignment by training on the target model's actual
    output distribution. Recommended for optimal speculative decoding performance.

    Uses scripts/regenerate_train_data.py with multiple sglang servers for throughput.

    Args:
        model: HuggingFace model ID (e.g., "Qwen/Qwen3-8B")
        input_file: Input dataset filename in volume (e.g., "sharegpt_train.jsonl")
        output_file: Output filename (defaults to "<input>_regenerated.jsonl")
        tp_size: Tensor parallelism per sglang server. Use 1 for ≤8B models,
                 2-4 for 8B-32B, 8 for 70B+.
        num_servers: Number of sglang servers to launch for data parallelism.
                     Must satisfy: tp_size * num_servers <= 8 (available GPUs).
        concurrency: Number of concurrent requests per server.
        num_samples: Number of samples to process (None = all).
        temperature: Sampling temperature (default 0.7, use 0 for greedy).
        max_tokens: Maximum tokens to generate per response.

    Examples:
        # 8B model: 8 servers with TP=1 for max throughput
        modal run modal_eagle3_opd.py::regenerate_dataset --model Qwen/Qwen3-8B --input-file sharegpt_train.jsonl

        # 70B model: 1 server with TP=8
        modal run modal_eagle3_opd.py::regenerate_dataset --model Qwen/Qwen3-70B --input-file sharegpt_train.jsonl --tp-size 8 --num-servers 1

        # 32B model: 4 servers with TP=2
        modal run modal_eagle3_opd.py::regenerate_dataset --model Qwen/QwQ-32B --input-file sharegpt_train.jsonl --tp-size 2 --num-servers 4
    """
    import time

    import requests

    hf_cache_vol.reload()
    dataset_vol.reload()

    # Validate GPU allocation
    total_gpus = 8
    if tp_size * num_servers > total_gpus:
        raise ValueError(
            f"tp_size ({tp_size}) * num_servers ({num_servers}) = {tp_size * num_servers} "
            f"exceeds available GPUs ({total_gpus})"
        )

    # Set up paths
    input_path = f"{DATASET_PATH}/{input_file}"
    if output_file is None:
        base_name = input_file.rsplit(".", 1)[0]
        output_file = f"{base_name}_regenerated.jsonl"
    output_path = f"{DATASET_PATH}/{output_file}"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Launch sglang servers
    base_port = 30000
    server_processes = []
    server_addresses = []

    def cleanup_servers():
        for proc in server_processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()

    try:
        for i in range(num_servers):
            port = base_port + i
            # Calculate which GPUs this server uses
            start_gpu = i * tp_size
            gpu_ids = ",".join(str(g) for g in range(start_gpu, start_gpu + tp_size))

            cmd = [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model",
                model,
                "--trust-remote-code",
                "--tp-size",
                str(tp_size),
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.85",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--max-running-requests",
                "128",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids

            print(
                f"Launching server {i + 1}/{num_servers} on port {port} (GPUs: {gpu_ids})"
            )
            proc = subprocess.Popen(cmd, env=env)
            server_processes.append(proc)
            server_addresses.append(f"localhost:{port}")

        # Wait for all servers to be ready
        print("Waiting for servers to be ready...")
        for addr in server_addresses:
            start_time = time.time()
            while time.time() - start_time < 3600:
                try:
                    resp = requests.get(f"http://{addr}/health", timeout=5)
                    if resp.status_code == 200:
                        print(f"  {addr} ready")
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError(f"Server at {addr} failed to start")

        print(f"All {num_servers} servers ready!")

        # Build command for regenerate_train_data.py
        cmd = [
            "python3",
            "/root/regenerate_train_data.py",
            "--model",
            model,
            "--input-file-path",
            input_path,
            "--output-file-path",
            output_path,
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
            "--concurrency",
            str(concurrency),
            "--server-address",
            *server_addresses,
        ]

        if num_samples is not None:
            cmd.extend(["--num-samples", str(num_samples)])

        print(f"Running: {' '.join(cmd[:8])}...")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            raise RuntimeError(
                f"Regeneration failed with return code {result.returncode}"
            )

        print(f"Regeneration complete: {output_path}")
        dataset_vol.commit()
        return output_file

    finally:
        cleanup_servers()


@app.function(
    image=train_image,
    gpu="H100:1",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DATASET_PATH: dataset_vol,
    },
    secrets=[hf_secret],
    timeout=4 * 60 * 60,  # 4 hours
)
def preprocess_dataset(
    dataset_file: str,
    target_model: str,
    chat_template: str = "qwen",
    max_length: int = 2048,
    num_proc: int = 8,
):
    """
    Pre-tokenize and cache dataset for faster training startup.

    This runs the tokenization and loss mask generation ahead of time,
    storing results in the dataset volume. Training will load from cache.

    Args:
        dataset_file: Dataset filename in volume (e.g., "sharegpt_train.jsonl")
        target_model: HuggingFace model ID for tokenizer (e.g., "Qwen/Qwen3-8B")
        chat_template: Chat template name (e.g., "qwen", "llama3")
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel tokenization

    Output:
        Cache file at: /root/.cache/specforge/datasets/preprocessed/<dataset>_<model>.pkl
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from specforge.data.preprocessing import build_eagle3_dataset

    hf_cache_vol.reload()
    dataset_vol.reload()

    dataset_path = f"{DATASET_PATH}/{dataset_file}"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Create cache key from dataset + model + max_length
    model_safe = target_model.replace("/", "_")
    base_name = dataset_file.rsplit(".", 1)[0]
    cache_key = f"{base_name}_{model_safe}_len{max_length}"
    cache_dir = f"{DATASET_PATH}/preprocessed"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading tokenizer: {target_model}")
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    print(f"Loading dataset: {dataset_path}")
    ds = load_dataset("json", data_files=dataset_path, split="train")

    print(f"Preprocessing dataset (cache_key={cache_key})...")
    processed_ds = build_eagle3_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        num_proc=num_proc,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )

    print(f"Preprocessed {len(processed_ds)} samples")
    print(f"Cache saved to: {cache_dir}/{cache_key}.pkl")

    dataset_vol.commit()
    return f"{cache_dir}/{cache_key}.pkl"


@app.function(
    image=download_image,
    volumes={DATASET_PATH: dataset_vol},
    timeout=60,
)
def check_dataset_exists(filename: str) -> bool:
    """Check if a dataset file exists in the dataset volume."""
    dataset_vol.reload()
    path = f"{DATASET_PATH}/{filename}"
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        print(f"Dataset found: {path} ({size} bytes)")
    else:
        print(f"Dataset not found: {path}")
    return exists


@app.function(
    image=train_image,
    gpu="H100!:8",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DATASET_PATH: dataset_vol,
        CKPT_PATH: ckpt_vol,
    },
    secrets=[wandb_secret, hf_secret],
    timeout=24 * 60 * 60,  # 24 hours
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
)
def run_training(train_cfg_dict: dict) -> str:
    """
    Run EAGLE-3 OPD training using torchrun.

    Returns the output directory path on success.
    """
    cfg = Eagle3TrainConfig(**train_cfg_dict)

    # Refresh volumes to get latest state
    hf_cache_vol.reload()
    dataset_vol.reload()
    ckpt_vol.reload()

    # Ensure output directory exists
    output_dir = f"{CKPT_PATH}/{cfg.output_subdir}"
    os.makedirs(output_dir, exist_ok=True)

    # Build torchrun command
    train_script = "/root/specforge/scripts/train_eagle3_online.py"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(cfg.num_gpus),
        train_script,
    ] + cfg.to_cli_args()

    print(f"Running training: {cfg.wandb_name or cfg.output_subdir}")
    print(f"Command: {' '.join(cmd[:10])}...")  # Print first 10 args

    # Run training
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Commit checkpoints
    ckpt_vol.commit()

    print(f"Training complete: {output_dir}")
    return output_dir


# Smaller GPU config for testing
@app.function(
    image=train_image,
    gpu="H100!",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DATASET_PATH: dataset_vol,
        CKPT_PATH: ckpt_vol,
    },
    secrets=[wandb_secret, hf_secret],
    timeout=6 * 60 * 60,  # 6 hours
)
def run_training_single_gpu(train_cfg_dict: dict) -> str:
    """Run training on a single GPU (for testing)."""
    cfg = Eagle3TrainConfig(**train_cfg_dict)

    # Override GPU settings for single GPU
    cfg.num_gpus = 1
    cfg.tp_size = 1

    hf_cache_vol.reload()
    dataset_vol.reload()
    ckpt_vol.reload()

    output_dir = f"{CKPT_PATH}/{cfg.output_subdir}"
    os.makedirs(output_dir, exist_ok=True)

    train_script = "/root/specforge/scripts/train_eagle3_online.py"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        train_script,
    ] + cfg.to_cli_args()

    print(f"Running single-GPU training: {cfg.wandb_name or cfg.output_subdir}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    ckpt_vol.commit()
    return output_dir


# =============================================================================
# Local Entrypoints
# =============================================================================


# GPU configuration constants (must match @app.function gpu= parameters)
MULTI_GPU_COUNT = 8  # run_training uses gpu="H100:8"
SINGLE_GPU_COUNT = 1  # run_training_single_gpu uses gpu="H100:1"


@app.local_entrypoint()
def main(
    # Model configuration
    target_model: str = "Qwen/Qwen3-8B",
    # Ablation selection
    ablations: str = "all",  # comma-separated list or "all"
    # Dataset (uses <dataset>_train.jsonl naming from prepare_data.py)
    train_data: str = "sharegpt_train.jsonl",
    # Training params
    num_epochs: int = 10,
    max_length: int = 2048,
    # Output
    output_prefix: str = "eagle3-opd",
    wandb_project: str = "specforge-eagle3-opd",
    # Flags
    test: bool = False,
    skip_download: bool = False,
    sequential: bool = False,
    single_gpu: bool = False,
    no_lr_sweep: bool = False,  # Use only middle LR per ablation
):
    """
    Run EAGLE-3 OPD training sweep on Modal.

    Sweeps over ablation configurations AND learning rates. Each ablation has
    3 per-tuned LRs (see ABLATIONS), resulting in 12 runs for all ablations.

    Note: Multi-GPU runs use 8x H100 GPUs (fixed). Use --single-gpu for 1x H100.

    Examples:
        # Full sweep: 4 ablations × 3 LRs = 12 parallel runs (8x H100 each)
        modal run modal_eagle3_opd.py

        # Test mode (smaller model, 1 GPU, fewer epochs, subset of ablations)
        modal run modal_eagle3_opd.py --test

        # Specific ablations only (still sweeps LRs for each)
        modal run modal_eagle3_opd.py --ablations baseline,full-opd

        # Skip LR sweep, use middle LR only (4 runs)
        modal run modal_eagle3_opd.py --no-lr-sweep

        # Sequential execution (one at a time)
        modal run modal_eagle3_opd.py --sequential

        # Single GPU mode (1x H100)
        modal run modal_eagle3_opd.py --single-gpu

        # Call remote functions directly:
        modal run modal_eagle3_opd.py::download_model --model-id Qwen/Qwen3-8B
        modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt
        modal run modal_eagle3_opd.py::prep_test_dataset
    """
    # Determine GPU count based on mode
    num_gpus = SINGLE_GPU_COUNT if single_gpu else MULTI_GPU_COUNT

    # Apply test mode overrides
    if test:
        target_model = "Qwen/Qwen3-0.6B"  # Much smaller model
        num_epochs = 1
        max_length = 512
        ablations = ",".join(TEST_ABLATIONS)
        output_prefix = "test-eagle3-opd"
        train_data = "test_train.jsonl"  # Use test dataset
        single_gpu = True
        no_lr_sweep = True  # Use middle LR only in test mode
        print("=== TEST MODE ===")
        print(f"Using smaller model: {target_model}")
        print(f"Epochs: {num_epochs}, Max length: {max_length}")
        print("Preparing test dataset...")
        prep_test_dataset.remote()
        print("Test dataset ready.")

    # Parse ablation selection
    if ablations == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [a.strip() for a in ablations.split(",")]

    # Validate ablation names
    for name in ablation_names:
        if name not in ABLATIONS:
            raise ValueError(
                f"Unknown ablation: {name}. Available: {list(ABLATIONS.keys())}"
            )

    selected_ablations = [ABLATIONS[name] for name in ablation_names]

    # Generate (ablation, lr) combinations
    sweep_configs: list[tuple[AblationConfig, float]] = []
    for ab in selected_ablations:
        if no_lr_sweep:
            sweep_configs.append((ab, ab.default_lr))
        else:
            for lr in ab.learning_rates:
                sweep_configs.append((ab, lr))

    print(f"\n{'=' * 60}")
    print("SpecForge EAGLE-3 OPD Training Sweep")
    print(f"{'=' * 60}")
    print(f"Target model: {target_model}")
    print(f"GPUs per run: {1 if single_gpu else num_gpus}")
    print(f"Ablations: {[a.name for a in selected_ablations]}")
    print(f"LR sweep: {'disabled (middle LR only)' if no_lr_sweep else 'enabled'}")
    print(f"Total runs: {len(sweep_configs)}")
    for ab, lr in sweep_configs:
        print(f"  - {ab.name} @ lr={lr:.0e}")
    print(f"Training data: {train_data}")
    print(f"Epochs: {num_epochs}")
    print(f"Output prefix: {output_prefix}")
    print(f"Mode: {'sequential' if sequential else 'parallel'}")
    print(f"{'=' * 60}\n")

    # Step 1: Validate dataset exists
    print(f"Checking dataset: {train_data}")
    if not check_dataset_exists.remote(train_data):
        raise RuntimeError(
            f"Dataset '{train_data}' not found in volume. "
            f"Prepare it first with:\n"
            f"  modal run modal_eagle3_opd.py::prep_dataset --dataset sharegpt\n"
            f"  (supported: {', '.join(SUPPORTED_DATASETS)})\n"
            f"Or use --test mode which auto-creates a synthetic dataset."
        )
    print("Dataset validated.\n")

    # Step 2: Download model (if not skipped)
    if not skip_download:
        print(f"Downloading model: {target_model}")
        download_model.remote(target_model)
        print("Model download complete.\n")

    # Step 3: Build base config
    base_config = Eagle3TrainConfig(
        target_model_path=target_model,
        train_data_path=train_data,
        num_epochs=num_epochs,
        max_length=max_length,
        num_gpus=1 if single_gpu else num_gpus,
        tp_size=1 if single_gpu else num_gpus,
        output_subdir=output_prefix,
        wandb_project=wandb_project,
    )

    # Step 4: Create configs for each (ablation, lr) combination
    configs = [
        make_config_for_ablation(base_config, ab, lr) for ab, lr in sweep_configs
    ]

    # Step 5: Run training
    train_fn = run_training_single_gpu if single_gpu else run_training

    if sequential:
        # Run one at a time
        results = []
        for cfg in configs:
            print(f"Starting: {cfg.wandb_name}")
            result = train_fn.remote(asdict(cfg))
            results.append(result)
            print(f"Completed: {result}\n")
    else:
        # Run all in parallel using spawn
        print(f"Launching {len(configs)} training runs in parallel...")
        handles = [train_fn.spawn(asdict(cfg)) for cfg in configs]

        # Wait for all to complete
        results = modal.FunctionCall.gather(*handles)

    print(f"\n{'=' * 60}")
    print("All training runs complete!")
    print(f"{'=' * 60}")
    for i, result in enumerate(results):
        ab, lr = sweep_configs[i]
        print(f"  {ab.name} @ lr={lr:.0e}: {result}")

    return results
