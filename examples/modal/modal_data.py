"""
Modal data pipeline for SpecForge EAGLE-3 training.

Provides functions for:
- prep_dataset: Convert raw data to JSONL format
- regenerate_dataset: Regenerate with target model outputs (autoscaling)
- preprocess_dataset: Tokenize and generate vocab mapping

Run from repo root:
    modal run -m examples.modal.modal_data::prep_dataset --dataset sharegpt
    modal run -m examples.modal.modal_data::regenerate_dataset --model Qwen/Qwen3-8B --input-file sharegpt_train.jsonl
    modal run -m examples.modal.modal_data::preprocess_dataset --input-file sharegpt_train.jsonl --target-model Qwen/Qwen3-8B --output-name sharegpt_qwen3
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Optional

import modal
from modal import FunctionCall

from .common import (
    DATASET_PATH,
    HF_CACHE_PATH,
    dataset_vol,
    hf_cache_vol,
    hf_secret,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sglang_image,
    train_image,
)

# =============================================================================
# App Setup
# =============================================================================

app = modal.App("specforge-data")

# =============================================================================
# Constants
# =============================================================================

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

# Rate limit for spawning regeneration tasks (per second)
REGEN_RATE_LIMIT = 200

# =============================================================================
# prep_dataset
# =============================================================================


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
) -> str:
    """
    Prepare training dataset using scripts/prepare_data.py.

    Args:
        dataset: Dataset name (ultrachat, sharegpt, eaglechat, perfectblend,
                 magpie-qwen2.5-pro-1m-v0.1, sharegpt4v, allava4v, opc)
        sample_size: Number of samples to process (None = all)
        split_eval: Whether to split into train/eval sets
        data_path: Custom data path for sharegpt format datasets

    Output files are saved to DATASET_PATH/<dataset>_train.jsonl
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


# =============================================================================
# RegenServer (modal.Cls for autoscaling SGLang inference)
# =============================================================================

# Default model for regeneration (can be overridden via REGEN_MODEL env var)
DEFAULT_REGEN_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TP_SIZE = 1


@app.cls(
    image=sglang_image,
    gpu="H100",
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=600,  # 10 min per sample max
)
class RegenServer:
    """
    SGLang-based server for regenerating conversation responses.

    Configure the model via environment variable before calling:
        server = RegenServer.with_options(env={"REGEN_MODEL": "Qwen/Qwen3-8B"})
        result = server().regenerate.remote(sample=...)

    The engine is initialized once at container startup via @modal.enter().
    """

    @modal.enter()
    def setup(self):
        """Initialize SGLang engine at container startup."""
        import sglang as sgl
        from transformers import AutoTokenizer

        model = os.environ.get("REGEN_MODEL", DEFAULT_REGEN_MODEL)
        tp_size = int(os.environ.get("REGEN_TP_SIZE", DEFAULT_TP_SIZE))

        print(f"Initializing SGLang engine for {model} (tp_size={tp_size})")

        self.engine = sgl.Engine(model_path=model, tp_size=tp_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model

        print("SGLang engine initialized")

    @modal.method()
    async def regenerate(
        self,
        sample: dict,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Regenerate all assistant turns in a conversation.

        Uses async_generate to allow concurrent inputs to be batched by SGLang.

        Args:
            sample: Dict with 'id' and 'messages' (list of {role, content})
            temperature: Sampling temperature
            max_tokens: Max tokens per response

        Returns:
            Dict with 'id' and 'messages' (regenerated)
        """
        messages = sample.get("messages") or sample.get("conversations", [])
        regenerated_messages = []

        for msg in messages:
            if msg["role"] == "user":
                regenerated_messages.append(msg)
            elif msg["role"] == "assistant":
                # Generate new assistant response from conversation so far
                response = await self.engine.async_generate(
                    prompt=self.tokenizer.apply_chat_template(
                        regenerated_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    sampling_params={
                        "temperature": temperature,
                        "max_new_tokens": max_tokens,
                    },
                )
                regenerated_messages.append({
                    "role": "assistant",
                    "content": response["text"],
                })

        return {
            "id": sample["id"],
            "messages": regenerated_messages,
        }

    @modal.exit()
    def shutdown(self):
        """Clean up SGLang engine on container shutdown."""
        if hasattr(self, "engine") and self.engine is not None:
            print("Shutting down SGLang engine")
            self.engine.shutdown()


# =============================================================================
# regenerate_dataset (client-side orchestration)
# =============================================================================


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol},
    timeout=86400,  # 24h for large datasets
)
def regenerate_dataset(
    model: str,
    input_file: str,
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tp_size: int = 1,
    max_concurrent_inputs: int = 16,
    rate_limit: int = REGEN_RATE_LIMIT,
    checkpoint_interval: int = 1000,
    _stop_after_samples: Optional[int] = None,
) -> str:
    """
    Regenerate dataset using autoscaling SGLang servers.

    Uses .spawn() for scale (up to 1M pending) and persists handles
    to volume for preemption recovery.

    Args:
        model: HuggingFace model ID
        input_file: Input JSONL file path
        output_file: Output JSONL file path (default: <input>_regenerated.jsonl)
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        tp_size: Tensor parallelism per server
        max_concurrent_inputs: Max concurrent inputs per container (tune based on
            model size and VRAM - smaller models can handle more, e.g. 32 for 0.6B,
            16 for 7B, 8 for 70B)
        rate_limit: Max spawns per second
        checkpoint_interval: Checkpoint every N samples
        _stop_after_samples: Stop after N samples (for testing preemption)

    Returns:
        Output file path
    """
    dataset_vol.reload()

    # Resolve paths
    if not input_file.startswith("/"):
        input_file = f"{DATASET_PATH}/{input_file}"

    if output_file is None:
        base = input_file.rsplit(".", 1)[0]
        output_file = f"{base}_regenerated.jsonl"
    elif not output_file.startswith("/"):
        output_file = f"{DATASET_PATH}/{output_file}"

    # Load input data
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} samples from {input_file}")

    # State files for preemption recovery
    state_dir = f"{output_file}.state"
    handles_file = f"{state_dir}/handles.json"
    completed_file = f"{state_dir}/completed.json"

    os.makedirs(state_dir, exist_ok=True)

    # Load existing state (for preemption recovery)
    handles: dict[str, str] = {}
    completed: dict[str, dict] = {}

    if os.path.exists(handles_file):
        handles = load_json(handles_file)
        print(f"Loaded {len(handles)} existing handles")

    if os.path.exists(completed_file):
        completed = load_json(completed_file)
        print(f"Loaded {len(completed)} completed results")

    # Normalize sample format (handle both 'messages' and 'conversations')
    for sample in data:
        if "messages" not in sample and "conversations" in sample:
            sample["messages"] = sample["conversations"]

    # Phase 1: Submit pending samples via .spawn()
    pending = [
        s for s in data
        if s["id"] not in completed and s["id"] not in handles
    ]
    print(f"Submitting {len(pending)} pending samples")

    # Configure RegenServer with model and concurrency
    server = RegenServer.with_options(
        env={"REGEN_MODEL": model, "REGEN_TP_SIZE": str(tp_size)}
    ).with_concurrency(max_inputs=max_concurrent_inputs)

    for i, sample in enumerate(pending):
        # Early stop for testing
        if _stop_after_samples is not None and i >= _stop_after_samples:
            print(f"Stopping early after {i} samples (test mode)")
            break

        # Rate limit
        if i > 0 and i % rate_limit == 0:
            time.sleep(1.0)

        # Spawn async task
        fc = server().regenerate.spawn(
            sample=sample,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        handles[sample["id"]] = fc.object_id

        # Checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            print(f"Checkpointing at {i + 1} samples")
            save_json(handles_file, handles)
            dataset_vol.commit()

    # Final handles checkpoint
    save_json(handles_file, handles)
    dataset_vol.commit()
    print(f"Submitted {len(handles)} total tasks")

    # Early exit for partial testing
    if _stop_after_samples is not None:
        print("Exiting early (test mode - partial submission)")
        return output_file

    # Phase 2: Collect results
    outstanding = {
        sid: fid for sid, fid in handles.items()
        if sid not in completed
    }
    print(f"Collecting {len(outstanding)} outstanding results")

    last_checkpoint_count = len(completed)

    while outstanding:
        for sample_id, fc_id in list(outstanding.items()):
            fc = FunctionCall.from_id(fc_id)

            try:
                result = fc.get(timeout=0.1)
                completed[sample_id] = result
                del outstanding[sample_id]
            except TimeoutError:
                continue  # Still running
            except Exception as e:
                print(f"Error for sample {sample_id}: {e}")
                completed[sample_id] = {"id": sample_id, "error": str(e)}
                del outstanding[sample_id]

        # Checkpoint periodically
        if len(completed) - last_checkpoint_count >= checkpoint_interval:
            print(f"Checkpointing at {len(completed)} completed")
            save_json(completed_file, completed)
            dataset_vol.commit()
            last_checkpoint_count = len(completed)

        # Progress update
        if len(outstanding) > 0 and len(outstanding) % 100 == 0:
            print(f"  {len(completed)} completed, {len(outstanding)} outstanding")

        time.sleep(0.5)

    # Final write
    print(f"Writing {len(completed)} results to {output_file}")
    results = [completed[s["id"]] for s in data if s["id"] in completed]
    save_jsonl(output_file, results)
    dataset_vol.commit()

    # Cleanup state files
    shutil.rmtree(state_dir)
    dataset_vol.commit()

    print(f"Regeneration complete: {output_file}")
    return output_file


# =============================================================================
# preprocess_dataset
# =============================================================================


@app.function(
    image=train_image,
    gpu="H100",  # Need GPU for tokenization with large models
    volumes={DATASET_PATH: dataset_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,  # 2h for large datasets
)
def preprocess_dataset(
    input_file: str,
    target_model: str,
    chat_template: str = "qwen",
    max_length: int = 2048,
) -> str:
    """
    Pre-generate vocab mapping for training.

    This function generates the vocab mapping file that train_eagle3.py would
    otherwise compute at training startup. By pre-computing it, we save time
    on subsequent training runs.

    The vocab mapping is saved to the cache location that train_eagle3.py expects:
    {DATASET_PATH}/vocab_mapping/{cache_key}.pt

    Args:
        input_file: Input JSONL file (from prep or regen)
        target_model: HuggingFace model ID for tokenizer
        chat_template: Chat template name (must match training)
        max_length: Maximum sequence length (must match training)

    Returns:
        Path to the generated vocab mapping file
    """
    import hashlib

    from transformers import AutoTokenizer

    from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
    from specforge.utils import generate_draft_model_config

    hf_cache_vol.reload()
    dataset_vol.reload()

    # Resolve input path
    if not input_file.startswith("/"):
        input_file = f"{DATASET_PATH}/{input_file}"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Generate draft model config from target model (same as train_eagle3.py)
    # Pass explicit template path since sys.argv[0] won't work in Modal
    template_config_path = "/root/specforge/configs/llama3-8B-eagle3.json"
    print(f"Generating draft config from target model: {target_model}")
    draft_config = generate_draft_model_config(
        target_model_path=target_model,
        template_config_path=template_config_path,
        cache_dir=HF_CACHE_PATH,
    )
    draft_vocab_size = draft_config["draft_vocab_size"]
    print(f"Using draft_vocab_size: {draft_vocab_size}")

    print(f"Loading tokenizer: {target_model}")
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    # Compute the same cache_key that train_eagle3.py uses
    cache_params_string = (
        f"{input_file}-"
        f"{max_length}-"
        f"{chat_template}-"
        f"{target_model}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    print(f"Cache key: {cache_key}")

    # Check if already cached
    vocab_mapping_dir = os.path.join(DATASET_PATH, "vocab_mapping")
    vocab_mapping_path = os.path.join(vocab_mapping_dir, f"{cache_key}.pt")

    if os.path.exists(vocab_mapping_path):
        print(f"Vocab mapping already cached at: {vocab_mapping_path}")
        return vocab_mapping_path

    # Build the dataset the same way train_eagle3.py does
    print(f"Building dataset from: {input_file}")
    from datasets import load_dataset

    raw_dataset = load_dataset("json", data_files=input_file, split="train")

    # Use specforge's build_eagle3_dataset to get the same format
    processed_dataset_dir = os.path.join(DATASET_PATH, "processed_dataset")
    eagle3_dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        cache_dir=processed_dataset_dir,
        cache_key=cache_key,
    )

    # Get vocab sizes from tokenizer and parameter
    target_vocab_size = len(tokenizer)

    print(f"Generating vocab mapping (target={target_vocab_size}, draft={draft_vocab_size})")

    # Use specforge's generate_vocab_mapping_file
    result_path = generate_vocab_mapping_file(
        dataset=eagle3_dataset,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        cache_dir=vocab_mapping_dir,
        cache_key=cache_key,
    )

    dataset_vol.commit()

    print(f"Vocab mapping saved to: {result_path}")
    return result_path


# =============================================================================
# Local Entrypoints
# =============================================================================


@app.local_entrypoint()
def prep(
    dataset: str = "sharegpt",
    sample_size: Optional[int] = None,
    split_eval: bool = False,
):
    """Prepare a dataset."""
    result = prep_dataset.remote(
        dataset=dataset,
        sample_size=sample_size,
        split_eval=split_eval,
    )
    print(f"Result: {result}")


@app.local_entrypoint()
def regen(
    model: str,
    input_file: str,
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """Regenerate a dataset with target model outputs."""
    result = regenerate_dataset.remote(
        model=model,
        input_file=input_file,
        output_file=output_file,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f"Result: {result}")


@app.local_entrypoint()
def preprocess(
    input_file: str,
    target_model: str,
    chat_template: str = "qwen",
    max_length: int = 2048,
):
    """Pre-generate vocab mapping for training."""
    result = preprocess_dataset.remote(
        input_file=input_file,
        target_model=target_model,
        chat_template=chat_template,
        max_length=max_length,
    )
    print(f"Result: {result}")
