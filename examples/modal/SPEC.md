# Spec: Modal Training App for SpecForge EAGLE-3 OPD

## Overview

This spec defines the intended future state for the Modal training app in `examples/modal/`. The goal is to expose the training benchmark pipeline steps conveniently while avoiding excessive abstraction.

**Scope** (from TODO.md, immediate items only):
1. Train single run with kwargs
2. Sweeps (same file as training, simple layered config approach)
3. Dataset regeneration with autoscaling (Modal-native, cache-aware routing)
4. SGLang placement for online training (teacher queries - already co-located)

**Out of scope**: Runtime-configurable GPUs, DDP with `@clustered`, token-wise exponential decay tuning, evaluation/benchmarking.

---

## 1. Train Single Run

### Interface

A Modal function that can be invoked via:
- `modal run modal_train.py train --target-model Qwen/Qwen3-8B --preprocessed sharegpt_qwen3 ...`
- Programmatically via `train_run.spawn(**kwargs)` or `train_run.remote(**kwargs)`

### Design

```python
@app.function(
    gpu="H100:8",
    image=train_image,
    volumes={DATASET_PATH: dataset_vol, HF_CACHE_PATH: hf_cache_vol, CKPT_PATH: ckpt_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=86400,  # 24h
)
def train_run(
    # Required
    target_model: str,
    preprocessed: str,            # Name of preprocessed dataset (from preprocess_dataset)
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

    # Optional
    eval_preprocessed: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    resume: bool = True,
    seed: int = 0,
    **extra_kwargs,
) -> str:
    """Run a single EAGLE-3 OPD training job. Returns output directory path.

    Expects preprocessed data from preprocess_dataset():
    - {preprocessed}_dataset.pkl
    - {preprocessed}_vocab_mapping.pt
    """
```

### Key Points

- **Flat kwargs**: No config dataclass in the function signature; just pass kwargs directly
- **Sensible defaults**: Most params have defaults matching current best practices
- **Extra kwargs passthrough**: `**extra_kwargs` allows passing additional train_eagle3.py args without modifying this function
- **Return value**: Output directory path for downstream use

### GPU Configuration

- Default: 8x H100 for full training runs
- Single GPU variant: `train_run_single_gpu()` with `gpu="H100"` for quick iterations

---

## 2. Sweeps (Same File as Training)

Sweeps live in `modal_train.py` alongside single-run training. CLI-invokable functions for both.

### Layered Config Pattern

Simple dict-merge approach (no Hydra, no inheritance):

```python
# Base config (defaults for all runs in a sweep)
BASE_CONFIG = {
    "target_model": "Qwen/Qwen3-8B",
    "preprocessed": "sharegpt_qwen3",  # Name of preprocessed dataset
    "num_epochs": 10,
    "max_length": 2048,
}

# Named ablations (override specific params)
ABLATIONS = {
    "baseline": {"lambda_ce": 1.0, "lambda_rkl": 0.0, "beta_hinge": 0.0},
    "rkl-only": {"lambda_ce": 0.0, "lambda_rkl": 1.0, "beta_hinge": 0.0},
    "mixed": {"lambda_ce": 0.5, "lambda_rkl": 0.5, "beta_hinge": 0.0},
    "full-opd": {"lambda_ce": 0.4, "lambda_rkl": 0.5, "beta_hinge": 0.1},
}

# Per-ablation LR grids (optional)
LR_GRIDS = {
    "baseline": [5e-5, 1e-4, 2e-4],
    "rkl-only": [3e-5, 5e-5, 1e-4],  # More conservative for pure RKL
    "mixed": [5e-5, 1e-4, 2e-4],
    "full-opd": [3e-5, 5e-5, 1e-4],
}

def make_sweep_configs(
    ablations: list[str],
    base_overrides: dict = None,
    lr_sweep: bool = True,
) -> list[dict]:
    """Generate configs for all (ablation, lr) combinations."""
    configs = []
    for name in ablations:
        ablation = ABLATIONS[name]
        lrs = LR_GRIDS[name] if lr_sweep else [LR_GRIDS[name][1]]  # middle LR
        for lr in lrs:
            config = {**BASE_CONFIG, **ablation, "learning_rate": lr}
            if base_overrides:
                config.update(base_overrides)
            config["output_dir"] = f"checkpoints/{name}_lr{lr}"
            config["wandb_name"] = f"{name}_lr{lr}"
            configs.append(config)
    return configs
```

### Sweep Execution

```python
@app.local_entrypoint()
def sweep(
    # Ablation selection (flexible)
    ablations: str = "all",           # comma-separated or "all"
    lr_sweep: bool = True,            # sweep all LRs per ablation
    ablation_lrs: str = None,         # explicit: "baseline:1e-4,full-opd:5e-5"

    sequential: bool = False,

    # Base overrides (apply to all runs)
    target_model: str = None,
    preprocessed: str = None,
    num_epochs: int = None,
):
    """Run ablation sweep with flexible configuration.

    Examples:
        # All ablations, all LRs (full grid)
        modal run modal_train.py sweep

        # Single ablation, all LRs
        modal run modal_train.py sweep --ablations baseline

        # Multiple ablations, no LR sweep (use default LR each)
        modal run modal_train.py sweep --ablations baseline,full-opd --no-lr-sweep

        # Explicit ablation:LR pairs (most flexible)
        modal run modal_train.py sweep --ablation-lrs "baseline:1e-4,full-opd:5e-5,mixed:1e-4"
    """
    if ablation_lrs:
        # Explicit mode: parse "ablation:lr,ablation:lr,..."
        configs = []
        for pair in ablation_lrs.split(","):
            name, lr = pair.split(":")
            config = {**BASE_CONFIG, **ABLATIONS[name], "learning_rate": float(lr)}
            configs.append(config)
    else:
        # Grid mode: ablations × LRs
        ablation_names = list(ABLATIONS.keys()) if ablations == "all" else ablations.split(",")
        configs = make_sweep_configs(ablation_names, lr_sweep=lr_sweep)

    # Apply base overrides to all configs
    overrides = {k: v for k, v in locals().items() if v is not None and k not in [...]}
    for cfg in configs:
        cfg.update(overrides)
        # Auto-generate output_dir and wandb_name
        ...

    if sequential:
        results = [train_run.remote(**cfg) for cfg in configs]
    else:
        handles = [train_run.spawn(**cfg) for cfg in configs]
        results = modal.FunctionCall.gather(*handles)

    return results
```

### Key Points

- **Simple dict merge**: `{**base, **ablation, **overrides}` - no complex config system
- **Parallel by default**: Use `.spawn()` + `.gather()` for concurrent runs
- **CLI overrides**: Can override any base param from command line
- **Flexible ablation selection**:
  - `--ablations baseline,full-opd` - run specific ablations
  - `--no-lr-sweep` - use default LR per ablation instead of grid
  - `--ablation-lrs "baseline:1e-4,full-opd:5e-5"` - explicit ablation:LR pairs
- **Naming convention**: Auto-generate output dirs and wandb names from ablation + LR

---

## 3. Dataset Regeneration with Autoscaling

### Current State

Single Modal function runs N SGLang servers on one node (8 GPUs max), then runs `regenerate_train_data.py` against them. Limited to single-node throughput.

### Target State

Modal-native autoscaling with **cache-aware routing**:
- Each container runs 1 SGLang server
- Modal scales containers based on workload
- **Cache-aware routing** ensures multi-turn conversation samples route to the same container (avoids KV cache misses on prefill)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (modal_data.py)                 │
│  - Loads dataset samples                                    │
│  - Submits samples to regen_server via .map()               │
│  - Collects results + checkpoints progress                  │
└─────────────────────────────────────────────────────────────┘
                              │
              (Modal routes requests, cache-aware)
                              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ regen_server│  │ regen_server│  │ regen_server│  ... (Modal autoscales)
│ (1 SGLang)  │  │ (1 SGLang)  │  │ (1 SGLang)  │
│ GPU: H100   │  │ GPU: H100   │  │ GPU: H100   │
│ KV cache    │  │ KV cache    │  │ KV cache    │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Design: Client Side

**Key constraints from Modal:**
- `.map()` limited to 1000 concurrent inputs - not suitable for large datasets
- `.spawn()` allows up to 1 million pending inputs - required for scale
- Client function can be preempted during long-running jobs - must handle gracefully

The client uses `.spawn()` with handle persistence for preemption recovery:

```python
import time
import json
from modal import FunctionCall

REGEN_RATE_LIMIT = 200  # inputs/sec

@app.function(
    volumes={DATASET_PATH: dataset_vol},
    timeout=86400,  # 24h for large datasets
)
def regenerate_dataset(
    model: str,
    input_file: str,
    output_file: str,
    tp_size: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    rate_limit: int = REGEN_RATE_LIMIT,
) -> str:
    """Regenerate dataset using autoscaling SGLang servers.

    Uses .spawn() for scale (up to 1M pending) and persists handles
    to volume for preemption recovery.
    """
    dataset_vol.reload()
    data = load_jsonl(input_file)

    # State files (persisted to volume for preemption recovery)
    state_dir = f"{output_file}.state"
    handles_file = f"{state_dir}/handles.json"      # sample_id -> function_call_id
    completed_file = f"{state_dir}/completed.json"  # sample_id -> result

    os.makedirs(state_dir, exist_ok=True)

    # Load existing state (handles preemption recovery)
    handles = load_json(handles_file) if exists(handles_file) else {}
    completed = load_json(completed_file) if exists(completed_file) else {}

    # Phase 1: Submit pending samples via .spawn()
    pending = [s for s in data if s["id"] not in completed and s["id"] not in handles]

    for i, sample in enumerate(pending):
        # Rate limit: sleep between submissions
        if i > 0 and i % rate_limit == 0:
            time.sleep(1.0)

        # Spawn async task
        fc = regen_server.spawn(
            sample=sample,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        handles[sample["id"]] = fc.object_id

        # Checkpoint handles periodically (for preemption recovery)
        if i % 1000 == 0:
            save_json(handles_file, handles)
            dataset_vol.commit()

    # Final handles checkpoint
    save_json(handles_file, handles)
    dataset_vol.commit()

    # Phase 2: Collect results from spawned tasks
    outstanding = {sid: fid for sid, fid in handles.items() if sid not in completed}

    while outstanding:
        for sample_id, fc_id in list(outstanding.items()):
            fc = FunctionCall.from_id(fc_id)

            try:
                # Non-blocking check (or short timeout)
                result = fc.get(timeout=0.1)
                completed[sample_id] = result
                del outstanding[sample_id]
            except TimeoutError:
                continue  # Still running
            except Exception as e:
                log_error(sample_id, e)
                completed[sample_id] = {"error": str(e)}
                del outstanding[sample_id]

        # Checkpoint progress periodically
        if len(completed) % 100 == 0:
            save_json(completed_file, completed)
            dataset_vol.commit()

        # Brief sleep to avoid tight polling
        time.sleep(0.5)

    # Final write
    write_jsonl(output_file, [completed[s["id"]] for s in data if s["id"] in completed])
    dataset_vol.commit()

    # Cleanup state files
    shutil.rmtree(state_dir)
    dataset_vol.commit()

    return output_file
```

**Preemption recovery:**
- If client is preempted mid-job, state is persisted to volume
- On restart, loads existing handles and completed results
- Uses `FunctionCall.from_id()` to reconnect to in-flight tasks
- Spawned server tasks continue running even if client dies

**Rate limiting:**
- Submits at most `rate_limit` tasks per second
- Prevents overwhelming Modal's queue
- 200/sec default = 720k samples/hour submission rate

### Design: Server Side (TBD)

The regen server is a Modal function with SGLang. **Cache-aware routing configuration is TBD** - this will be designed hands-on using Modal's undocumented cache-aware routing system.

```python
@app.function(
    gpu="H100",  # or "H100:2" for larger models
    image=sglang_image,
    secrets=[hf_secret],
    timeout=600,  # 10min per sample max
    # TODO: cache-aware routing config (Modal-specific, TBD)
)
def regen_server(
    sample: dict,  # Single conversation sample
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """Regenerate a single sample using local SGLang server."""
    # 1. Start/reuse SGLang server on localhost (container-local)
    # 2. Process sample via OpenAI-compatible API
    # 3. Return regenerated conversation
    ...
```

**Server-side design notes** (to be finalized during implementation):
- Container startup: SGLang server initialization on first request
- KV cache reuse: Multi-turn samples should route to same container
- Health checks: Handle server startup failures gracefully
- GPU scaling: `tp_size` determines GPU count per container

### Fault Tolerance

- **Preemption-safe**: Client persists handles + completed results to volume
- **Resume capability**: On restart/re-run, reconnects to in-flight tasks via `FunctionCall.from_id()`
- **Server task independence**: Spawned tasks continue running even if client is preempted
- **Error handling**: Log failed samples, continue with remaining tasks
- **State cleanup**: Removes state files only after successful completion

### GPU Scaling by Model Size

- **≤8B models**: `tp_size=1`, `gpu="H100"`
- **8B-32B models**: `tp_size=2`, `gpu="H100:2"`
- **70B+ models**: `tp_size=4` or `tp_size=8`, fewer parallel containers

---

## 4. SGLang for Online Training (Teacher Queries)

### Current State (Verified)

The SGLang teacher engine is **co-located by default**:
- Runs in the same Python process as training
- Uses `torch.cuda.current_device()` - same GPU
- Shares NCCL distributed backend via `get_specforge_tp_group()`
- No separate server process (`nccl_port=None`)

### Implication for Modal

**No changes needed** - the existing `train_run()` function already handles this correctly because SpecForge's `train_eagle3.py` manages SGLang internally.

Uncolocated teacher engines (separate containers) would require a significant SpecForge rewrite and are **completely out of scope** for this project.

---

## 5. File Organization

### Final Structure

```
examples/modal/
├── common.py               # Shared: volumes, images, secrets, utilities
├── modal_train.py          # train_run(), train_run_single_gpu(), sweep(), sweep configs
├── modal_data.py           # prep_dataset(), regenerate_dataset(), regen_server(), preprocess_dataset()
├── test_train.py           # Tests for modal_train.py
├── test_data.py            # Tests for modal_data.py
└── README.md               # Usage examples
```

### Rationale

- **Shared common module**: `common.py` for volumes, images, secrets, and utilities (DRY but not excessively so)
- **2 main files**: Training + sweeps in one file, data processing in another
- **2 test files**: `test_data.py` and `test_train.py` for separation of concerns
- **Clear entrypoints**: Each main file has its own `@app.local_entrypoint()` for direct invocation

### Infrastructure Definitions (common.py)

```python
# Volumes
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
dataset_vol = modal.Volume.from_name("specforge-datasets", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("specforge-checkpoints", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
DATASET_PATH = "/root/.cache/specforge/datasets"
CKPT_PATH = "/root/.cache/specforge/checkpoints"

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")

# Images
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .copy_local_dir("../..", "/root/specforge", exclude=[...])
    .pip_install("specforge", "torch==2.8.0", ...)
)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129")
    .pip_install("huggingface_hub", "openai")
)
```

### Model Download

Model download is handled implicitly via HuggingFace cache volume. If a model isn't cached, it downloads on first use. No explicit `download_model()` step needed in the pipeline (simplifies UX).

---

## 6. Pipeline Steps (Independent)

Each step is independently invocable:

| Step | Function | File | Description |
|------|----------|------|-------------|
| 1. Prep | `prep_dataset()` | modal_data.py | Convert raw data to JSONL format |
| 2. Regen | `regenerate_dataset()` | modal_data.py | Regenerate with target model outputs |
| 3. Preprocess | `preprocess_dataset()` | modal_data.py | Tokenize, generate vocab mapping, cache |
| 4. Train | `train_run()` | modal_train.py | Single training run |
| 5. Sweep | `sweep()` | modal_train.py | Multiple training runs |

### Preprocess Details

`preprocess_dataset()` performs all preprocessing independent of individual training runs:

```python
@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,  # 2h for large datasets
)
def preprocess_dataset(
    input_file: str,              # JSONL from prep or regen
    target_model: str,            # For tokenizer
    output_name: str,             # Output cache name
    chat_template: str = "qwen",
    max_length: int = 2048,
    draft_vocab_size: int = 32000,
    num_proc: int = 8,
) -> str:
    """Preprocess dataset: tokenize + generate vocab mapping."""
    # 1. Load tokenizer from target model
    # 2. Tokenize conversations with chat template
    # 3. Generate loss masks (assistant spans only)
    # 4. Count token frequencies in loss-masked regions
    # 5. Generate vocab mapping (d2t, t2d)
    # 6. Save preprocessed dataset + vocab mapping to volume
    # 7. Return output path
```

**Outputs**:
- `{output_name}_dataset.pkl` - Preprocessed dataset
- `{output_name}_vocab_mapping.pt` - d2t/t2d tensors

This moves vocab mapping out of training startup (currently in `build_dataloaders()`) into explicit preprocessing.

User composes as needed:
```bash
# Full pipeline
modal run modal_data.py prep --dataset sharegpt
modal run modal_data.py regen --model Qwen/Qwen3-8B --input sharegpt_train.jsonl
modal run modal_data.py preprocess --input sharegpt_train_regenerated.jsonl \
    --target-model Qwen/Qwen3-8B --output-name sharegpt_qwen3
modal run modal_train.py train --preprocessed sharegpt_qwen3 --target-model Qwen/Qwen3-8B

# Skip regen (use original data)
modal run modal_data.py prep --dataset sharegpt
modal run modal_data.py preprocess --input sharegpt_train.jsonl \
    --target-model Qwen/Qwen3-8B --output-name sharegpt_qwen3
modal run modal_train.py train --preprocessed sharegpt_qwen3 --target-model Qwen/Qwen3-8B

# Run a sweep (assumes preprocessed data exists)
modal run modal_train.py sweep --preprocessed sharegpt_qwen3 --ablations baseline,full-opd
```

---

## 7. Implementation Steps

### Step 0: Test Infrastructure (First)

Create test files to guide implementation and verify correctness at each stage:

**`test_data.py`** - tests for modal_data.py:
```python
"""Tests for modal_data.py functions.

Run specific tests as implementation progresses:
    modal run test_data.py test-prep          # Test prep_dataset()
    modal run test_data.py test-regen         # Test regenerate_dataset()
    modal run test_data.py test-preprocess    # Test preprocess_dataset()
    modal run test_data.py test-all           # Run all data tests
"""
```

**`test_train.py`** - tests for modal_train.py:
```python
"""Tests for modal_train.py functions.

Run specific tests as implementation progresses:
    modal run test_train.py test-train        # Test train_run()
    modal run test_train.py test-sweep        # Test sweep()
    modal run test_train.py test-all          # Run all training tests
"""
```

Each test:
- Uses minimal data (10-100 samples)
- Uses smallest viable model (Qwen3-0.6B or similar)
- Verifies output format and correctness
- Runs in <5 minutes

### Step 1: Create `common.py`

- Volume definitions (hf_cache_vol, dataset_vol, ckpt_vol)
- Path constants (HF_CACHE_PATH, DATASET_PATH, CKPT_PATH)
- Secret references (hf_secret, wandb_secret)
- Image definitions (train_image, sglang_image)
- Shared utilities (load_jsonl, save_json, etc.) as needed

### Step 2: Create `modal_data.py`

- Import from common.py
- `prep_dataset()` - convert raw data to JSONL
- `preprocess_dataset()` - tokenize + vocab mapping generation
- `regenerate_dataset()` - rate-limited client with checkpointing
- `regen_server()` - server-side TBD for cache-aware routing
- Local entrypoint with CLI subcommands: `prep`, `regen`, `preprocess`

### Step 3: Create `modal_train.py`

- Import from common.py
- `train_run()` with flat kwargs
- `train_run_single_gpu()` variant
- Sweep config pattern (BASE_CONFIG, ABLATIONS, LR_GRIDS)
- `make_sweep_configs()` helper
- `sweep()` function with flexible ablation selection
- Local entrypoint with CLI subcommands: `train`, `sweep`

### Step 4: Update README.md

- Usage examples for each pipeline step
- Common workflows (full pipeline, skip regen, sweep only)
- Configuration reference

### Step 5: Archive `modal_eagle3_opd.py`

- Move current implementation to `modal_eagle3_opd.py.bak` or delete
- Ensure no references remain

---

## Open Questions (Resolved)

1. ~~Chunk size for regen~~ → Modal handles routing, no explicit chunking
2. ~~Max workers for regen~~ → Modal autoscales automatically
3. ~~Progress persistence~~ → **Checkpoint to volume** for fault tolerance
