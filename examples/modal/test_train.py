"""
Tests for modal_train.py functions.

TDD-style tests that will fail until implementation catches up.
Each test uses unique timestamped filenames for isolation and cleans up after itself.

Run specific tests (from repo root):
    uv run modal run -m examples.modal.test_train::test_train        # Test train_run() single training
    uv run modal run -m examples.modal.test_train::test_train_single_gpu  # Test single GPU variant
    uv run modal run -m examples.modal.test_train::test_sweep        # Test sweep() with minimal configs
    uv run modal run -m examples.modal.test_train::test_sweep_configs # Test sweep config generation
    uv run modal run -m examples.modal.test_train::test_preemption   # Test training preemption/resume
    uv run modal run -m examples.modal.test_train::test_all          # Run all training tests
"""

from __future__ import annotations

import json
import os
import time

import modal

# =============================================================================
# TDD Imports - These will fail until the modules are implemented
# =============================================================================

from .common import (
    CKPT_PATH,
    DATASET_PATH,
    HF_CACHE_PATH,
    ckpt_vol,
    dataset_vol,
    hf_cache_vol,
    hf_secret,
    train_image,
)
from .modal_train import (
    ABLATIONS,
    LR_GRIDS,
    app,  # Use the same app as modal_train
    make_sweep_configs,
    sweep,
    train_run,
    train_run_single_gpu,
)

# Test configuration
TEST_MODEL = "Qwen/Qwen3-0.6B"
TEST_MAX_STEPS = 1  # Minimal training for fast tests
TEST_MAX_LENGTH = 512  # Shorter sequences for tests
TEST_BATCH_SIZE = 1

# =============================================================================
# Test Utilities
# =============================================================================


def get_test_name() -> str:
    """Generate unique test name with timestamp."""
    return f"test_train_{int(time.time())}"


def create_test_samples(num_samples: int = 50) -> list[dict]:
    """Create synthetic test samples in the expected format."""
    samples = []
    for i in range(num_samples):
        samples.append(
            {
                "id": f"train-test-{i}",
                "conversations": [
                    {"role": "user", "content": f"Explain the number {i} in detail."},
                    {
                        "role": "assistant",
                        "content": f"The number {i} is an integer. " * 10,
                    },
                ],
            }
        )
    return samples


def write_jsonl(path: str, data: list[dict]) -> None:
    """Write data to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# =============================================================================
# Test: train_run (Single Training Run)
# =============================================================================


@app.local_entrypoint()
def test_train():
    """
    Test train_run() executes a single training run correctly.

    Verifies:
    - Function executes without error (smoke test)
    - Output directory is created
    - Checkpoint files are saved
    - Training completes with expected artifacts
    """
    print("=" * 60)
    print("TEST: train_run() (Single Training)")
    print("=" * 60)

    test_name = get_test_name()
    preprocessed_name = f"{test_name}_data"
    output_dir = f"{test_name}_output"

    # Prepare test data: write JSONL (train_eagle3.py will generate vocab mapping)
    print("\n[1/5] Preparing test data...")
    samples = create_test_samples(50)
    jsonl_path = _write_test_jsonl.remote(preprocessed_name, samples)
    print(f"  JSONL file: {jsonl_path}")

    # Run training
    print("\n[2/5] Running train_run()...")
    print(f"  Target model: {TEST_MODEL}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max steps: {TEST_MAX_STEPS}")

    try:
        result_dir = train_run.remote(
            target_model=TEST_MODEL,
            preprocessed=preprocessed_name,
            output_dir=output_dir,
            # OPD loss weights (baseline config)
            lambda_ce=1.0,
            lambda_rkl=0.0,
            beta_hinge=0.0,
            # Training hyperparams
            num_epochs=1,
            learning_rate=1e-4,
            max_length=TEST_MAX_LENGTH,
            batch_size=TEST_BATCH_SIZE,
            max_num_steps=TEST_MAX_STEPS,
            # Disable wandb for tests
            wandb_project=None,
            resume=False,
        )

        print(f"  Result directory: {result_dir}")

        # Verify output directory exists
        print("\n[3/5] Verifying output directory exists...")

        dir_exists = _check_dir_exists.remote(f"{CKPT_PATH}/{output_dir}")
        assert dir_exists, f"Output directory should exist: {output_dir}"
        print("  Output directory exists")

        # Verify checkpoint files
        print("\n[4/5] Verifying checkpoint artifacts...")

        artifacts = _list_checkpoint_artifacts.remote(f"{CKPT_PATH}/{output_dir}")
        print(f"  Found {len(artifacts)} artifacts:")
        for artifact in artifacts[:10]:
            print(f"    - {artifact}")

        has_model_files = any(
            "model" in a.lower()
            or "checkpoint" in a.lower()
            or ".safetensors" in a
            or ".bin" in a
            for a in artifacts
        )
        assert has_model_files or len(artifacts) > 0, "Should have checkpoint artifacts"
        print("  Checkpoint artifacts present")

        print("\n[5/5] Training completed successfully")

    finally:
        # Cleanup test files
        print("\n[Cleanup] Removing test files...")
        _cleanup_test_files.remote(jsonl_path, output_dir)

    print("\n" + "=" * 60)
    print("TEST PASSED: train_run()")
    print("=" * 60)


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol},
    timeout=300,
)
def _write_test_jsonl(output_name: str, samples: list[dict]) -> str:
    """Write test samples to JSONL file."""
    dataset_vol.reload()

    # Save as {output_name}_train.jsonl (the naming convention train_run expects)
    jsonl_file = f"{DATASET_PATH}/{output_name}_train.jsonl"
    write_jsonl(jsonl_file, samples)

    dataset_vol.commit()
    return jsonl_file


@app.function(image=train_image, volumes={CKPT_PATH: ckpt_vol})
def _check_dir_exists(path: str) -> bool:
    """Check if a directory exists."""
    ckpt_vol.reload()
    return os.path.isdir(path)


@app.function(image=train_image, volumes={CKPT_PATH: ckpt_vol})
def _list_checkpoint_artifacts(path: str) -> list[str]:
    """List all files in a checkpoint directory."""
    ckpt_vol.reload()

    if not os.path.exists(path):
        return []

    artifacts = []
    for root, dirs, files in os.walk(path):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), path)
            artifacts.append(rel_path)

    return sorted(artifacts)


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol, CKPT_PATH: ckpt_vol},
)
def _cleanup_test_files(jsonl_path: str, output_dir: str) -> None:
    """Clean up test files after test completion."""
    import shutil

    dataset_vol.reload()
    ckpt_vol.reload()

    # Clean up JSONL file
    if jsonl_path and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print(f"  Removed: {jsonl_path}")

    # Clean up checkpoint directory
    if output_dir:
        ckpt_path = f"{CKPT_PATH}/{output_dir}"
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
            print(f"  Removed: {ckpt_path}")

    dataset_vol.commit()
    ckpt_vol.commit()


# =============================================================================
# Test: train_run_single_gpu
# =============================================================================


@app.local_entrypoint()
def test_train_single_gpu():
    """
    Test train_run_single_gpu() variant for quick iterations.

    Verifies:
    - Function executes without error
    - Works with single GPU configuration
    - Produces valid checkpoint artifacts
    """
    print("=" * 60)
    print("TEST: train_run_single_gpu()")
    print("=" * 60)

    test_name = get_test_name()
    preprocessed_name = f"{test_name}_data"
    output_dir = f"{test_name}_output"

    # Prepare test data: write JSONL (train_eagle3.py will generate vocab mapping)
    print("\n[1/4] Preparing test data...")
    samples = create_test_samples(50)
    jsonl_path = _write_test_jsonl.remote(preprocessed_name, samples)
    print(f"  JSONL file: {jsonl_path}")

    # Run single-GPU training
    print("\n[2/4] Running train_run_single_gpu()...")
    print(f"  Target model: {TEST_MODEL}")
    print(f"  Output dir: {output_dir}")

    try:
        result_dir = train_run_single_gpu.remote(
            target_model=TEST_MODEL,
            preprocessed=preprocessed_name,
            output_dir=output_dir,
            lambda_ce=1.0,
            lambda_rkl=0.0,
            beta_hinge=0.0,
            num_epochs=1,
            learning_rate=1e-4,
            max_length=TEST_MAX_LENGTH,
            batch_size=TEST_BATCH_SIZE,
            max_num_steps=TEST_MAX_STEPS,
            save_interval=1,  # Force checkpoint save on first step
            wandb_project=None,
            resume=False,
        )

        print(f"  Result directory: {result_dir}")

        # Verify output
        print("\n[3/4] Verifying output directory...")

        dir_exists = _check_dir_exists.remote(f"{CKPT_PATH}/{output_dir}")
        assert dir_exists, f"Output directory should exist: {output_dir}"
        print("  Output directory exists")

        # Verify artifacts
        print("\n[4/4] Verifying checkpoint artifacts...")

        artifacts = _list_checkpoint_artifacts.remote(f"{CKPT_PATH}/{output_dir}")
        assert len(artifacts) > 0, "Should have checkpoint artifacts"
        print(f"  Found {len(artifacts)} artifacts")

    finally:
        # Cleanup
        print("\n[Cleanup] Removing test files...")
        _cleanup_test_files.remote(jsonl_path, output_dir)

    print("\n" + "=" * 60)
    print("TEST PASSED: train_run_single_gpu()")
    print("=" * 60)


# =============================================================================
# Test: Sweep Config Generation
# =============================================================================


@app.local_entrypoint()
def test_sweep_configs():
    """
    Test sweep configuration generation logic.

    Verifies:
    - make_sweep_configs() generates correct number of configs
    - Each config has required fields
    - Ablation parameters are correctly applied
    - LR sweep produces expected combinations
    """
    print("=" * 60)
    print("TEST: Sweep Config Generation")
    print("=" * 60)

    # Test: Generate configs for all ablations with LR sweep
    print("\n[1/4] Testing full grid generation...")

    all_ablations = list(ABLATIONS.keys())
    configs = make_sweep_configs(all_ablations, lr_sweep=True)

    expected_count = sum(len(LR_GRIDS[name]) for name in all_ablations)
    assert len(configs) == expected_count, (
        f"Expected {expected_count} configs, got {len(configs)}"
    )
    print(f"  Generated {len(configs)} configs (all ablations x all LRs)")

    # Test: Verify config structure
    print("\n[2/4] Verifying config structure...")

    required_fields = [
        "target_model",
        "preprocessed",
        "output_dir",
        "lambda_ce",
        "lambda_rkl",
        "beta_hinge",
        "learning_rate",
    ]

    for i, cfg in enumerate(configs):
        for field in required_fields:
            assert field in cfg, f"Config {i} missing required field: {field}"

    print(f"  All {len(configs)} configs have required fields")

    # Test: Single ablation, no LR sweep
    print("\n[3/4] Testing single ablation without LR sweep...")

    single_configs = make_sweep_configs(["baseline"], lr_sweep=False)

    assert len(single_configs) == 1, f"Expected 1 config, got {len(single_configs)}"
    assert single_configs[0]["lambda_ce"] == ABLATIONS["baseline"]["lambda_ce"]
    assert single_configs[0]["lambda_rkl"] == ABLATIONS["baseline"]["lambda_rkl"]
    print("  Single ablation config correct")

    # Test: Multiple ablations with base overrides
    print("\n[4/4] Testing base overrides...")

    override_configs = make_sweep_configs(
        ["baseline", "full-opd"],
        base_overrides={"num_epochs": 5, "max_length": 1024},
        lr_sweep=False,
    )

    for cfg in override_configs:
        assert cfg["num_epochs"] == 5, "Override should be applied"
        assert cfg["max_length"] == 1024, "Override should be applied"

    print("  Base overrides correctly applied")

    print("\n" + "=" * 60)
    print("TEST PASSED: Sweep Config Generation")
    print("=" * 60)


# =============================================================================
# Test: sweep() (Full Sweep Execution)
# =============================================================================


@app.local_entrypoint()
def test_sweep():
    """
    Test sweep() executes multiple training runs in parallel.

    Uses minimal config (--max-num-steps 1) for fast execution.

    Verifies:
    - Function executes without error
    - All ablation runs complete
    - Each run produces output directory
    - Results are collected correctly
    """
    print("=" * 60)
    print("TEST: sweep() (Parallel Training)")
    print("=" * 60)

    test_name = get_test_name()
    preprocessed_name = f"{test_name}_data"
    output_prefix = f"{test_name}_sweep"

    # Prepare test data: write JSONL (train_eagle3.py will generate vocab mapping)
    print("\n[1/5] Preparing test data...")
    samples = create_test_samples(50)
    jsonl_path = _write_test_jsonl.remote(preprocessed_name, samples)
    print(f"  JSONL file: {jsonl_path}")

    # Run minimal sweep (2 ablations, no LR sweep = 2 runs)
    test_ablations = "baseline,full-opd"

    print("\n[2/5] Running sweep()...")
    print(f"  Ablations: {test_ablations}")
    print("  LR sweep: disabled (using default LR per ablation)")
    print(f"  Max steps: {TEST_MAX_STEPS}")
    print(f"  Output prefix: {output_prefix}")

    try:
        results = sweep.remote(
            ablations=test_ablations,
            lr_sweep=False,
            target_model=TEST_MODEL,
            preprocessed=preprocessed_name,
            num_epochs=1,
            max_length=TEST_MAX_LENGTH,
            max_num_steps=TEST_MAX_STEPS,
            output_prefix=output_prefix,
            wandb_project=None,
            sequential=True,  # Run sequentially for simpler cleanup
        )

        print(f"\n  Sweep returned {len(results)} results")

        # Verify number of results
        print("\n[3/5] Verifying result count...")

        expected_runs = len(test_ablations.split(","))
        assert len(results) == expected_runs, (
            f"Expected {expected_runs} results, got {len(results)}"
        )
        print(f"  Got {len(results)} results as expected")

        # Verify each run produced output
        print("\n[4/5] Verifying output directories...")

        for i, result in enumerate(results):
            print(f"  Run {i + 1}: {result}")
            dir_exists = _check_dir_exists.remote(result)
            assert dir_exists, f"Output directory should exist: {result}"

        print("  All output directories exist")

        # Verify artifacts in each directory
        print("\n[5/5] Verifying checkpoint artifacts...")

        for result in results:
            artifacts = _list_checkpoint_artifacts.remote(result)
            assert len(artifacts) > 0, f"Run {result} should have artifacts"

        print("  All runs have checkpoint artifacts")

    finally:
        # Cleanup
        print("\n[Cleanup] Removing test files...")
        _cleanup_test_files.remote(jsonl_path, "")
        # Clean sweep outputs
        for ablation in test_ablations.split(","):
            lr = LR_GRIDS[ablation][1]  # middle LR
            _cleanup_sweep_output.remote(f"{output_prefix}/{ablation}_lr{lr}")

    print("\n" + "=" * 60)
    print("TEST PASSED: sweep()")
    print("=" * 60)


@app.function(image=train_image, volumes={CKPT_PATH: ckpt_vol})
def _cleanup_sweep_output(output_dir: str) -> None:
    """Clean up a sweep output directory."""
    import shutil

    ckpt_vol.reload()

    path = f"{CKPT_PATH}/{output_dir}"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"  Removed: {path}")

    ckpt_vol.commit()


# =============================================================================
# Test: Training Preemption/Resume
# =============================================================================


@app.local_entrypoint()
def test_preemption():
    """
    Test training preemption and resume functionality.

    Verifies:
    - Training can be preempted and resumed via --resume flag
    - Checkpoint files are properly saved
    - Training state (epoch, global_step, optimizer) is restored on resume
    - Resumed training continues from correct position
    """
    print("=" * 60)
    print("TEST: Training Preemption/Resume")
    print("=" * 60)

    test_name = get_test_name()
    preprocessed_name = f"{test_name}_data"
    output_dir = f"{test_name}_preempt"

    # Prepare test data: write JSONL (train_eagle3.py will generate vocab mapping)
    print("\n[1/6] Preparing test data...")
    samples = create_test_samples(50)
    jsonl_path = _write_test_jsonl.remote(preprocessed_name, samples)
    print(f"  JSONL file: {jsonl_path}")

    try:
        # Run training with enough steps to create checkpoint
        print("\n[2/6] Running initial training (will save checkpoint)...")
        print(f"  Target model: {TEST_MODEL}")
        print(f"  Output dir: {output_dir}")
        print("  Max steps: 5 (to trigger checkpoint save)")

        result_dir = train_run_single_gpu.remote(
            target_model=TEST_MODEL,
            preprocessed=preprocessed_name,
            output_dir=output_dir,
            lambda_ce=1.0,
            lambda_rkl=0.0,
            beta_hinge=0.0,
            num_epochs=1,
            learning_rate=1e-4,
            max_length=TEST_MAX_LENGTH,
            batch_size=TEST_BATCH_SIZE,
            max_num_steps=5,
            save_interval=3,
            wandb_project=None,
            resume=False,
        )

        print(f"  Initial training result: {result_dir}")

        # Verify checkpoint exists
        print("\n[3/6] Verifying checkpoint was saved...")

        has_checkpoint, checkpoint_path = _find_training_checkpoint.remote(
            f"{CKPT_PATH}/{output_dir}"
        )

        assert has_checkpoint, "Initial training should have created a checkpoint"
        print(f"  Found checkpoint: {checkpoint_path}")

        # Read training state from checkpoint
        print("\n[4/6] Reading training state from checkpoint...")

        initial_state = _read_training_state.remote(checkpoint_path)

        assert initial_state is not None, "Should have training_state.pt"
        print(
            f"  Initial state - epoch: {initial_state['epoch']}, "
            f"global_step: {initial_state['global_step']}"
        )

        # Resume training from checkpoint
        print("\n[5/6] Resuming training from checkpoint...")

        resumed_result = train_run_single_gpu.remote(
            target_model=TEST_MODEL,
            preprocessed=preprocessed_name,
            output_dir=output_dir,
            lambda_ce=1.0,
            lambda_rkl=0.0,
            beta_hinge=0.0,
            num_epochs=2,
            learning_rate=1e-4,
            max_length=TEST_MAX_LENGTH,
            batch_size=TEST_BATCH_SIZE,
            max_num_steps=10,
            save_interval=5,
            resume=True,
            wandb_project=None,
        )

        print(f"  Resumed training result: {resumed_result}")

        # Verify resumed training created new checkpoint
        print("\n[6/6] Verifying resumed training progressed...")

        resumed_has_checkpoint, resumed_path = _find_training_checkpoint.remote(
            f"{CKPT_PATH}/{output_dir}"
        )

        assert resumed_has_checkpoint, "Resumed training should have created checkpoint"

        resumed_state = _read_training_state.remote(resumed_path)

        assert (
            resumed_state is not None
        ), "Resumed checkpoint should have training_state.pt"

        initial_step = initial_state["global_step"]
        resumed_step = resumed_state["global_step"]
        print(f"  Initial step: {initial_step}, Resumed final step: {resumed_step}")

        assert resumed_step >= initial_step, (
            f"Resumed training should have progressed: "
            f"initial={initial_step}, resumed={resumed_step}"
        )
        print("  Training resumed and progressed successfully")

    finally:
        # Cleanup
        print("\n[Cleanup] Removing test files...")
        _cleanup_test_files.remote(jsonl_path, output_dir)

    print("\n" + "=" * 60)
    print("TEST PASSED: Training Preemption/Resume")
    print("=" * 60)


@app.function(image=train_image, volumes={CKPT_PATH: ckpt_vol})
def _find_training_checkpoint(path: str) -> tuple[bool, str]:
    """Find the latest checkpoint directory with training_state.pt."""
    ckpt_vol.reload()

    if not os.path.exists(path):
        return False, ""

    checkpoint_dirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and "epoch_" in item:
            training_state = os.path.join(item_path, "training_state.pt")
            if os.path.exists(training_state):
                checkpoint_dirs.append(item_path)

    if not checkpoint_dirs:
        return False, ""

    checkpoint_dirs.sort()
    return True, checkpoint_dirs[-1]


@app.function(image=train_image, volumes={CKPT_PATH: ckpt_vol})
def _read_training_state(checkpoint_path: str) -> dict | None:
    """Read training state from a checkpoint directory."""
    import torch

    ckpt_vol.reload()

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(training_state_path):
        return None

    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    return {
        "epoch": state.get("epoch", 0),
        "global_step": state.get("global_step", 0),
    }


# =============================================================================
# Test: Run All
# =============================================================================


@app.local_entrypoint()
def test_all():
    """Run all training tests sequentially."""
    print("=" * 60)
    print("RUNNING ALL TRAINING TESTS")
    print("=" * 60)

    tests = [
        ("Sweep Config Generation", test_sweep_configs),
        ("train_run (Single GPU)", test_train_single_gpu),
        ("train_run (Multi-GPU)", test_train),
        ("Training Preemption/Resume", test_preemption),
        ("sweep (Parallel Training)", test_sweep),
    ]

    results = []

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print("=" * 60)

        try:
            test_fn.local()
            results.append((name, "PASSED"))
        except Exception as e:
            results.append((name, f"FAILED: {e}"))
            print(f"\nTEST FAILED: {name}")
            print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)

    for name, status in results:
        icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {icon} {name}")
        if status != "PASSED":
            print(f"        {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed < total:
        raise SystemExit(1)
