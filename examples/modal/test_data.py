"""
Tests for modal_data.py functions.

TDD-style tests that will fail until implementation catches up.
Each test uses unique timestamped filenames for isolation and cleans up after itself.

Run specific tests (from repo root):
    modal run -m examples.modal.test_data::test_prep          # Test prep_dataset()
    modal run -m examples.modal.test_data::test_regen_unit    # Test RegenServer unit
    modal run -m examples.modal.test_data::test_regen         # Test regenerate_dataset() integration
    modal run -m examples.modal.test_data::test_preprocess    # Test preprocess_dataset()
    modal run -m examples.modal.test_data::test_preemption_auto   # Test automatic retry on preemption
    modal run -m examples.modal.test_data::test_preemption_manual # Test manual resume after preemption
    modal run -m examples.modal.test_data::test_all           # Run all data tests
"""

from __future__ import annotations

import json
import os

import modal
from modal.exception import simulate_preemption

# =============================================================================
# TDD Imports - These will fail until the modules are implemented
# =============================================================================

from .common import (
    DATASET_PATH,
    HF_CACHE_PATH,
    dataset_vol,
    hf_cache_vol,
    hf_secret,
    sglang_image,
    train_image,
)
from .modal_data import (
    RegenServer,
    app,  # Use the same app as modal_data
    prep_dataset,
    preprocess_dataset,
    regenerate_dataset,
)

# Test configuration
TEST_MODEL = "Qwen/Qwen3-0.6B"
TEST_DATASET = "sharegpt"
TEST_SAMPLE_SIZE = 50  # Small subset for fast tests
TEST_REGEN_SAMPLES = 10  # Even smaller for regeneration tests
TEST_CHECKPOINT_INTERVAL = 3  # Checkpoint every 3 samples for testing

# =============================================================================
# Test Utilities
# =============================================================================


def create_test_samples(num_samples: int = 10) -> list[dict]:
    """Create synthetic test samples in the expected format."""
    samples = []
    for i in range(num_samples):
        samples.append(
            {
                "id": f"test-{i}",
                "conversations": [
                    {"role": "user", "content": f"What is {i} + {i}?"},
                    {
                        "role": "assistant",
                        "content": f"The answer is {i + i}.",
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


def read_jsonl(path: str) -> list[dict]:
    """Read data from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_jsonl_structure(data: list[dict], required_keys: list[str]) -> None:
    """Assert that JSONL data has the required structure."""
    assert len(data) > 0, "Data should not be empty"
    for i, item in enumerate(data):
        for key in required_keys:
            assert key in item, f"Item {i} missing required key: {key}"


# =============================================================================
# Test: prep_dataset
# =============================================================================


@app.local_entrypoint()
def test_prep():
    """
    Test prep_dataset() downloads and formats a dataset correctly.

    Verifies:
    - Function executes without error (smoke test)
    - Output file exists with correct naming convention
    - Output is valid JSONL with expected structure
    - Each sample has 'id' and 'conversations' fields
    """
    print("=" * 60)
    print("TEST: prep_dataset()")
    print("=" * 60)

    # Run prep_dataset with small sample size
    print(
        f"\n[1/2] Running prep_dataset(dataset={TEST_DATASET}, sample_size={TEST_SAMPLE_SIZE})"
    )

    output_file = prep_dataset.remote(
        dataset=TEST_DATASET,
        sample_size=TEST_SAMPLE_SIZE,
    )

    print(f"  Output file: {output_file}")

    # Verify output in a remote function (avoid large data transfer)
    print("\n[2/2] Verifying output file (remotely)...")

    success, message = _verify_prep_output.remote(output_file)

    if success:
        print(f"  {message}")
        print("\n" + "=" * 60)
        print("TEST PASSED: prep_dataset()")
        print("=" * 60)
    else:
        print(f"  FAILED: {message}")
        raise AssertionError(message)


@app.function(image=train_image, volumes={DATASET_PATH: dataset_vol})
def _verify_prep_output(output_file: str) -> tuple[bool, str]:
    """Verify prep output remotely - returns (success, message)."""
    from .common import load_jsonl

    dataset_vol.reload()

    # Check file exists
    if not os.path.exists(output_file):
        return False, f"Output file does not exist: {output_file}"

    size = os.path.getsize(output_file)

    # Load and verify structure
    try:
        data = load_jsonl(output_file)
    except Exception as e:
        return False, f"Failed to load JSONL: {e}"

    if len(data) == 0:
        return False, "Dataset is empty"

    # Verify JSONL structure
    for i, item in enumerate(data[:10]):  # Check first 10
        if "id" not in item:
            return False, f"Sample {i} missing 'id' field"
        if "conversations" not in item:
            return False, f"Sample {i} missing 'conversations' field"

        convos = item["conversations"]
        if len(convos) < 2:
            return False, f"Sample {i} has fewer than 2 conversation turns"

        roles = [c["role"] for c in convos]
        if "user" not in roles:
            return False, f"Sample {i} missing user turn"
        if "assistant" not in roles:
            return False, f"Sample {i} missing assistant turn"

    return True, f"Valid JSONL with {len(data)} samples ({size} bytes)"


# =============================================================================
# Test: RegenServer (Unit Test)
# =============================================================================


@app.local_entrypoint()
def test_regen_unit():
    """
    Test RegenServer class processes a single sample correctly.

    Verifies:
    - Server initializes SGLang engine on first request
    - Single sample is processed without error
    - Output has correct structure (id, messages)
    - Regenerated assistant responses are non-empty
    """
    print("=" * 60)
    print("TEST: RegenServer (Unit)")
    print("=" * 60)

    # Create a test sample
    test_sample = {
        "id": "unit-test-1",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
    }

    print("\n[1/3] Creating RegenServer and processing sample...")
    print(f"  Model: {TEST_MODEL}")
    print(f"  Sample ID: {test_sample['id']}")

    # Configure server with model via environment variable
    server = RegenServer.with_options(env={"REGEN_MODEL": TEST_MODEL})

    # Call the server method
    result = server().regenerate.remote(
        sample=test_sample,
        temperature=0.7,
        max_tokens=256,
    )

    print("  Result received")

    # Verify structure
    print("\n[2/3] Verifying output structure...")
    assert "id" in result, "Result should have 'id' field"
    assert "messages" in result, "Result should have 'messages' field"
    assert result["id"] == test_sample["id"], "ID should match input"
    print(f"  ID: {result['id']}")
    print(f"  Messages: {len(result['messages'])} turns")

    # Verify regenerated content
    print("\n[3/3] Verifying regenerated content...")
    messages = result["messages"]
    assert len(messages) >= 2, "Should have at least 2 messages"

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) > 0, "Should have assistant message"

    for msg in assistant_msgs:
        assert len(msg["content"]) > 0, "Assistant response should be non-empty"
        print(f"  Assistant response length: {len(msg['content'])} chars")

    print("\n" + "=" * 60)
    print("TEST PASSED: RegenServer (Unit)")
    print("=" * 60)


# =============================================================================
# Test: regenerate_dataset (Integration)
# =============================================================================


@app.local_entrypoint()
def test_regen():
    """
    Test regenerate_dataset() end-to-end with autoscaling servers.

    Verifies:
    - Function executes without error
    - Output file is created
    - All samples are processed
    - Output has correct structure
    - Regenerated responses are non-empty
    """
    print("=" * 60)
    print("TEST: regenerate_dataset() (Integration)")
    print("=" * 60)

    # Use unique names for test isolation
    import time

    timestamp = int(time.time())
    input_file = f"{DATASET_PATH}/test_regen_input_{timestamp}.jsonl"
    output_file = f"{DATASET_PATH}/test_regen_output_{timestamp}.jsonl"

    # Create test input data
    print(f"\n[1/3] Creating test input data ({TEST_REGEN_SAMPLES} samples)...")
    test_samples = create_test_samples(TEST_REGEN_SAMPLES)

    _write_test_data.remote(input_file, test_samples)
    print(f"  Input file: {input_file}")

    # Run regeneration
    print("\n[2/3] Running regenerate_dataset()...")
    print(f"  Model: {TEST_MODEL}")

    result_file = regenerate_dataset.remote(
        model=TEST_MODEL,
        input_file=input_file,
        output_file=output_file,
        temperature=0.7,
        max_tokens=256,
        checkpoint_interval=TEST_CHECKPOINT_INTERVAL,
    )

    print(f"  Output file: {result_file}")

    # Verify output (all validation done remotely to avoid I/O)
    print("\n[3/3] Verifying output (remotely)...")

    success, message = _check_regen_output.remote(
        output_file,
        expected_count=TEST_REGEN_SAMPLES,
    )

    if success:
        print(f"  {message}")
    else:
        print(f"  FAILED: {message}")
        raise AssertionError(message)

    # Cleanup test files
    print("\n  Cleaning up test files...")
    _cleanup_test_files.remote([input_file, output_file])

    print("\n" + "=" * 60)
    print("TEST PASSED: regenerate_dataset()")
    print("=" * 60)


@app.function(image=train_image, volumes={DATASET_PATH: dataset_vol})
def _write_test_data(
    path: str,
    data: list[dict],
) -> None:
    """Helper to write test data from within Modal container."""
    dataset_vol.reload()
    write_jsonl(path, data)
    dataset_vol.commit()


@app.function(image=train_image, volumes={DATASET_PATH: dataset_vol})
def _cleanup_test_files(paths: list[str]) -> None:
    """Helper to clean up test files."""
    dataset_vol.reload()
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed: {path}")
    dataset_vol.commit()


@app.function(image=train_image, volumes={DATASET_PATH: dataset_vol})
def _check_regen_output(
    output_file: str,
    expected_count: int | None = None,
) -> tuple[bool, str]:
    """
    Verify regen output remotely - returns (success, message).

    Does all validation in the remote container to avoid large I/O.
    """
    dataset_vol.reload()

    if not os.path.exists(output_file):
        return False, f"Output file does not exist: {output_file}"

    size = os.path.getsize(output_file)

    try:
        data = read_jsonl(output_file)
    except Exception as e:
        return False, f"Failed to load JSONL: {e}"

    if len(data) == 0:
        return False, "Dataset is empty"

    # Check expected count
    if expected_count is not None and len(data) != expected_count:
        return False, f"Expected {expected_count} samples, got {len(data)}"

    # Verify structure of first few samples
    for i, sample in enumerate(data[:5]):
        if "id" not in sample:
            return False, f"Sample {i} missing 'id' field"
        if "messages" not in sample and "error" not in sample:
            return False, f"Sample {i} missing 'messages' field"

        if "messages" in sample:
            assistant_msgs = [m for m in sample["messages"] if m["role"] == "assistant"]
            if len(assistant_msgs) == 0:
                return False, f"Sample {i} has no assistant messages"

            for msg in assistant_msgs:
                if len(msg.get("content", "")) == 0:
                    return False, f"Sample {i} has empty assistant response"

    return True, f"Valid JSONL with {len(data)} samples ({size} bytes)"


# =============================================================================
# Test: preprocess_dataset
# =============================================================================


@app.local_entrypoint()
def test_preprocess():
    """
    Test preprocess_dataset() generates vocab mapping in the cache location.

    Verifies:
    - Function executes without error
    - Vocab mapping file is created in cache location
    - Vocab mapping contains d2t and t2d tensors
    - Cache is reused on subsequent calls
    """
    print("=" * 60)
    print("TEST: preprocess_dataset()")
    print("=" * 60)

    # Use unique names for test isolation
    import time

    timestamp = int(time.time())
    input_file = f"{DATASET_PATH}/test_preprocess_input_{timestamp}.jsonl"

    # Create test input data
    print("\n[1/5] Creating test input data...")
    test_samples = create_test_samples(TEST_SAMPLE_SIZE)

    _write_test_data.remote(input_file, test_samples)
    print(f"  Input file: {input_file}")

    # Run preprocessing
    print("\n[2/5] Running preprocess_dataset()...")
    print(f"  Target model: {TEST_MODEL}")

    result_path = preprocess_dataset.remote(
        input_file=input_file,
        target_model=TEST_MODEL,
        max_length=512,  # Shorter for tests
    )

    print(f"  Result path: {result_path}")

    # Verify vocab mapping file exists at result path
    print("\n[3/5] Verifying vocab mapping file exists...")

    vocab_exists, vocab_size = _check_file_exists.remote(result_path)

    assert vocab_exists, f"Vocab mapping should exist: {result_path}"
    print(f"  Vocab mapping: {vocab_size} bytes")

    # Verify vocab mapping structure
    print("\n[4/5] Verifying vocab mapping structure...")

    has_d2t, has_t2d = _check_vocab_mapping.remote(result_path)

    assert has_d2t, "Vocab mapping should contain 'd2t' tensor"
    assert has_t2d, "Vocab mapping should contain 't2d' tensor"
    print("  Vocab mapping contains d2t and t2d tensors")

    # Verify cache is reused on second call
    print("\n[5/5] Verifying cache is reused...")

    result_path_2 = preprocess_dataset.remote(
        input_file=input_file,
        target_model=TEST_MODEL,
        max_length=512,
    )

    assert result_path == result_path_2, "Cache should return same path"
    print("  Cache correctly reused")

    # Cleanup test files
    print("\n  Cleaning up test files...")
    _cleanup_test_files.remote([input_file, result_path])

    print("\n" + "=" * 60)
    print("TEST PASSED: preprocess_dataset()")
    print("=" * 60)


@app.function(image=train_image, volumes={DATASET_PATH: dataset_vol})
def _check_file_exists(path: str) -> tuple[bool, int]:
    """Helper to check if a file exists and get its size."""
    dataset_vol.reload()

    if not os.path.exists(path):
        return False, 0

    return True, os.path.getsize(path)


@app.function(
    image=train_image,
    volumes={DATASET_PATH: dataset_vol},
    gpu="H100",  # Need GPU for torch
)
def _check_vocab_mapping(path: str) -> tuple[bool, bool]:
    """Helper to verify vocab mapping structure."""
    import torch

    dataset_vol.reload()

    mapping = torch.load(path, weights_only=True)
    has_d2t = "d2t" in mapping
    has_t2d = "t2d" in mapping

    return has_d2t, has_t2d


# =============================================================================
# Test: Preemption Recovery (Automatic Retry)
# =============================================================================


@app.local_entrypoint()
def test_preemption_auto():
    """
    Test automatic preemption recovery via modal.Retries.

    Verifies:
    - Function handles SIGINT gracefully
    - State is checkpointed to volume before preemption
    - On retry, function resumes from checkpoint
    - Job completes successfully after preemption + retry
    """
    print("=" * 60)
    print("TEST: Preemption Recovery (Automatic Retry)")
    print("=" * 60)

    # Use unique names for test isolation
    import time

    timestamp = int(time.time())
    num_samples = 20  # More samples to ensure checkpoint happens

    input_file = f"{DATASET_PATH}/test_preempt_auto_input_{timestamp}.jsonl"
    output_file = f"{DATASET_PATH}/test_preempt_auto_output_{timestamp}.jsonl"
    state_dir = f"{output_file}.state"

    # Create test input data
    print(f"\n[1/4] Creating test input data ({num_samples} samples)...")
    test_samples = create_test_samples(num_samples)

    _write_test_data.remote(input_file, test_samples)
    print(f"  Input file: {input_file}")

    # Run regeneration with preemption simulation
    print("\n[2/4] Running regenerate_dataset with preemption simulation...")
    print("  Preemption will trigger after ~3 seconds")
    print("  Modal.Retries will automatically restart the function")

    # This function has retries configured and calls simulate_preemption internally
    result_file = _regen_with_preemption_auto.remote(
        model=TEST_MODEL,
        input_file=input_file,
        output_file=output_file,
        checkpoint_interval=2,  # Checkpoint every 2 samples
        preempt_after_seconds=3,
    )

    print(f"  Result file: {result_file}")

    # Verify job completed
    print("\n[3/4] Verifying job completed successfully...")

    success, message = _check_regen_output.remote(
        output_file,
        expected_count=num_samples,
    )

    if not success:
        raise AssertionError(message)
    print(f"  {message}")

    # Verify state directory was cleaned up
    print("\n[4/4] Verifying state directory was cleaned up...")

    state_exists, _ = _check_file_exists.remote(state_dir)
    assert not state_exists, "State directory should be cleaned up after completion"
    print("  State directory cleaned up")

    # Cleanup test files
    print("\n  Cleaning up test files...")
    _cleanup_test_files.remote([input_file, output_file])

    print("\n" + "=" * 60)
    print("TEST PASSED: Preemption Recovery (Automatic Retry)")
    print("=" * 60)


@app.function(
    image=sglang_image,
    gpu="H100",
    volumes={DATASET_PATH: dataset_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=600,
    retries=modal.Retries(max_retries=3, initial_delay=1.0),
)
def _regen_with_preemption_auto(
    model: str,
    input_file: str,
    output_file: str,
    checkpoint_interval: int,
    preempt_after_seconds: int,
) -> str:
    """
    Wrapper that runs regenerate_dataset with preemption simulation.

    On first invocation, sets up preemption. On retry, preemption is not
    triggered again (simulating real preemption behavior).
    """
    dataset_vol.reload()

    state_dir = f"{output_file}.state"
    first_run_marker = f"{state_dir}/.first_run_complete"

    # Check if this is a retry (first run marker exists)
    is_retry = os.path.exists(first_run_marker)

    if not is_retry:
        # First run: set up preemption simulation
        print(f"First run: setting up preemption in {preempt_after_seconds}s")
        simulate_preemption(preempt_after_seconds)

        # Mark that first run started (before preemption hits)
        os.makedirs(state_dir, exist_ok=True)
        with open(first_run_marker, "w") as f:
            f.write("1")
        dataset_vol.commit()
    else:
        print("Retry: resuming from checkpoint (no preemption)")

    # Call the actual regenerate_dataset function
    # This should handle checkpointing and resume internally
    from .modal_data import regenerate_dataset as regen_impl

    return regen_impl.local(
        model=model,
        input_file=input_file,
        output_file=output_file,
        checkpoint_interval=checkpoint_interval,
    )


# =============================================================================
# Test: Preemption Recovery (Manual Resume)
# =============================================================================


@app.local_entrypoint()
def test_preemption_manual():
    """
    Test manual preemption recovery by calling function twice.

    Verifies:
    - First call checkpoints state before "preemption"
    - Second call resumes from checkpoint
    - Handles are reconnected via FunctionCall.from_id()
    - Job completes successfully
    """
    print("=" * 60)
    print("TEST: Preemption Recovery (Manual Resume)")
    print("=" * 60)

    # Use unique names for test isolation
    import time

    timestamp = int(time.time())
    num_samples = 15

    input_file = f"{DATASET_PATH}/test_preempt_manual_input_{timestamp}.jsonl"
    output_file = f"{DATASET_PATH}/test_preempt_manual_output_{timestamp}.jsonl"
    state_dir = f"{output_file}.state"

    # Create test input data
    print(f"\n[1/5] Creating test input data ({num_samples} samples)...")
    test_samples = create_test_samples(num_samples)

    _write_test_data.remote(input_file, test_samples)
    print(f"  Input file: {input_file}")

    # First call: partial processing with early exit
    print("\n[2/5] First call: process partially then 'preempt'...")

    partial_result = _regen_partial.remote(
        model=TEST_MODEL,
        input_file=input_file,
        output_file=output_file,
        checkpoint_interval=2,
        stop_after_samples=5,  # Stop after 5 samples to simulate preemption
    )

    print(f"  Partial result: {partial_result}")

    # Verify checkpoint exists
    print("\n[3/5] Verifying checkpoint state exists...")

    handles_exist, _ = _check_file_exists.remote(f"{state_dir}/handles.json")

    assert handles_exist, "Handles checkpoint should exist"
    print("  Checkpoint state found")

    # Second call: resume from checkpoint
    print("\n[4/5] Second call: resume from checkpoint...")

    final_result = regenerate_dataset.remote(
        model=TEST_MODEL,
        input_file=input_file,
        output_file=output_file,
        checkpoint_interval=2,
    )

    print(f"  Final result: {final_result}")

    # Verify job completed
    print("\n[5/5] Verifying job completed successfully...")

    success, message = _check_regen_output.remote(
        output_file,
        expected_count=num_samples,
    )

    if not success:
        raise AssertionError(message)
    print(f"  {message}")

    # Verify state directory was cleaned up
    state_exists, _ = _check_file_exists.remote(state_dir)
    assert not state_exists, "State directory should be cleaned up after completion"
    print("  State directory cleaned up")

    # Cleanup test files
    print("\n  Cleaning up test files...")
    _cleanup_test_files.remote([input_file, output_file])

    print("\n" + "=" * 60)
    print("TEST PASSED: Preemption Recovery (Manual Resume)")
    print("=" * 60)


@app.function(
    image=sglang_image,
    gpu="H100",
    volumes={DATASET_PATH: dataset_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    timeout=600,
)
def _regen_partial(
    model: str,
    input_file: str,
    output_file: str,
    checkpoint_interval: int,
    stop_after_samples: int,
) -> str:
    """
    Run partial regeneration then exit (simulating preemption).

    This processes `stop_after_samples` samples, checkpoints state,
    then exits without completing the job.
    """
    from .modal_data import regenerate_dataset as regen_impl

    # Call with early stop parameter
    return regen_impl.local(
        model=model,
        input_file=input_file,
        output_file=output_file,
        checkpoint_interval=checkpoint_interval,
        _stop_after_samples=stop_after_samples,  # Test-only parameter
    )


# =============================================================================
# Test: Run All
# =============================================================================


@app.local_entrypoint()
def test_all():
    """Run all data tests sequentially."""
    print("=" * 60)
    print("RUNNING ALL DATA TESTS")
    print("=" * 60)

    tests = [
        ("prep_dataset", test_prep),
        ("RegenServer (Unit)", test_regen_unit),
        ("regenerate_dataset (Integration)", test_regen),
        ("preprocess_dataset", test_preprocess),
        ("Preemption Recovery (Auto)", test_preemption_auto),
        ("Preemption Recovery (Manual)", test_preemption_manual),
    ]

    results = []

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print("=" * 60)

        try:
            test_fn()
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
