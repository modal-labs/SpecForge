"""
SpecForge Modal Training Package

A Modal app for training production-ready speculative decoding draft models.

Usage:
    # From repo root
    modal run -m examples.modal.modal_data::prep_dataset --dataset sharegpt
    modal run -m examples.modal.modal_train::train --target-model Qwen/Qwen3-8B
    modal run -m examples.modal.test_data::test_prep
"""
