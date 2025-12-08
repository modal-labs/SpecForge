"""
Base class for benchmark implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.case import skip

import numpy as np
import sglang as sgl
from sglang import set_default_backend
from sglang.test.test_utils import select_sglang_backend

from benchmarks.utils import BenchmarkMetrics, compute_metrics, print_results


class BaseBenchmark(ABC):
    """
    Base class for benchmark implementations.

    Subclasses should implement:
    - load_data(): Load and preprocess dataset
    - create_sgl_function(): Create the SGL function for inference

    Optional overrides:
    - extract_answer(): Extract answer from model output (if needed)
    - compute_accuracy(): Compute accuracy metric (if applicable)
    - get_answer_keys(): Get list of answer keys for multi-turn conversations
    """

    def __init__(self, args):
        """
        Initialize the benchmark.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.sgl_function = None
        self.questions = []
        self.labels = []

    @abstractmethod
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Load and preprocess the dataset.

        Returns:
            Tuple of (questions, labels) where:
            - questions: List of question dicts for SGL function
            - labels: List of ground truth labels (can be None if not applicable)
        """
        raise NotImplementedError

    @abstractmethod
    def create_sgl_function(self) -> Callable:
        """
        Create the SGL function for inference.

        Returns:
            SGL function decorated with @sgl.function
        """
        raise NotImplementedError

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[Any]:
        """
        Extract answer from model output.

        Args:
            output: Raw model output string
            label: Optional ground truth label for reference

        Returns:
            Extracted answer, or None if extraction fails
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Compute accuracy metric.

        Args:
            predictions: List of predicted answers
            labels: List of ground truth labels

        Returns:
            Accuracy score (0-1), or None if not applicable
        """
        return None

    def get_answer_keys(self) -> Optional[List[str]]:
        """
        Get list of answer keys for multi-turn conversations.

        Returns:
            List of answer keys (e.g., ["answer_1", "answer_2"]), or None for single-turn
        """
        return None

    def get_max_new_tokens(self) -> int:
        """
        Get maximum number of new tokens to generate.

        Returns:
            Maximum tokens (default: 2048)
        """
        return getattr(self.args, "max_new_tokens", 2048)

    def run(self, skip_labels: bool = False):
        """
        Run the benchmark evaluation.

        This method handles the common workflow:
        1. Initialize backend
        2. Load data
        3. Create SGL function
        4. Run inference loops
        5. Compute metrics
        6. Print results
        """
        # Initialize backend
        set_default_backend(select_sglang_backend(self.args))

        # Load data
        self.questions, self.labels = self.load_data()
        if len(self.questions) == 0:
            print("No valid questions found. Please check the dataset format.")
            return

        print(f"Loaded {len(self.questions)} questions.")

        # Create SGL function
        self.sgl_function = self.create_sgl_function()

        # Run evaluation loops
        metrics_list = []
        answer_keys = self.get_answer_keys()

        for run_idx in range(self.args.num_runs):
            tic = time.perf_counter()
            states = self.sgl_function.run_batch(
                self.questions,
                temperature=0,
                max_new_tokens=self.get_max_new_tokens(),
                num_threads=self.args.parallel,
                progress_bar=True,
            )
            latency = time.perf_counter() - tic

            # Extract predictions
            predictions = []
            primary_answer_key = answer_keys[0] if answer_keys else "answer"
            for i in range(len(states)):
                # Access answer from state object (states[i] supports dict-like access)
                output = states[i][primary_answer_key]
                if isinstance(output, str):
                    extracted = self.extract_answer(
                        output,
                        (
                            self.labels[i]
                            if self.labels and i < len(self.labels)
                            else None
                        ),
                    )
                else:
                    extracted = output
                predictions.append(extracted)

            # Compute accuracy if applicable
            accuracy = None
            # Check if we have a labels list (even if all labels are None)
            has_labels_list = self.labels and len(self.labels) > 0

            if has_labels_list and not skip_labels:
                # Always call compute_accuracy if we have a labels list
                # This allows it to return None, which will be displayed in print_results
                accuracy = self.compute_accuracy(predictions, self.labels)
                if accuracy is not None:
                    valid_count = sum(1 for p in predictions if p is not None)
                    if valid_count < len(predictions):
                        print(
                            f"Warning: {len(predictions) - valid_count} predictions could not be extracted."
                        )

            # Compute performance metrics
            metrics = compute_metrics(
                states,
                latency,
                answer_key=primary_answer_key,
                additional_answer_keys=(
                    answer_keys[1:] if answer_keys and len(answer_keys) > 1 else None
                ),
            )
            # Always set accuracy if we have a labels list (even if compute_accuracy returns None)
            # This allows print_results to show None when compute_accuracy returns None
            if has_labels_list and not skip_labels:
                metrics.accuracy = (
                    accuracy  # Can be None if compute_accuracy returns None
                )
                if accuracy is not None:
                    metrics.num_valid_predictions = sum(
                        1 for p in predictions if p is not None
                    )

            metrics_list.append(metrics)

        # Print results
        benchmark_name = (
            self.__class__.__name__.replace("Benchmark", "").replace("Run", "").upper()
        )
        # Show accuracy if we have a labels list (even if compute_accuracy returns None)
        # Check if we have a labels list by checking if accuracy was set
        # (accuracy is set when has_labels_list is True, even if it's None)
        has_labels_list = self.labels and len(self.labels) > 0
        show_accuracy = has_labels_list
        print_results(metrics_list, benchmark_name, show_accuracy=show_accuracy)
