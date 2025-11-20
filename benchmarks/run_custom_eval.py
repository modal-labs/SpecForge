"""
Benchmark for custom OpenAI format dataset (GLM Eval).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from sglang.test.test_utils import add_common_sglang_args_and_parse
import sglang as sgl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_benchmark import BaseBenchmark


class CustomEvalBenchmark(BaseBenchmark):
    """Benchmark for custom OpenAI format dataset."""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Load and preprocess the dataset.
        Expected format: JSONL with "conversations" list of messages.
        """
        dataset_path = self.args.dataset_path

        # Handle relative paths
        if not Path(dataset_path).is_absolute():
            # Try finding it in current dir or project root
            if Path(dataset_path).exists():
                dataset_path = Path(dataset_path).absolute()
            elif (Path(__file__).parent.parent / dataset_path).exists():
                dataset_path = (Path(__file__).parent.parent / dataset_path).absolute()
            else:
                raise FileNotFoundError(f"Dataset {dataset_path} not found.")

        print(f"Loading data from {dataset_path}")

        questions = []
        labels = []

        try:
            with open(dataset_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "conversations" in data:
                        # Take all messages except the last one (which is the assistant's response to generate)
                        full_conv = data["conversations"]
                        if not full_conv:
                            continue

                        # Check if last message is assistant
                        if full_conv[-1]["role"] == "assistant":
                            context_messages = full_conv[:-1]
                            label = full_conv[-1]["content"]
                        else:
                            # If last message is user, we just generate a response
                            context_messages = full_conv
                            label = None

                        questions.append({"messages": context_messages})
                        labels.append(label)
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []

        # Limit number of questions if specified
        if hasattr(self.args, "num_questions") and self.args.num_questions is not None:
            questions = questions[: self.args.num_questions]
            labels = labels[: self.args.num_questions]

        return questions, labels

    def create_sgl_function(self) -> Callable:
        """Create the SGL function for inference."""

        @sgl.function
        def chat_gen(s, messages):
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    s += sgl.system(content)
                elif role == "user":
                    s += sgl.user(content)
                elif role == "assistant":
                    s += sgl.assistant(content)

            # Generate the response
            s += sgl.assistant(
                sgl.gen(
                    "answer",
                    max_tokens=self.get_max_new_tokens(),
                    temperature=0.4,
                    top_p=1.0,
                )
            )

        return chat_gen


def main(args):
    benchmark = CustomEvalBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--num-questions", type=int, default=None, help="Number of questions to run"
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
