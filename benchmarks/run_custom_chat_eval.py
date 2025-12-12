"""
Benchmark for chat-style datasets that are already in OpenAI-style message format
and need to be rendered with a tokenizer chat template (including tool/observation turns).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from sglang.test.test_utils import add_common_sglang_args_and_parse
import sglang as sgl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_benchmark import BaseBenchmark


def render_example(
    tokenizer,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render messages with tokenizer chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


class CustomChatEvalBenchmark(BaseBenchmark):
    """Benchmark for chat-format datasets with tokenizer rendering."""

    def _build_gen_kwargs(self) -> Dict[str, Any]:
        gen_kwargs = {}
        if getattr(self.args, "temperature", None) is not None:
            gen_kwargs["temperature"] = self.args.temperature
        if getattr(self.args, "top_p", None) is not None:
            gen_kwargs["top_p"] = self.args.top_p
        if getattr(self.args, "max_tokens", None) is not None:
            gen_kwargs["max_tokens"] = self.args.max_tokens
        return gen_kwargs

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        dataset_path = Path(self.args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = (Path(__file__).parent.parent / dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_path} not found.")

        print(f"Loading data from {dataset_path}")

        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []

        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                conv = data.get("conversations") or data.get("messages")
                if not conv:
                    continue

                # label: last assistant content if present
                label = None
                if conv[-1].get("role") == "assistant":
                    label = conv[-1].get("content", "")
                    context = conv[:-1]
                else:
                    context = conv

                tools = data.get("tools")

                rendered = render_example(
                    tokenizer,
                    context,
                    add_generation_prompt=True,
                    tools=tools,
                )
                questions.append({"prompt": rendered})
                labels.append(label)

                if (
                    hasattr(self.args, "num_questions")
                    and self.args.num_questions is not None
                    and len(questions) >= self.args.num_questions
                ):
                    break

        return questions, labels

    def create_sgl_function(self) -> Callable:
        gen_kwargs = self._build_gen_kwargs()

        @sgl.function
        def chat_gen(s, prompt):
            # prompt already includes generation prompt (assistant header)
            s += prompt
            s += sgl.gen("answer", **gen_kwargs)

        return chat_gen


def main(args):
    benchmark = CustomChatEvalBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path/name for AutoTokenizer with chat template",
    )
    parser.add_argument(
        "--num-questions", type=int, default=None, help="Number of questions to run"
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature for generation (backend default if omitted).",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Optional top-p for nucleus sampling (backend default if omitted).",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Optional max tokens for generation (backend default if omitted).",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens passed to run_batch (defaults to 2048).",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)
