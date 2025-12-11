"""
Preformat a ShareGPT-style dataset into a single `text` field using the tokenizer's
chat template. This avoids pyarrow schema inference issues during later loading by
SpecForge when `--is-preformatted` is used.

Input JSONL rows (unbounded schema):
{
  "id": "xxxx",
  "conversations": [...],
  "tools": [...]  # optional
}

Output JSONL rows:
{
  "id": "xxxx",
  "text": "<rendered chat template string>"
}
"""

import argparse
import json
import multiprocessing as mp
from functools import partial
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL file.")
    ap.add_argument("--output", required=True, help="Path to output JSONL file.")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path with chat template support.",
    )
    ap.add_argument(
        "--chat-template",
        default="glm4_moe",
        help="Template name registered in specforge.data.template.TEMPLATE_REGISTRY.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of worker processes.",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="Chunksize for imap; tune for throughput/memory tradeoff.",
    )
    return ap.parse_args()


# Globals set in worker init
_tok = None
_template: ChatTemplate | None = None


def _init_worker(tokenizer_path: str, template_name: str):
    global _tok, _template
    _tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _template = TEMPLATE_REGISTRY.get(template_name)


def _format_line(line: str) -> Dict[str, Any]:
    obj = json.loads(line)
    conv: List[Dict[str, Any]] = obj.get("conversations") or []
    tools = obj.get("tools")

    messages: List[Dict[str, Any]] = []
    if conv and conv[0].get("role") != "system" and _template.system_prompt:
        messages.append({"role": "system", "content": _template.system_prompt})
    messages.extend(conv)

    rendered = _tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tools
    )
    return {"id": obj.get("id"), "text": rendered}


def main():
    args = parse_args()

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.tokenizer, args.chat_template),
        ) as pool:
            for rec in tqdm(
                pool.imap(_format_line, fin, chunksize=args.chunksize),
                desc="Preformatting",
            ):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote preformatted dataset to {args.output}")


if __name__ == "__main__":
    main()
