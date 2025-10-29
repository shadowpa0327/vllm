#!/usr/bin/env python3
"""
Convert math_eval.py preprocessing format to vLLM bench serve format.

This script takes the same preprocessing logic from math_eval.py and
outputs a JSONL file with format compatible with vLLM's CustomDataset:
    {"prompt": "...", "expected_output_len": N}

Usage:
    python convert_math_eval_to_vllm_bench.py \\
        --data_names gsm8k,math \\
        --data_dir ./data \\
        --model_name_or_path <model> \\
        --output_file bench_prompts.jsonl \\
        [--apply_chat_template] \\
        [--enable_thinking] \\
        [other math_eval.py arguments...]
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

# Import functions from math_eval.py directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/math_benchmarks_backup1022')

from data_loader import load_data
from parser import parse_question, parse_ground_truth
from utils import construct_prompt
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Convert math_eval data to vLLM bench serve format")

    # Data arguments (same as math_eval.py)
    parser.add_argument("--data_names", default="gsm8k,math", type=str,
                      help="Comma-separated list of dataset names")
    parser.add_argument("--data_dir", default="./data", type=str,
                      help="Directory containing the datasets")
    parser.add_argument("--model_name_or_path", required=True, type=str,
                      help="Model name or path (for tokenizer)")
    parser.add_argument("--output_file", required=True, type=str,
                      help="Output JSONL file path")

    # Prompt configuration
    parser.add_argument("--prompt_type", default="tool-integrated", type=str,
                      help="Prompt type for math problems")
    parser.add_argument("--split", default="test", type=str,
                      help="Dataset split to use")
    parser.add_argument("--num_test_sample", default=-1, type=int,
                      help="Number of samples (-1 for all)")
    parser.add_argument("--seed", default=0, type=int,
                      help="Random seed")
    parser.add_argument("--start", default=0, type=int,
                      help="Start index")
    parser.add_argument("--end", default=-1, type=int,
                      help="End index")
    parser.add_argument("--shuffle", action="store_true",
                      help="Shuffle the dataset")
    parser.add_argument("--num_shots", type=int, default=0,
                      help="Number of few-shot examples")
    parser.add_argument("--repeat_dataset", type=int, default=1,
                      help="Repeat the dataset this many times")

    # Chat template arguments
    parser.add_argument("--apply_chat_template", action="store_true",
                      help="Apply chat template to prompts")
    parser.add_argument("--enable_thinking", action="store_true",
                      help="Enable Qwen3 thinking mode when applying chat template")

    # Expected output length
    parser.add_argument("--expected_output_len", type=int, default=None,
                      help="Expected output length for each prompt (for vLLM bench)")

    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    """
    Prepare data from a dataset, similar to math_eval.py prepare_data function.
    Returns list of examples for the specified dataset.
    """
    examples = load_data(data_name, args.split, args.data_dir)

    # Sample num_test_sample from dataset
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]

    # Shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # Select start and end
    examples = examples[args.start: len(examples) if args.end == -1 else args.end]

    # Handle repeat_dataset logic (from math_eval.py)
    if args.repeat_dataset < 1:
        raise ValueError("repeat_dataset must be >= 1")

    if args.repeat_dataset > 1:
        base_examples = list(examples)
        if not base_examples:
            examples = []
        else:
            base_stride = max(example["idx"] for example in base_examples) + 1
            repeated_examples = []
            for repeat_idx in range(args.repeat_dataset):
                for example in base_examples:
                    source_idx = example["idx"]
                    duplicated = copy.deepcopy(example)
                    duplicated["repeat_id"] = repeat_idx
                    duplicated["source_idx"] = source_idx
                    if repeat_idx > 0:
                        duplicated["idx"] = source_idx + repeat_idx * base_stride
                    repeated_examples.append(duplicated)
            examples = repeated_examples

    return examples


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load tokenizer if applying chat template
    tokenizer = None
    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )

    # Process all datasets
    data_list = args.data_names.split(",")
    all_prompts = []

    for data_name in data_list:
        print(f"Processing dataset: {data_name}")
        examples = prepare_data(data_name, args)

        print(f"  Total examples: {len(examples)}")

        for example in examples:
            idx = example["idx"]

            # Parse question and answer (same logic as math_eval.py)
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue

            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans

            # Construct prompt
            full_prompt = construct_prompt(example, data_name, args)

            # Apply chat template if requested
            if args.apply_chat_template:
                full_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": full_prompt.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )

            # Create entry in vLLM bench serve format
            entry = {
                "prompt": full_prompt,
            }

            # Add expected_output_len if specified
            if args.expected_output_len is not None:
                entry["expected_output_len"] = args.expected_output_len

            # Optionally, add metadata (will be ignored by CustomDataset but useful for analysis)
            entry["metadata"] = {
                "idx": idx,
                "dataset": data_name,
                "question": example["question"],
                "gt_ans": gt_ans,
            }

            if args.repeat_dataset > 1:
                entry["metadata"]["repeat_id"] = example.get("repeat_id", 0)
                entry["metadata"]["source_idx"] = example.get("source_idx", idx)

            all_prompts.append(entry)

    # Write to output file
    print(f"\\nWriting {len(all_prompts)} prompts to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)

    with open(args.output_file, 'w') as f:
        for entry in all_prompts:
            f.write(json.dumps(entry) + '\\n')

    print(f"Done! Output saved to: {args.output_file}")
    print(f"\\nYou can now use this with vLLM bench serve:")
    print(f"  vllm bench serve \\\\")
    print(f"    --backend openai \\\\")
    print(f"    --model {args.model_name_or_path} \\\\")
    print(f"    --dataset-name custom \\\\")
    print(f"    --dataset-path {args.output_file} \\\\")
    print(f"    --num-prompts {len(all_prompts)} \\\\")
    print(f"    --request-rate <rate>")


if __name__ == "__main__":
    main()
