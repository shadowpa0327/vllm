#!/usr/bin/env python3
"""
Dump prompts from math_eval.py preprocessing for vLLM bench serve.

This script extracts the exact prompts that would be sent to the LLM
at line 830 of math_eval.py (the prompts list), and outputs them in a format
compatible with vLLM's CustomDataset for bench serve.

Output format: JSONL with {"prompt": "..."}
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

# Import functions from math_eval.py directory
math_eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'math_benchmarks_backup1022')
sys.path.insert(0, math_eval_dir)

from data_loader import load_data
from parser import parse_question, parse_ground_truth
from utils import construct_prompt
from transformers import AutoTokenizer


def parse_args():
    """Parse arguments - reuse same args as math_eval.py for consistency"""
    parser = argparse.ArgumentParser(description="Dump math_eval prompts for vLLM bench serve")

    # Data arguments (same as math_eval.py)
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str,
                       help="Output JSONL file for vLLM bench serve")
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--n_sampling", default=1, type=int,
                       help="Number of times to repeat each prompt (for sampling)")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--repeat_dataset", type=int, default=1)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--adapt_few_shot", action="store_true",
                       help="Few shot for multiple-choice questions, zero shot for others")

    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    """
    Prepare data - exact same logic as math_eval.py prepare_data function.
    Returns examples list.
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

    if args.repeat_dataset < 1:
        raise ValueError("repeat_dataset must be >= 1")

    if args.repeat_dataset > 1:
        processed_pairs = set()  # Empty for fresh dump
        base_examples = list(examples)
        if not base_examples:
            examples = []
        else:
            base_stride = max(example["idx"] for example in base_examples) + 1
            repeated_examples = []
            for repeat_idx in range(args.repeat_dataset):
                for example in base_examples:
                    source_idx = example["idx"]
                    # No filtering against processed_pairs for fresh dump
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

    # Load tokenizer if needed
    tokenizer = None
    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )

    # Process all datasets
    data_list = args.data_names.split(",")
    all_samples = []

    for data_name in data_list:
        print(f"Processing dataset: {data_name}")
        examples = prepare_data(data_name, args)
        print(f"  Loaded {len(examples)} examples")

        # Build samples list (same as math_eval.py lines 722-767)
        samples = []
        for example in examples:
            idx = example["idx"]

            # Parse question and answer
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue

            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans

            # Construct prompt (this is what goes into sample["prompt"])
            full_prompt = construct_prompt(example, data_name, args)

            sample = {
                "idx": idx,
                "question": example["question"],
                "gt_cot": gt_cot,
                "gt": gt_ans,
                "prompt": full_prompt,
            }

            # Add metadata fields
            for key in ["level", "type", "unit", "solution_type", "choices",
                       "solution", "ques_type", "ans_type", "answer_type",
                       "dataset", "subfield", "filed", "theorem", "answer",
                       "repeat_id", "source_idx"]:
                if key in example:
                    sample[key] = example[key]

            samples.append(sample)

        all_samples.extend(samples)

    print(f"Total samples before n_sampling: {len(all_samples)}")

    # Now create input_prompts (lines 770-782 of math_eval.py)
    # This is the list that becomes prompts at line 830
    input_prompts = [
        sample["prompt"] for sample in all_samples for _ in range(args.n_sampling)
    ]

    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for prompt in input_prompts
        ]

    print(f"Total prompts after n_sampling={args.n_sampling}: {len(input_prompts)}")

    # Write to JSONL in vLLM CustomDataset format
    print(f"\nWriting prompts to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(input_prompts):
            # CustomDataset expects {"prompt": "..."}
            # We can add metadata but only "prompt" is required
            entry = {
                "prompt": prompt,
            }
            # Add index for tracking
            entry["request_id"] = i
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Done! Wrote {len(input_prompts)} prompts.")
    print(f"\nUsage with vLLM bench serve:")
    print(f"  # First, start vLLM server:")
    print(f"  vllm serve {args.model_name_or_path} \\")
    print(f"    --trust-remote-code \\")
    print(f"    [your vLLM engine arguments...]")
    print(f"")
    print(f"  # Then, run benchmark client:")
    print(f"  vllm bench serve \\")
    print(f"    --backend openai \\")
    print(f"    --model {args.model_name_or_path} \\")
    print(f"    --dataset-name custom \\")
    print(f"    --dataset-path {args.output_file} \\")
    print(f"    --num-prompts {len(input_prompts)} \\")
    print(f"    --request-rate <rate_in_qps>")


if __name__ == "__main__":
    main()
