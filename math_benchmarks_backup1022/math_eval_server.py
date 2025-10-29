#!/usr/bin/env python3
"""
Math benchmark evaluation script that submits requests to vLLM server.

This script extracts the data preprocessing logic from math_eval.py
and submits requests to a vLLM server using the async request pattern
similar to vllm/benchmarks/serve.py.

Usage:
    python math_eval_server.py \
        --data_names gsm8k,math \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --host 127.0.0.1 \
        --port 8000 \
        --num_test_sample 100 \
        --max_tokens_per_call 2048 \
        --request_rate 10
"""

import argparse
import asyncio
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

# Import from math_eval.py directory
from data_loader import load_data
from evaluate import evaluate
from parser import parse_question, parse_ground_truth, choice_answer_clean
from python_executor import PythonExecutor
from trajectory import run_execute, extract_program
from utils import construct_prompt, load_jsonl, save_jsonl, set_seed

# Import vLLM benchmark utilities
import sys
sys.path.insert(0, '/home/ubuntu/vllm')
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput
)


@dataclass
class MathSample:
    """Data structure for a math problem sample."""
    idx: int
    question: str
    gt: str
    gt_cot: str
    prompt: str
    # Optional fields
    level: Optional[str] = None
    type: Optional[str] = None
    solution: Optional[str] = None
    repeat_id: Optional[int] = None
    source_idx: Optional[int] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Math benchmark with server-based inference")

    # Data arguments
    parser.add_argument("--data_names", default="gsm8k,math", type=str,
                       help="Comma-separated list of dataset names")
    parser.add_argument("--data_dir", default="./data", type=str,
                       help="Directory containing benchmark data")
    parser.add_argument("--split", default="test", type=str,
                       help="Dataset split to use")
    parser.add_argument("--num_test_sample", default=-1, type=int,
                       help="Number of samples to test (-1 for all)")
    parser.add_argument("--start", default=0, type=int,
                       help="Start index for samples")
    parser.add_argument("--end", default=-1, type=int,
                       help="End index for samples")
    parser.add_argument("--shuffle", action="store_true",
                       help="Shuffle the dataset")

    # Model and server arguments
    parser.add_argument("--model_name_or_path", required=True, type=str,
                       help="Model name or path")
    parser.add_argument("--host", default="127.0.0.1", type=str,
                       help="Server host")
    parser.add_argument("--port", default=8000, type=int,
                       help="Server port")
    parser.add_argument("--endpoint", default="/v1/completions", type=str,
                       help="API endpoint")
    parser.add_argument("--backend", default="openai", type=str,
                       help="Backend type (openai, vllm)")

    # Generation arguments
    parser.add_argument("--max_tokens_per_call", default=2048, type=int,
                       help="Maximum tokens per generation")
    parser.add_argument("--temperature", default=0.0, type=float,
                       help="Sampling temperature")
    parser.add_argument("--top_p", default=1.0, type=float,
                       help="Top-p sampling")
    parser.add_argument("--n_sampling", default=1, type=int,
                       help="Number of times to sample each prompt")

    # Prompt arguments
    parser.add_argument("--prompt_type", default="tool-integrated", type=str,
                       help="Prompt type for math problems")
    parser.add_argument("--num_shots", type=int, default=0,
                       help="Number of few-shot examples")
    parser.add_argument("--apply_chat_template", action="store_true",
                       help="Apply chat template to prompts")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable Qwen3 thinking mode")

    # Request control arguments
    parser.add_argument("--request_rate", default=float("inf"), type=float,
                       help="Request rate (requests per second)")
    parser.add_argument("--max_concurrency", default=None, type=int,
                       help="Maximum concurrent requests")
    parser.add_argument("--burstiness", default=1.0, type=float,
                       help="Burstiness factor (1.0 = Poisson process)")

    # Output arguments
    parser.add_argument("--output_dir", default="./output", type=str,
                       help="Output directory for results")
    parser.add_argument("--save_outputs", action="store_true",
                       help="Save detailed outputs")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing results")

    # Misc arguments
    parser.add_argument("--seed", default=0, type=int,
                       help="Random seed")
    parser.add_argument("--repeat_dataset", type=int, default=1,
                       help="Repeat dataset N times")
    parser.add_argument("--disable_tqdm", action="store_true",
                       help="Disable progress bar")

    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    """
    Prepare dataset for evaluation.
    Extracted from math_eval.py prepare_data function.
    """
    examples = load_data(data_name, args.split, args.data_dir)

    # Sample num_test_sample from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # Shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # Select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Get output file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    repeat_suffix = "" if args.repeat_dataset == 1 else f"_repeat{args.repeat_dataset}"
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}{repeat_suffix}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # Load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    processed_samples = {sample.get("idx"): sample for sample in processed_samples}
    processed_samples = [sample for sample in processed_samples.values() if sample is not None]

    # Handle dataset repetition
    if args.repeat_dataset < 1:
        raise ValueError("repeat_dataset must be >= 1")

    if args.repeat_dataset > 1:
        processed_pairs = set()
        for sample in processed_samples:
            repeat_id = sample.get("repeat_id", 0)
            source_idx = sample.get("source_idx", sample.get("idx"))
            processed_pairs.add((repeat_id, source_idx))
        base_examples = list(examples)
        if not base_examples:
            examples = []
        else:
            base_stride = max(example["idx"] for example in base_examples) + 1
            repeated_examples = []
            for repeat_idx in range(args.repeat_dataset):
                for example in base_examples:
                    source_idx = example["idx"]
                    if (repeat_idx, source_idx) in processed_pairs:
                        continue
                    duplicated = copy.deepcopy(example)
                    duplicated["repeat_id"] = repeat_idx
                    duplicated["source_idx"] = source_idx
                    if repeat_idx > 0:
                        duplicated["idx"] = source_idx + repeat_idx * base_stride
                    repeated_examples.append(duplicated)
            examples = repeated_examples
    else:
        processed_idxs = {sample["idx"] for sample in processed_samples}
        examples = [example for example in examples if example["idx"] not in processed_idxs]

    return examples, processed_samples, out_file


def prepare_samples(examples, data_name, args, tokenizer=None):
    """
    Process raw examples into samples with prompts.
    Extracted from math_eval.py main function.
    """
    samples = []
    for example in examples:
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # Add remaining fields
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer", "repeat_id", "source_idx",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # Repeat n times for n_sampling
    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]

    # Apply chat template if needed
    if args.apply_chat_template:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for prompt in input_prompts
        ]

    return samples, input_prompts, tokenizer


async def get_request(
    input_prompts: list[str],
    request_rate: float,
    burstiness: float = 1.0,
):
    """
    Generate requests at specified rate with optional burstiness.
    Simplified from serve.py get_request function.
    """
    total_requests = len(input_prompts)
    assert total_requests > 0, "No requests provided."

    # Precompute delays
    delay_ts = []
    for _ in range(total_requests):
        if request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (request_rate * burstiness)
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    # Calculate cumulative delay
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]

    # Normalize to target rate (if not inf)
    if delay_ts[-1] != 0 and request_rate != float("inf"):
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for i, prompt in enumerate(input_prompts):
        if delay_ts[i] > 0:
            current_ts = time.time()
            sleep_interval_s = start_ts + delay_ts[i] - current_ts
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield i, prompt


async def send_request(
    request_func,
    session: aiohttp.ClientSession,
    api_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_words: list[str],
    request_id: int,
    pbar=None,
) -> RequestFuncOutput:
    """Send a single request to the server."""
    request_func_input = RequestFuncInput(
        model=model_id,
        model_name=model_id,
        prompt=prompt,
        api_url=api_url,
        prompt_len=len(prompt.split()),  # Approximate
        output_len=max_tokens,
        logprobs=None,
        multi_modal_content=None,
        ignore_eos=False,
        extra_headers=None,
        extra_body={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop_words,
        },
        request_id=str(request_id),
    )

    output = await request_func(
        request_func_input=request_func_input,
        session=session,
        pbar=pbar,
    )
    return output


async def benchmark_math(
    args,
    samples: list[dict],
    input_prompts: list[str],
    stop_words: list[str],
):
    """
    Run benchmark by sending requests to server.
    Adapted from serve.py benchmark function.
    """
    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    model_id = args.model_name_or_path

    # Get request function
    if args.backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[args.backend]
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Setup session
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency or 0,
        limit_per_host=args.max_concurrency or 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print(f"Sending {len(input_prompts)} requests to {api_url}")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Max concurrency: {args.max_concurrency}")

    # Setup semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def limited_request_func(idx, prompt, pbar):
        if semaphore is None:
            return await send_request(
                request_func, session, api_url, model_id, prompt,
                args.max_tokens_per_call, args.temperature, args.top_p,
                stop_words, idx, pbar
            )
        async with semaphore:
            return await send_request(
                request_func, session, api_url, model_id, prompt,
                args.max_tokens_per_call, args.temperature, args.top_p,
                stop_words, idx, pbar
            )

    # Send requests
    pbar = None if args.disable_tqdm else tqdm(total=len(input_prompts))
    tasks = []

    start_time = time.time()
    async for idx, prompt in get_request(input_prompts, args.request_rate, args.burstiness):
        task = asyncio.create_task(limited_request_func(idx, prompt, pbar))
        tasks.append(task)

    outputs = await asyncio.gather(*tasks)
    duration = time.time() - start_time

    if pbar is not None:
        pbar.close()

    await session.close()

    # Extract generated texts
    generated_texts = []
    for output in outputs:
        if output.success:
            generated_texts.append(output.generated_text)
        else:
            print(f"Request failed: {output.error}")
            generated_texts.append("")

    print(f"\nCompleted {len(outputs)} requests in {duration:.2f}s")
    print(f"Throughput: {len(outputs) / duration:.2f} req/s")

    return generated_texts, duration


def is_multi_choice(answer):
    """Check if answer is multi-choice format."""
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


async def main_async(args):
    """Main evaluation function."""
    set_seed(args.seed)

    # Setup tokenizer if needed
    tokenizer = None
    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    # Process each dataset
    data_list = args.data_names.split(",")
    results = []

    for data_name in data_list:
        print("=" * 50)
        print(f"Processing dataset: {data_name}")

        # Prepare data
        examples, processed_samples, out_file = prepare_data(data_name, args)
        print(f"Remaining samples: {len(examples)}")

        if len(examples) == 0:
            print("No new samples to process")
            continue

        # Prepare samples and prompts
        samples, input_prompts, tokenizer = prepare_samples(
            examples, data_name, args, tokenizer
        )

        # Define stop words based on prompt type
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        if args.prompt_type in ["cot"]:
            stop_words.append("\n\nQuestion:")
        if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
            stop_words.extend(["\n\n---", "```output"])
        elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
            stop_words.extend(["Instruction", "Response"])
        elif "jiuzhang" in args.prompt_type:
            stop_words.append("\n\n## Question")
        elif "numina" in args.prompt_type:
            stop_words.append("\n### Problem")
        elif "pure" in args.prompt_type:
            stop_words.append("\n\n\n")

        # Run benchmark
        generated_texts, duration = await benchmark_math(
            args, samples, input_prompts, stop_words
        )

        # Process outputs
        # For tool-integrated prompts, we may need to execute code
        executor = PythonExecutor(get_answer_from_stdout=True)
        if "pal" in args.prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")

        # Split outputs back to samples
        codes = []
        for i, text in enumerate(generated_texts):
            code = text.strip()
            for stop_word in stop_words:
                if stop_word in code:
                    code = code.split(stop_word)[0].strip()
            codes.append(code)

        # Execute and extract predictions
        results_exec = [
            run_execute(executor, code, args.prompt_type, data_name)
            for code in codes
        ]

        # Put results back to samples
        all_samples = []
        for i, sample in enumerate(samples):
            code_batch = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            result_batch = results_exec[i * args.n_sampling : (i + 1) * args.n_sampling]
            preds = [item[0] for item in result_batch]
            reports = [item[1] for item in result_batch]

            # Clean up predictions for multiple choice
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                    preds[j] = choice_answer_clean(code_batch[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])

            sample.pop("prompt")
            sample.update({"code": code_batch, "pred": preds, "report": reports})
            all_samples.append(sample)

        # Add processed samples
        all_samples.extend(processed_samples)
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )

        # Save outputs
        if len(processed_samples) < len(all_samples) and args.save_outputs:
            save_jsonl(all_samples, out_file)

        # Add timing info
        result_json["duration_seconds"] = duration
        result_json["throughput_req_per_sec"] = len(generated_texts) / duration if duration > 0 else 0

        # Save metrics
        with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
            json.dump(result_json, f, indent=4)

        results.append(result_json)

    # Print final results
    if len(results) > 0:
        data_list.append("avg")
        results.append({
            "acc": sum([result["acc"] for result in results]) / len(results),
        })

        pad = max([len(data_name) for data_name in data_list])
        print("\n=== Final Evaluation Results ===")
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
