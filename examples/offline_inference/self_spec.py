# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import random
import numpy as np
import torch

# Set environment variables
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def load_prompts(dataset_path, num_prompts):
    """Load prompts from dataset file or use default prompts."""
    if os.path.exists(dataset_path):
        prompts = []
        try:
            with open(dataset_path) as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["turns"][0])
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return []
    else:
        # Default prompts if dataset file doesn't exist
        prompts = ["The future of AI is", "The future of technology is"]
    return prompts[:num_prompts]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Self-speculative decoding inference script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./examples/data/gsm8k.jsonl",
        help="Path to dataset file",
    )
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Maximum number of sequences")
    parser.add_argument("--num_prompts", type=int, default=80, help="Number of prompts to process")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--enforce_eager", action="store_true", help="Enforce eager execution")
    parser.add_argument("--enable_chunked_prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("--max_num_batched_tokens", type=int, default=2048, help="Maximum batched tokens")
    parser.add_argument("--temp", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--enable_speculative", action="store_true", help="Enable self-speculative decoding")
    parser.add_argument("--num_speculative_tokens", type=int, default=4, help="Number of speculative tokens for self-spec")
    return parser.parse_args()


def get_speculative_config(args):
    """Get self-speculative decoding configuration based on arguments."""
    if not args.enable_speculative:
        return None
    
    return {
        "method": "self_specs",
        "model": None,
        "num_speculative_tokens": args.num_speculative_tokens,
    }


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    args = parse_args()

    model_dir = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_model_len = 2048

    # Load tokenizer and prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = load_prompts(args.dataset, args.num_prompts)
    
    print(f"Loaded {len(prompts)} prompts")
    
    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        for prompt in prompts
    ]

    # Get speculative config
    speculative_config = get_speculative_config(args)
    
    # Initialize LLM
    llm_kwargs = {
        "model": model_dir,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "block_size": 1, # NOTE(brian1009): Set to 1 to disable prefix caching
    }
    
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
        print(f"Using self-speculative decoding with {speculative_config['num_speculative_tokens']} tokens")
    else:
        print("Self-speculative decoding disabled")

    llm = LLM(**llm_kwargs)

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=args.temp, max_tokens=256)

    # Generate outputs
    print("Starting generation...")
    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    # Print generated text
    for i, output in enumerate(outputs):
        print(f"\n{'='*60}")
        print(f"Output {i+1}/{len(outputs)}")
        print(f"{'='*60}")
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

    # Print metrics if available
    try:
        metrics = llm.get_metrics()
        print_metrics(metrics)
    except AssertionError:
        print("\nMetrics are not supported in the V0 engine.")


def print_metrics(metrics):
    """Print self-speculative decoding metrics."""
    num_drafts = num_accepted = 0
    acceptance_counts = [0] * 16  # Track acceptance at each position
    
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    if num_drafts > 0:
        print(f"\n{'='*60}")
        print("SELF-SPECULATIVE DECODING METRICS")
        print(f"{'='*60}")
        print(f"Mean acceptance length: {1 + (num_accepted / num_drafts):.2f}")
        print(f"Total drafts: {num_drafts}")
        print(f"Total accepted: {num_accepted}")
        
        print("\nAcceptance rate by token position:")
        for i in range(len(acceptance_counts)):
            if acceptance_counts[i] > 0:
                rate = acceptance_counts[i] / num_drafts
                print(f"  Position {i}: {rate:.3f} ({acceptance_counts[i]}/{num_drafts})")


if __name__ == "__main__":
    main()
