import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime

# HARDCODED: Set V1 environment variables BEFORE importing vllm
# Required for FlashInfer backend to work properly with your local vLLM changes
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from parser import *

import torch
from data_loader import load_data
from evaluate import evaluate
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from python_executor import PythonExecutor
from tqdm import tqdm
from trajectory import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams

try:
    import sglang

    SGLANG_ENGINE = sglang.Engine
    SGLANG_AVAILABLE = True
except ImportError as e:
    SGLANG_AVAILABLE = False
    SGLANG_ENGINE = None
    print(f"SGLang not available: {e}")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available (optional)")

# serve.* imports removed - not needed for basic vLLM self-spec testing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=None, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument(
        "--use_sglang", action="store_true", help="Use sglang inference"
    )
    parser.add_argument("--use_sspec", action="store_true", help="Use sspec inference")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--repeat_dataset",
        type=int,
        default=1,
        help="Repeat the dataset this many times to simulate multiple request rounds.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode when applying chat template.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )

    # sspec specific arguments
    parser.add_argument(
        "--sspec_max_batch_size",
        type=int,
        default=128,
        help="Max batch size for sspec inference",
    )
    parser.add_argument(
        "--sspec_tp_ranks", type=int, default=1, help="Number of tensor parallel ranks"
    )
    parser.add_argument(
        "--sspec_kv_cache",
        type=str,
        default="full",
        choices=["full", "streaming", "pillar"],
        help="KV cache type (full or streaming)",
    )
    parser.add_argument(
        "--sspec_admit_policy",
        type=str,
        default="naive",
        choices=["naive", "oracle", "offloading"],
        help="Admission policy (note: 'naive' not compatible with 'sspec' kv_cache)",
    )
    parser.add_argument(
        "--sspec_cuda_graph", action="store_true", help="Enable CUDA graph optimization"
    )
    parser.add_argument(
        "--sspec_debug", action="store_true", help="Enable debug mode for sspec"
    )
    parser.add_argument(
        "--sspec_async_cpu", action="store_true", help="Enable async CPU scheduling"
    )
    parser.add_argument(
        "--sspec_keep_input",
        action="store_true",
        help="Keep input text in output (for debugging)",
    )
    parser.add_argument(
        "--sspec_verbose",
        action="store_true",
        help="Print detailed debugging information",
    )
    # removed: sspec_clip_length (sspec uses max_tokens_per_call as clip_length; None -> auto from HF)
    parser.add_argument(
        "--sspec_vanilla_attn",
        action="store_true",
        help="Use vanilla attention backend",
    )
    parser.add_argument(
        "--sspec_mem_fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use per rank for sspec inference (e.g., 0.9)",
    )
    parser.add_argument(
        "--sspec_stride",
        type=int,
        default=16,
        help="Speculative stride for sspec inference",
    )

    # vLLM ngram speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_ngram",
        action="store_true",
        help="Enable ngram-based speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_num_speculative_tokens",
        type=int,
        default=5,
        help="Number of speculative tokens for vLLM ngram speculation",
    )
    parser.add_argument(
        "--vllm_ngram_prompt_lookup_min",
        type=int,
        default=1,
        help="Minimum ngram size for vLLM prompt lookup",
    )
    parser.add_argument(
        "--vllm_ngram_prompt_lookup_max",
        type=int,
        default=4,
        help="Maximum ngram size for vLLM prompt lookup",
    )

    # vLLM suffix decoding speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_suffix",
        action="store_true",
        help="Enable suffix decoding speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_suffix_num_speculative_tokens",
        type=int,
        default=5,
        help="Number of speculative tokens for suffix decoding",
    )
    parser.add_argument(
        "--vllm_suffix_max_tree_depth",
        type=int,
        default=24,
        help="Maximum depth for suffix tree pattern matching",
    )
    parser.add_argument(
        "--vllm_suffix_max_cached_requests",
        type=int,
        default=10000,
        help="Maximum number of cached request responses (0 = disable global cache)",
    )
    parser.add_argument(
        "--vllm_suffix_max_spec_factor",
        type=float,
        default=1.0,
        help="Speculation length factor: max_spec_tokens = factor × prefix_match_length",
    )
    parser.add_argument(
        "--vllm_suffix_min_token_prob",
        type=float,
        default=0.1,
        help="Minimum frequency threshold for speculating a token",
    )

    # vLLM EAGLE3 speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_eagle3",
        action="store_true",
        help="Enable EAGLE3 speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_draft_model_path",
        type=str,
        default=None,
        help="Path to the draft model for EAGLE3 speculative decoding in vLLM",
    )
    parser.add_argument(
        "--vllm_eagle3_num_speculative_tokens",
        type=int,
        default=None,
        help="Number of speculative tokens for EAGLE3 (default: None, uses vLLM's default)",
    )

    # vLLM self-speculative decoding arguments (aligned with self_spec.py)
    parser.add_argument(
        "--vllm_enable_sspec",
        action="store_true",
        help="Enable self-speculative decoding in vLLM (requires V1 engine)",
    )
    parser.add_argument(
        "--vllm_sspec_num_speculative_tokens",
        type=int,
        default=16,
        help="Number of speculative tokens for self-spec (default: 16)",
    )
    parser.add_argument(
        "--vllm_sspec_sink_size",
        type=int,
        default=8,
        help="Number of sink blocks for streaming cache (default: 8 blocks = 128 tokens with block_size=16)",
    )
    parser.add_argument(
        "--vllm_sspec_recent_ratio",
        type=float,
        default=0.10,
        help="Ratio of recent blocks for streaming cache (default: 0.10 = 10%% of computed tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_block_size",
        type=int,
        default=1,
        help="Block size for self-spec (default: 1, disables prefix caching)",
    )

    # vLLM self-spec with n-gram arguments
    parser.add_argument(
        "--vllm_enable_sspec_ngram",
        action="store_true",
        help="Enable self-spec with n-gram assistance in vLLM (requires V1 engine)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_num_speculative_tokens",
        type=int,
        default=16,
        help="Threshold for ACCUMULATING -> VERIFYING transition (default: 16)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_num_ngram_draft_tokens",
        type=int,
        default=3,
        help="Number of n-gram draft tokens per step during ACCUMULATING (default: 3)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_prompt_lookup_min",
        type=int,
        default=None,
        help="Minimum n-gram window size (default: same as num_ngram_draft_tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_prompt_lookup_max",
        type=int,
        default=None,
        help="Maximum n-gram window size (default: same as num_ngram_draft_tokens)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_sink_size",
        type=int,
        default=8,
        help="Number of sink blocks for streaming cache (default: 8 blocks)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_recent_ratio",
        type=float,
        default=0.10,
        help="Ratio of recent blocks for streaming cache (default: 0.10)",
    )
    parser.add_argument(
        "--vllm_sspec_ngram_block_size",
        type=int,
        default=1,
        help="Block size for self-spec ngram (default: 1, disables prefix caching)",
    )

    # vLLM self-spec suffix speculative decoding arguments
    parser.add_argument(
        "--vllm_enable_sspec_suffix",
        action="store_true",
        help="Enable self-spec with suffix decoding assistance in vLLM (requires V1 engine)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_num_speculative_tokens",
        type=int,
        default=6,
        help="Threshold for ACCUMULATING -> VERIFYING transition (default: 6)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_num_suffix_draft_tokens",
        type=int,
        default=3,
        help="Number of suffix draft tokens per step during ACCUMULATING (default: 3)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_max_tree_depth",
        type=int,
        default=24,
        help="Maximum depth for suffix tree pattern matching (default: 24)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_max_cached_requests",
        type=int,
        default=10000,
        help="Maximum number of cached request responses (default: 10000, 0 = disable global cache)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_max_spec_factor",
        type=float,
        default=1.0,
        help="Speculation length factor (default: 1.0)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_min_token_prob",
        type=float,
        default=0.1,
        help="Minimum frequency threshold for speculating a token (default: 0.1)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_sink_size",
        type=int,
        default=8,
        help="Number of sink blocks for streaming cache (default: 8 blocks)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_recent_ratio",
        type=float,
        default=0.10,
        help="Ratio of recent blocks for streaming cache (default: 0.10)",
    )
    parser.add_argument(
        "--vllm_sspec_suffix_block_size",
        type=int,
        default=1,
        help="Block size for self-spec suffix (default: 1, disables prefix caching)",
    )

    # SGLang specific arguments
    parser.add_argument(
        "--sglang_mem_fraction_static",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for SGLang static allocation",
    )
    parser.add_argument(
        "--sglang_enable_torch_compile",
        action="store_true",
        help="Enable torch.compile optimization in SGLang",
    )
    parser.add_argument(
        "--sglang_disable_radix_cache",
        action="store_true",
        help="Disable radix cache (prefix sharing) in SGLang",
    )

    # SGLang EAGLE3 speculative decoding arguments
    parser.add_argument(
        "--sglang_enable_eagle3",
        action="store_true",
        help="Enable EAGLE3 speculative decoding in SGLang",
    )
    parser.add_argument(
        "--sglang_draft_model_path",
        type=str,
        default=None,
        help="Path to the draft model for EAGLE3 speculative decoding",
    )
    parser.add_argument(
        "--sglang_speculative_num_steps",
        type=int,
        default=5,
        help="Number of steps to run the draft model before verification (EAGLE3)",
    )
    parser.add_argument(
        "--sglang_speculative_eagle_topk",
        type=int,
        default=8,
        help="Number of top-k candidates to keep per step for verification (EAGLE3)",
    )
    parser.add_argument(
        "--sglang_speculative_num_draft_tokens",
        type=int,
        default=32,
        help="Number of tokens to propose in the draft (EAGLE3)",
    )

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    repeat_suffix = "" if args.repeat_dataset == 1 else f"_repeat{args.repeat_dataset}"
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}{repeat_suffix}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = (
        f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    )
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    processed_samples = {sample.get("idx"): sample for sample in processed_samples}
    processed_samples = [
        sample for sample in processed_samples.values() if sample is not None
    ]

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
        examples = [
            example for example in examples if example["idx"] not in processed_idxs
        ]

    return examples, processed_samples, out_file


def setup(args):
    # load model
    if args.use_sspec:
        # For sspec, we don't need to load model here as run_inference handles it
        llm = None
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    elif args.use_sglang:
        # Check if SGLang is available
        if not SGLANG_AVAILABLE:
            raise ImportError(
                "SGLang is not installed. Please install it with: pip install 'sglang[all]'"
            )

        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        tp_size = len(available_gpus) // args.pipeline_parallel_size

        # Build SGLang Engine kwargs
        sglang_kwargs = {
            "model_path": args.model_name_or_path,
            "tp_size": tp_size,
            "trust_remote_code": True,
            "mem_fraction_static": args.sglang_mem_fraction_static,
        }

        if args.sglang_enable_torch_compile:
            sglang_kwargs["enable_torch_compile"] = True
            print("Enabled SGLang torch.compile optimization")

        if args.sglang_disable_radix_cache:
            sglang_kwargs["disable_radix_cache"] = True
            print("Disabled SGLang radix cache (prefix sharing)")

        # Enable EAGLE3 speculative decoding if requested
        if args.sglang_enable_eagle3:
            if args.sglang_draft_model_path is None:
                raise ValueError(
                    "sglang_draft_model_path must be specified when sglang_enable_eagle3 is enabled"
                )
            sglang_kwargs["speculative_algorithm"] = "EAGLE3"
            sglang_kwargs["speculative_draft_model_path"] = args.sglang_draft_model_path
            sglang_kwargs["speculative_num_steps"] = args.sglang_speculative_num_steps
            sglang_kwargs["speculative_eagle_topk"] = args.sglang_speculative_eagle_topk
            sglang_kwargs["speculative_num_draft_tokens"] = (
                args.sglang_speculative_num_draft_tokens
            )
            print(
                f"Enabled SGLang EAGLE3 speculative decoding with draft model: {args.sglang_draft_model_path}"
            )
            print(f"  - Speculative num steps: {args.sglang_speculative_num_steps}")
            print(f"  - EAGLE topk: {args.sglang_speculative_eagle_topk}")
            print(f"  - Num draft tokens: {args.sglang_speculative_num_draft_tokens}")

        # Initialize SGLang Engine
        llm = SGLANG_ENGINE(**sglang_kwargs)
        # Always load tokenizer for SGLang to properly count tokens for TPS statistics
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        print(f"Initialized SGLang Engine with TP={tp_size}")
    else:
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if args.use_vllm:
            # Check for mutually exclusive vLLM speculative decoding options
            spec_options = sum([args.vllm_enable_ngram, args.vllm_enable_eagle3, args.vllm_enable_sspec, args.vllm_enable_sspec_ngram, args.vllm_enable_sspec_suffix, args.vllm_enable_suffix])
            if spec_options > 1:
                raise ValueError(
                    "Cannot enable multiple speculative decoding methods at the same time. "
                    "Please choose only one: vllm_enable_ngram, vllm_enable_eagle3, vllm_enable_sspec, vllm_enable_sspec_ngram, vllm_enable_sspec_suffix, or vllm_enable_suffix."
                )

            # Set environment variables for self-spec methods (must be set before vLLM import)
            if args.vllm_enable_sspec or args.vllm_enable_sspec_ngram or args.vllm_enable_sspec_suffix:
                os.environ["VLLM_USE_V1"] = "1"
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
                if args.vllm_enable_sspec_ngram:
                    method_name = "self_spec_ngram"
                elif args.vllm_enable_sspec_suffix:
                    method_name = "self_spec_suffix"
                else:
                    method_name = "self_specs"
                print(f"Enabled V1 engine for {method_name}")
                print(f"  VLLM_USE_V1=1")
                print(f"  VLLM_ENABLE_V1_MULTIPROCESSING=1")
                print(f"  VLLM_ATTENTION_BACKEND=FLASHINFER")

            # Build vLLM kwargs
            vllm_kwargs = {
                "model": args.model_name_or_path,
                "tensor_parallel_size": len(available_gpus)
                // args.pipeline_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "trust_remote_code": True,
            }

            # Add ngram speculative decoding config if enabled
            if args.vllm_enable_ngram:
                speculative_config = {
                    "method": "ngram",
                    "num_speculative_tokens": args.vllm_num_speculative_tokens,
                    "ngram_prompt_lookup_min": args.vllm_ngram_prompt_lookup_min,
                    "ngram_prompt_lookup_max": args.vllm_ngram_prompt_lookup_max,
                }
                vllm_kwargs["speculative_config"] = speculative_config
                # IMPORTANT: Enable stat logging to access spec decode metrics via get_metrics()
                vllm_kwargs["disable_log_stats"] = False
                print(
                    f"Enabled vLLM ngram speculative decoding with config: {speculative_config}"
                )
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add suffix decoding speculative decoding config if enabled
            if args.vllm_enable_suffix:
                speculative_config = {
                    "method": "suffix",
                    "num_speculative_tokens": args.vllm_suffix_num_speculative_tokens,
                    "suffix_decoding_max_tree_depth": args.vllm_suffix_max_tree_depth,
                    "suffix_decoding_max_cached_requests": args.vllm_suffix_max_cached_requests,
                    "suffix_decoding_max_spec_factor": args.vllm_suffix_max_spec_factor,
                    "suffix_decoding_min_token_prob": args.vllm_suffix_min_token_prob,
                }
                vllm_kwargs["speculative_config"] = speculative_config
                # IMPORTANT: Enable stat logging to access spec decode metrics via get_metrics()
                vllm_kwargs["disable_log_stats"] = False
                print(
                    f"Enabled vLLM suffix decoding with config: {speculative_config}"
                )
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add EAGLE3 speculative decoding config if enabled
            if args.vllm_enable_eagle3:
                if args.vllm_draft_model_path is None:
                    raise ValueError(
                        "vllm_draft_model_path must be specified when vllm_enable_eagle3 is enabled"
                    )
                # Configure EAGLE3 using speculative_config
                eagle3_config = {
                    "method": "eagle3",
                    "model": args.vllm_draft_model_path,
                    "draft_tensor_parallel_size": 1,  # EAGLE3 requires TP=1 for draft model
                }
                # Add num_speculative_tokens if specified
                if args.vllm_eagle3_num_speculative_tokens is not None:
                    eagle3_config["num_speculative_tokens"] = (
                        args.vllm_eagle3_num_speculative_tokens
                    )

                vllm_kwargs["speculative_config"] = eagle3_config
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                print(
                    f"Enabled vLLM EAGLE3 speculative decoding with draft model: {args.vllm_draft_model_path}"
                )
                print(f"  - Method: eagle3")
                print(f"  - Draft model tensor parallel size: 1")
                if args.vllm_eagle3_num_speculative_tokens is not None:
                    print(
                        f"  - Num speculative tokens: {args.vllm_eagle3_num_speculative_tokens}"
                    )
                else:
                    print(f"  - Num speculative tokens: (using vLLM default)")
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add self-speculative decoding config if enabled (aligned with self_spec.py)
            if args.vllm_enable_sspec:
                sspec_config = {
                    "method": "self_specs",
                    "model": None,
                    "num_speculative_tokens": args.vllm_sspec_num_speculative_tokens,
                }
                vllm_kwargs["speculative_config"] = sspec_config
                # Add streaming cache parameters
                vllm_kwargs["sink_size"] = args.vllm_sspec_sink_size
                vllm_kwargs["recent_ratio"] = args.vllm_sspec_recent_ratio
                vllm_kwargs["block_size"] = args.vllm_sspec_block_size
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                # Additional V1-specific parameters
                vllm_kwargs["enforce_eager"] = False
                vllm_kwargs["enable_chunked_prefill"] = True
                vllm_kwargs["enable_prefix_caching"] = False
                vllm_kwargs["gpu_memory_utilization"] = 0.94
                vllm_kwargs["max_num_batched_tokens"] = 2048
                vllm_kwargs["max_num_seqs"] = 256
                vllm_kwargs["cuda_graph_sizes"] = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 1536]

                print(f"Enabled vLLM self-speculative decoding with config:")
                print(f"  - Method: self_specs")
                print(f"  - Num speculative tokens: {args.vllm_sspec_num_speculative_tokens}")
                print(f"  - Sink size (streaming cache): {args.vllm_sspec_sink_size} blocks")
                print(f"  - Recent ratio (streaming cache): {args.vllm_sspec_recent_ratio} ({args.vllm_sspec_recent_ratio*100:.1f}%)")
                print(f"  - Block size: {args.vllm_sspec_block_size}")
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add self-spec with n-gram config if enabled
            if args.vllm_enable_sspec_ngram:
                sspec_ngram_config = {
                    "method": "self_spec_ngram",
                    "model": None,
                    "num_speculative_tokens": args.vllm_sspec_ngram_num_speculative_tokens,
                    "num_ngram_draft_tokens": args.vllm_sspec_ngram_num_ngram_draft_tokens,
                }
                # Set n-gram window sizes (default to num_ngram_draft_tokens if not specified)
                lookup_max = args.vllm_sspec_ngram_prompt_lookup_max if args.vllm_sspec_ngram_prompt_lookup_max is not None else args.vllm_sspec_ngram_num_ngram_draft_tokens
                lookup_min = args.vllm_sspec_ngram_prompt_lookup_min if args.vllm_sspec_ngram_prompt_lookup_min is not None else args.vllm_sspec_ngram_num_ngram_draft_tokens
                sspec_ngram_config["prompt_lookup_max"] = lookup_max
                sspec_ngram_config["prompt_lookup_min"] = lookup_min

                vllm_kwargs["speculative_config"] = sspec_ngram_config
                # Add streaming cache parameters
                vllm_kwargs["sink_size"] = args.vllm_sspec_ngram_sink_size
                vllm_kwargs["recent_ratio"] = args.vllm_sspec_ngram_recent_ratio
                vllm_kwargs["block_size"] = args.vllm_sspec_ngram_block_size
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                # Additional V1-specific parameters
                #vllm_kwargs["enforce_eager"] = False
                vllm_kwargs["enable_chunked_prefill"] = True
                vllm_kwargs["enable_prefix_caching"] = False
                vllm_kwargs["gpu_memory_utilization"] = 0.94
                vllm_kwargs["max_num_batched_tokens"] = 2048
                vllm_kwargs["max_num_seqs"] = 256
                vllm_kwargs["cuda_graph_sizes"] = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 1536]
            

                print(f"Enabled vLLM self-spec with n-gram assistance:")
                print(f"  - Method: self_spec_ngram")
                print(f"  - Threshold (ACCUMULATING -> VERIFYING): {args.vllm_sspec_ngram_num_speculative_tokens} tokens")
                print(f"  - N-gram draft tokens per step: {args.vllm_sspec_ngram_num_ngram_draft_tokens} tokens")
                print(f"  - N-gram window size: max={lookup_max}, min={lookup_min}")
                print(f"  - Sink size (streaming cache): {args.vllm_sspec_ngram_sink_size} blocks")
                print(f"  - Recent ratio (streaming cache): {args.vllm_sspec_ngram_recent_ratio} ({args.vllm_sspec_ngram_recent_ratio*100:.1f}%)")
                print(f"  - Block size: {args.vllm_sspec_ngram_block_size}")
                print(f"Enabled vLLM stat logging for metrics collection")

            # Add self-spec with suffix decode config if enabled
            if args.vllm_enable_sspec_suffix:
                sspec_suffix_config = {
                    "method": "self_spec_suffix",
                    "model": None,
                    "num_speculative_tokens": args.vllm_sspec_suffix_num_speculative_tokens,
                    "num_suffix_draft_tokens": args.vllm_sspec_suffix_num_suffix_draft_tokens,
                    "suffix_decoding_max_tree_depth": args.vllm_sspec_suffix_max_tree_depth,
                    "suffix_decoding_max_cached_requests": args.vllm_sspec_suffix_max_cached_requests,
                    "suffix_decoding_max_spec_factor": args.vllm_sspec_suffix_max_spec_factor,
                    "suffix_decoding_min_token_prob": args.vllm_sspec_suffix_min_token_prob,
                }

                vllm_kwargs["speculative_config"] = sspec_suffix_config
                # Add streaming cache parameters
                vllm_kwargs["sink_size"] = args.vllm_sspec_suffix_sink_size
                vllm_kwargs["recent_ratio"] = args.vllm_sspec_suffix_recent_ratio
                vllm_kwargs["block_size"] = args.vllm_sspec_suffix_block_size
                # Enable stat logging to access spec decode metrics
                vllm_kwargs["disable_log_stats"] = False
                # Additional V1-specific parameters
                #vllm_kwargs["enforce_eager"] = False
                vllm_kwargs["enable_chunked_prefill"] = True
                vllm_kwargs["enable_prefix_caching"] = False
                vllm_kwargs["gpu_memory_utilization"] = 0.94
                vllm_kwargs["max_num_batched_tokens"] = 2048
                vllm_kwargs["max_num_seqs"] = 256
                vllm_kwargs["cuda_graph_sizes"] = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 1536]

                print(f"Enabled vLLM self-spec with suffix decoding assistance:")
                print(f"  - Method: self_spec_suffix")
                print(f"  - Threshold (ACCUMULATING -> VERIFYING): {args.vllm_sspec_suffix_num_speculative_tokens} tokens")
                print(f"  - Suffix draft tokens per step: {args.vllm_sspec_suffix_num_suffix_draft_tokens} tokens")
                print(f"  - Max tree depth: {args.vllm_sspec_suffix_max_tree_depth}")
                print(f"  - Max cached requests: {args.vllm_sspec_suffix_max_cached_requests}")
                print(f"  - Max spec factor: {args.vllm_sspec_suffix_max_spec_factor}")
                print(f"  - Min token prob: {args.vllm_sspec_suffix_min_token_prob}")
                print(f"  - Sink size (streaming cache): {args.vllm_sspec_suffix_sink_size} blocks")
                print(f"  - Recent ratio (streaming cache): {args.vllm_sspec_suffix_recent_ratio} ({args.vllm_sspec_suffix_recent_ratio*100:.1f}%)")
                print(f"  - Block size: {args.vllm_sspec_suffix_block_size}")
                print(f"Enabled vLLM stat logging for metrics collection")

            llm = LLM(**vllm_kwargs)
            tokenizer = None
            if args.apply_chat_template:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name_or_path, trust_remote_code=True
                )
        else:
            llm, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\n=== Final Evaluation Results ===")
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

    # Print speculative decoding summary if applicable
    if args.use_sspec and args.sspec_kv_cache != "full":
        spec_results = [
            result for result in results if "speculative_decoding" in result
        ]
        if spec_results:
            # Average the acceptance rates across datasets (excluding "avg" result)
            avg_acceptance_rate = sum(
                result["speculative_decoding"]["acceptance_rate"]
                for result in spec_results
            ) / len(spec_results)
            total_spec_tokens = sum(
                result["speculative_decoding"]["total_speculative_tokens"]
                for result in spec_results
            )
            total_accepted_tokens = sum(
                result["speculative_decoding"]["accepted_speculative_tokens"]
                for result in spec_results
            )

            print("\n=== Speculative Decoding Summary ===")
            print(
                f"Overall Acceptance Rate: {avg_acceptance_rate:.4f} ({avg_acceptance_rate * 100:.2f}%)"
            )
            print(f"Total Speculative Tokens: {total_spec_tokens}")
            print(f"Total Accepted Tokens: {total_accepted_tokens}")
            print("=" * 40)

    # Wandb support removed for simplicity


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))

    # Wandb support removed for simplicity

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        # Do not print the full prompt/question
        # if idx == args.start:
        #     print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "repeat_id",
            "source_idx",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
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
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

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

    # Throughput counters; use engine-side token counts when possible
    total_input_tokens = 0
    total_output_tokens = 0

    # EAGLE3 speculative decoding statistics
    # Normal spec decode metrics (n-gram drafting or traditional spec decode)
    total_draft_tokens = 0  # All tokens proposed by draft model
    total_accepted_tokens = 0  # Tokens actually accepted
    total_spec_verify_count = 0  # Number of verification steps

    # Self-spec metrics (verification of pending tokens)
    self_spec_draft_tokens = 0  # Tokens proposed for self-spec verification
    self_spec_accepted_tokens = 0  # Tokens accepted in self-spec verification
    self_spec_verify_count = 0  # Number of self-spec verifications

    # start inference
    # measure time use
    start_time = time.time()
    engine_time_s = 0.0  # measure only engine generation time for fair comparison
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            time_engine_phase_start = time.time()
            vllm_outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                ),
            )
            engine_time_s += time.time() - time_engine_phase_start

            # Sort by request_id and accumulate token counts from vLLM outputs
            vllm_outputs = sorted(vllm_outputs, key=lambda x: int(x.request_id))
            for i, out in enumerate(vllm_outputs):
                total_input_tokens += len(getattr(out, "prompt_token_ids", []))
                total_output_tokens += len(getattr(out.outputs[0], "token_ids", []))
                print(f"{i}: num outputs: {len(getattr(out.outputs[0], "token_ids", []))}")
                
            outputs = [out.outputs[0].text for out in vllm_outputs]
        elif args.use_sglang:
            time_engine_phase_start = time.time()
            # SGLang Engine.generate() accepts list of prompts and sampling params as dict
            sampling_params = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_tokens_per_call,
                "stop": stop_words,
            }
            sglang_outputs = llm.generate(prompts, sampling_params)
            engine_time_s += time.time() - time_engine_phase_start

            # Process SGLang outputs and count tokens
            # Note: System metrics are printed in real-time by scheduler (every 5s)
            outputs = []
            for idx, out in enumerate(sglang_outputs):
                # SGLang can return dict or object, handle both cases
                if isinstance(out, dict):
                    output_text = out.get("text", str(out))
                    meta_info = out.get("meta_info", {})
                else:
                    output_text = out.text if hasattr(out, "text") else str(out)
                    meta_info = out.meta_info if hasattr(out, "meta_info") else {}

                outputs.append(output_text)

                # Get token counts from meta_info
                if meta_info and isinstance(meta_info, dict):
                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                    completion_tokens_meta = meta_info.get("completion_tokens", 0)
                    spec_verify_ct = meta_info.get("spec_verify_ct", 0)

                    if prompt_tokens > 0:
                        # Use meta_info for input tokens
                        total_input_tokens += prompt_tokens

                        # For output tokens: tokenize the actual output text to get ACTUAL accepted tokens
                        # (completion_tokens_meta includes all draft tokens for EAGLE3)
                        if tokenizer is not None:
                            output_ids = tokenizer(
                                output_text,
                                return_tensors="pt",
                                add_special_tokens=False,
                            )["input_ids"][0]
                            completion_tokens = len(output_ids)
                            total_output_tokens += completion_tokens

                            # EAGLE3 statistics: track draft vs accepted tokens
                            if args.sglang_enable_eagle3 and spec_verify_ct > 0:
                                total_accepted_tokens += completion_tokens
                                # Estimate draft tokens from verification count
                                estimated_draft = (
                                    spec_verify_ct
                                    * args.sglang_speculative_num_draft_tokens
                                )
                                total_draft_tokens += estimated_draft
                                total_spec_verify_count += spec_verify_ct
                        else:
                            total_output_tokens += completion_tokens_meta
                else:
                    # Fallback: use tokenizer if no meta_info
                    if tokenizer is not None and idx < len(prompts):
                        prompt = prompts[idx]
                        input_ids = tokenizer(
                            prompt, return_tensors="pt", add_special_tokens=True
                        )["input_ids"][0]
                        output_ids = tokenizer(
                            output_text, return_tensors="pt", add_special_tokens=False
                        )["input_ids"][0]
                        total_input_tokens += len(input_ids)
                        total_output_tokens += len(output_ids)
        elif args.use_sspec:
            # NOTE: use_sspec requires serve.* modules from external project
            # For vLLM self-spec testing, use --vllm_enable_sspec instead
            raise NotImplementedError(
                "use_sspec requires serve.* modules not included in this repo. "
                "For vLLM self-speculative decoding, use --use_vllm --vllm_enable_sspec instead."
            )
        else:
            time_engine_phase_start = time.time()
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )
            engine_time_s += time.time() - time_engine_phase_start

            # As HF path doesn't expose token ids here, approximate counts via tokenizer
            if tokenizer is not None:
                total_input_tokens += sum(
                    len(tokenizer(p, return_tensors="pt")["input_ids"][0])
                    for p in prompts
                )
                total_output_tokens += sum(
                    len(tokenizer(t, return_tensors="pt")["input_ids"][0])
                    for t in outputs
                )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # Get vLLM speculation statistics after all generation is done (ngram, EAGLE3, self-spec, self-spec-ngram, self-spec-suffix, or suffix)
    if args.use_vllm and (args.vllm_enable_ngram or args.vllm_enable_eagle3 or args.vllm_enable_sspec or args.vllm_enable_sspec_ngram or args.vllm_enable_sspec_suffix or args.vllm_enable_suffix):
        try:
            if args.vllm_enable_ngram:
                spec_method = "ngram"
            elif args.vllm_enable_eagle3:
                spec_method = "EAGLE3"
            elif args.vllm_enable_sspec_ngram:
                spec_method = "self_spec_ngram"
            elif args.vllm_enable_sspec_suffix:
                spec_method = "self_spec_suffix"
            elif args.vllm_enable_suffix:
                spec_method = "suffix"
            else:
                spec_method = "self_specs"

            if hasattr(llm, "get_metrics"):
                metrics = llm.get_metrics()

                # Extract spec decode metrics from Prometheus snapshot
                for metric in metrics:
                    metric_name = (
                        metric.name if hasattr(metric, "name") else str(metric)
                    )

                    # Normal spec decode metrics
                    if metric_name == "vllm:spec_decode_num_draft_tokens" and hasattr(
                        metric, "value"
                    ):
                        total_draft_tokens = int(metric.value)
                    elif metric_name == "vllm:spec_decode_num_accepted_tokens" and hasattr(
                        metric, "value"
                    ):
                        total_accepted_tokens = int(metric.value)
                    elif metric_name == "vllm:spec_decode_num_drafts" and hasattr(
                        metric, "value"
                    ):
                        total_spec_verify_count = int(metric.value)

                    # Self-spec metrics
                    elif metric_name == "vllm:self_spec_num_draft_tokens" and hasattr(
                        metric, "value"
                    ):
                        self_spec_draft_tokens = int(metric.value)
                    elif metric_name == "vllm:self_spec_num_accepted_tokens" and hasattr(
                        metric, "value"
                    ):
                        self_spec_accepted_tokens = int(metric.value)
                    elif metric_name == "vllm:self_spec_num_drafts" and hasattr(
                        metric, "value"
                    ):
                        self_spec_verify_count = int(metric.value)

                # Print normal spec decode metrics (n-gram drafting)
                if total_draft_tokens > 0 and total_accepted_tokens > 0:
                    print(
                        f"\n[vLLM {spec_method}] Successfully retrieved spec decode statistics (N-gram/Draft):"
                    )
                    print(f"  Draft tokens: {total_draft_tokens:,}")
                    print(f"  Accepted tokens: {total_accepted_tokens:,}")
                    print(
                        f"  Acceptance rate: {(total_accepted_tokens/total_draft_tokens)*100:.2f}%"
                    )
                    if total_spec_verify_count > 0:
                        print(f"  Verifications: {total_spec_verify_count:,}")

                # Print self-spec metrics (verification)
                if self_spec_draft_tokens > 0 and self_spec_accepted_tokens > 0:
                    print(
                        f"\n[vLLM {spec_method}] Successfully retrieved self-spec statistics (Verification):"
                    )
                    print(f"  Draft tokens: {self_spec_draft_tokens:,}")
                    print(f"  Accepted tokens: {self_spec_accepted_tokens:,}")
                    print(
                        f"  Acceptance rate: {(self_spec_accepted_tokens/self_spec_draft_tokens)*100:.2f}%"
                    )
                    if self_spec_verify_count > 0:
                        print(f"  Verifications: {self_spec_verify_count:,}")

        except Exception as e:
            if "Stat logging disabled" in str(e):
                print(
                    f"\n[vLLM] Warning: Stat logging is disabled. Cannot retrieve spec decode statistics."
                )
            else:
                print(f"\n[vLLM] Could not retrieve spec decode statistics: {e}")

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # For fair comparison across engines, report time excluding one-time init/capture
    result_json["time_use_in_second"] = engine_time_s
    result_json["time_use_in_minutes"] = (
        f"{int(engine_time_s // 60)}:{int(engine_time_s % 60):02d}"
    )
    # Record engine-only generation time for fair throughput comparison
    result_json["engine_inference_seconds"] = engine_time_s

    # Note: use_sspec metrics removed (requires serve.* modules)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        # Add throughput metrics (engine-only time)
        result_json["throughput"] = {
            "input_tps": (
                (total_input_tokens / engine_time_s) if engine_time_s > 0 else None
            ),
            "output_tps": (
                (total_output_tokens / engine_time_s) if engine_time_s > 0 else None
            ),
            "total_tps": (
                ((total_input_tokens + total_output_tokens) / engine_time_s)
                if engine_time_s > 0
                else None
            ),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "elapsed_seconds": time_use,
            "engine_seconds": engine_time_s,
            "engine": (
                "vllm"
                if args.use_vllm
                else (
                    "sglang"
                    if args.use_sglang
                    else ("sspec" if args.use_sspec else "hf")
                )
            ),
        }

        # Add speculative decoding statistics if available (vLLM or SGLang)
        # Only add if we have REAL data, not fake estimates
        if (args.use_vllm or args.use_sglang):
            # Determine algorithm name
            if args.use_sglang and args.sglang_enable_eagle3:
                algorithm = "EAGLE3"
            elif args.use_vllm and args.vllm_enable_eagle3:
                algorithm = "EAGLE3"
            elif args.use_vllm and args.vllm_enable_ngram:
                algorithm = "ngram"
            elif args.use_vllm and args.vllm_enable_sspec_ngram:
                algorithm = "self_spec_ngram"
            elif args.use_vllm and args.vllm_enable_sspec_suffix:
                algorithm = "self_spec_suffix"
            elif args.use_vllm and args.vllm_enable_sspec:
                algorithm = "self_specs"
            elif args.use_vllm and args.vllm_enable_suffix:
                algorithm = "suffix"
            else:
                algorithm = "Unknown"

            # Add normal spec decode metrics (n-gram drafting or traditional spec decode)
            if total_draft_tokens > 0 and total_accepted_tokens > 0:
                acceptance_rate = (
                    total_accepted_tokens / total_draft_tokens
                    if total_draft_tokens > 0
                    else 0.0
                )

                result_json["speculative_decoding"] = {
                    "algorithm": algorithm,
                    "total_draft_tokens": total_draft_tokens,
                    "total_accepted_tokens": total_accepted_tokens,
                    "total_rejected_tokens": total_draft_tokens - total_accepted_tokens,
                    "acceptance_rate": acceptance_rate,
                    "acceptance_rate_percentage": acceptance_rate * 100,
                    "verification_count": total_spec_verify_count,
                    "avg_draft_per_verification": (
                        total_draft_tokens / total_spec_verify_count
                        if total_spec_verify_count > 0
                        else 0.0
                    ),
                    "avg_accepted_per_verification": (
                        total_accepted_tokens / total_spec_verify_count
                        if total_spec_verify_count > 0
                        else 0.0
                    ),
                    "mean_acceptance_length": (
                        1 + (total_accepted_tokens / total_spec_verify_count)
                        if total_spec_verify_count > 0
                        else 0.0
                    ),
                }

                # Add algorithm-specific notes
                if args.use_sglang and args.sglang_enable_eagle3:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "draft_tokens estimated from spec_verify_ct * num_draft_tokens config"
                elif args.use_vllm and args.vllm_enable_eagle3:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "EAGLE3 speculative decoding"
                elif args.use_vllm and args.vllm_enable_ngram:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "ngram prompt lookup speculation"
                elif args.use_vllm and args.vllm_enable_sspec_ngram:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "self-speculative decoding with n-gram draft assistance (n-gram drafting metrics)"
                elif args.use_vllm and args.vllm_enable_sspec_suffix:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "self-speculative decoding with suffix decoding draft assistance (suffix drafting metrics)"
                elif args.use_vllm and args.vllm_enable_suffix:
                    result_json["speculative_decoding"][
                        "note"
                    ] = "suffix decoding with frequency-based speculation"

                print(f"\n[{algorithm} Spec Decode Statistics (N-gram/Draft)]")
                print(f"  Draft tokens: {total_draft_tokens:,}")
                print(f"  Accepted tokens: {total_accepted_tokens:,}")
                print(f"  Rejected tokens: {total_draft_tokens - total_accepted_tokens:,}")
                print(f"  Acceptance rate: {acceptance_rate * 100:.2f}%")
                if total_spec_verify_count > 0:
                    print(f"  Verifications: {total_spec_verify_count:,}")
                    print(
                        f"  Avg draft per verification: {total_draft_tokens / total_spec_verify_count:.2f}"
                    )
                    print(
                        f"  Avg accepted per verification: {total_accepted_tokens / total_spec_verify_count:.2f}"
                    )
                    print(
                        f"  Mean acceptance length: {1 + (total_accepted_tokens / total_spec_verify_count):.2f}"
                    )

            # Add self-spec metrics (verification of pending tokens)
            if self_spec_draft_tokens > 0 and self_spec_accepted_tokens > 0:
                self_spec_acceptance_rate = (
                    self_spec_accepted_tokens / self_spec_draft_tokens
                    if self_spec_draft_tokens > 0
                    else 0.0
                )

                result_json["self_spec_verification"] = {
                    "algorithm": algorithm,
                    "total_draft_tokens": self_spec_draft_tokens,
                    "total_accepted_tokens": self_spec_accepted_tokens,
                    "total_rejected_tokens": self_spec_draft_tokens - self_spec_accepted_tokens,
                    "acceptance_rate": self_spec_acceptance_rate,
                    "acceptance_rate_percentage": self_spec_acceptance_rate * 100,
                    "verification_count": self_spec_verify_count,
                    "avg_draft_per_verification": (
                        self_spec_draft_tokens / self_spec_verify_count
                        if self_spec_verify_count > 0
                        else 0.0
                    ),
                    "avg_accepted_per_verification": (
                        self_spec_accepted_tokens / self_spec_verify_count
                        if self_spec_verify_count > 0
                        else 0.0
                    ),
                    "mean_acceptance_length": (
                        1 + (self_spec_accepted_tokens / self_spec_verify_count)
                        if self_spec_verify_count > 0
                        else 0.0
                    ),
                    "note": "Self-speculative verification metrics (pending token verification)"
                }

                print(f"\n[{algorithm} Self-Spec Statistics (Verification)]")
                print(f"  Draft tokens: {self_spec_draft_tokens:,}")
                print(f"  Accepted tokens: {self_spec_accepted_tokens:,}")
                print(f"  Rejected tokens: {self_spec_draft_tokens - self_spec_accepted_tokens:,}")
                print(f"  Acceptance rate: {self_spec_acceptance_rate * 100:.2f}%")
                if self_spec_verify_count > 0:
                    print(f"  Verifications: {self_spec_verify_count:,}")
                    print(
                        f"  Avg draft per verification: {self_spec_draft_tokens / self_spec_verify_count:.2f}"
                    )
                    print(
                        f"  Avg accepted per verification: {self_spec_accepted_tokens / self_spec_verify_count:.2f}"
                    )
                    print(
                        f"  Mean acceptance length: {1 + (self_spec_accepted_tokens / self_spec_verify_count):.2f}"
                    )

        json.dump(result_json, f, indent=4)

    # Wandb logging removed for simplicity

    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
