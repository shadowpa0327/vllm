#!/usr/bin/env python3
"""
Benchmark script for ParallelSuffixDecodingProposer.propose().

This script measures the latency of the propose() method after pre-initializing
the suffix cache with sample trees. It also supports torch.profiler for
detailed profiling with Chrome trace export.

Usage:
    # Basic benchmark
    python benchmark_suffix_decoding_proposer.py

    # With profiling
    python benchmark_suffix_decoding_proposer.py --profile

    # Custom parameters
    python benchmark_suffix_decoding_proposer.py --batch-size 64 --num-trees 100 --warmup 10 --iterations 100
"""

import argparse
import time
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

# =============================================================================
# Import Note: Why we import from arctic_inference instead of vLLM
# =============================================================================
# The vLLM proposer (ParallelSuffixDecodingProposer) is located at:
#   - vllm/v1/spec_decode/suffix_decoding_parallel.py
#
# However, importing from vLLM triggers heavy dependencies (CUDA, torch.distributed,
# model configs) that cause circular imports when running standalone benchmarks.
# Even vLLM itself uses lazy imports for this reason (see suffix_decoding.py:23).
#
# We import directly from arctic_inference, which provides the underlying cache:
#   - ParallelSuffixDecodingCache: arctic_inference/suffix_decoding/parallel_cache.py
# =============================================================================
try:
    from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
except ImportError:
    print("ERROR: Could not import ParallelSuffixDecodingCache from arctic_inference.")
    print("Make sure ArcticInference is installed: pip install -e third_party/ArcticInference_srt")
    sys.exit(1)


class MockInputBatch:
    """
    Mock InputBatch that mimics the real InputBatch interface
    used by ParallelSuffixDecodingProposer.
    """

    def __init__(
        self,
        batch_size: int,
        max_model_len: int = 4096,
        prompt_len: int = 512,
        num_generated: int = 100,
    ):
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.prompt_len = prompt_len
        self.num_generated = num_generated

        # Generate request IDs
        self.req_ids = [f"req_{i}" for i in range(batch_size)]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}

        # Token IDs CPU tensor (prompt + generated tokens)
        total_tokens = prompt_len + num_generated
        self.token_ids_cpu = np.zeros((batch_size, max_model_len), dtype=np.int32)

        # Fill with varied tokens per request
        for i in range(batch_size):
            # Prompt tokens
            self.token_ids_cpu[i, :prompt_len] = np.arange(
                i * 1000, i * 1000 + prompt_len, dtype=np.int32
            )
            # Generated tokens
            self.token_ids_cpu[i, prompt_len:total_tokens] = np.arange(
                i * 1000 + prompt_len,
                i * 1000 + total_tokens,
                dtype=np.int32
            )

        # Number of tokens (prompt + generated)
        self.num_tokens_no_spec = np.full(batch_size, total_tokens, dtype=np.int32)

        # Number of prompt tokens
        self.num_prompt_tokens = np.full(batch_size, prompt_len, dtype=np.int32)

        # Requests that don't support spec decode (empty for benchmark)
        self.spec_decode_unsupported_reqs: set = set()

        # Prompt hashes (optional, for hash-based tree sharing)
        self.prompt_hashes: dict = {}

    def add_generated_tokens(self, num_new_tokens: int = 1):
        """Simulate adding new generated tokens."""
        for i in range(self.batch_size):
            current_len = self.num_tokens_no_spec[i]
            new_len = min(current_len + num_new_tokens, self.max_model_len)
            # Add new tokens
            for j in range(current_len, new_len):
                self.token_ids_cpu[i, j] = i * 1000 + j
            self.num_tokens_no_spec[i] = new_len


class StandaloneProposer:
    """
    Standalone proposer that mimics ParallelSuffixDecodingProposer behavior
    using ParallelSuffixDecodingCache directly.
    """

    def __init__(
        self,
        num_threads: int = 8,
        parallel_threshold: int = 8,
        max_tree_depth: int = 64,
        num_speculative_tokens: int = 5,
        max_model_len: int = 4096,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_model_len = max_model_len

        # Initialize the suffix cache directly
        self.suffix_cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=num_threads,
            parallel_threshold=parallel_threshold,
        )

        print(f"Initialized StandaloneProposer with num_threads={num_threads}, "
              f"parallel_threshold={parallel_threshold}")

    def propose(
        self,
        input_batch: MockInputBatch,
        sampled_token_ids: list,
    ) -> list:
        """
        Propose speculative tokens for each request in the input batch.
        This mimics the behavior of ParallelSuffixDecodingProposer.propose().
        """
        # Collect data for batch operations
        req_ids_to_add_tokens = []
        tokens_to_add = []
        req_ids_to_speculate = []
        contexts_to_speculate = []
        max_spec_tokens_list = []
        input_indices_with_drafts = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue

            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                continue

            index = input_batch.req_id_to_index[req_id]

            # Start new requests if needed
            if req_id not in self.suffix_cache.active_requests:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                pre_computed_hash = input_batch.prompt_hashes.get(req_id)
                self.suffix_cache.start_request(
                    req_id, prompt_token_ids, pre_computed_hash=pre_computed_hash
                )

            req_ids_to_add_tokens.append(req_id)
            tokens_to_add.append(sampled_ids)

            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]

            req_ids_to_speculate.append(req_id)
            contexts_to_speculate.append(pattern)
            max_spec_tokens_list.append(
                min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
            )
            input_indices_with_drafts.append(i)

        # BATCH ADD TOKENS
        if req_ids_to_add_tokens:
            self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

        # BATCH SPECULATE
        drafts = []
        if req_ids_to_speculate:
            min_max_spec_tokens = min(max_spec_tokens_list) if max_spec_tokens_list else self.num_speculative_tokens
            drafts = self.suffix_cache.batch_speculate(
                req_ids=req_ids_to_speculate,
                contexts=contexts_to_speculate,
                max_spec_tokens=min_max_spec_tokens,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

        # Build result list
        draft_token_ids = []
        draft_idx = 0
        for i in range(len(sampled_token_ids)):
            if i in input_indices_with_drafts:
                draft_token_ids.append(drafts[draft_idx].token_ids)
                draft_idx += 1
            else:
                draft_token_ids.append([])

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def get_stats(self) -> dict:
        return self.suffix_cache.get_stats()


def create_proposer(
    num_threads: int = 8,
    parallel_threshold: int = 8,
    max_tree_depth: int = 64,
    num_speculative_tokens: int = 5,
) -> StandaloneProposer:
    """Create a StandaloneProposer for benchmarking."""
    return StandaloneProposer(
        num_threads=num_threads,
        parallel_threshold=parallel_threshold,
        max_tree_depth=max_tree_depth,
        num_speculative_tokens=num_speculative_tokens,
    )


def pre_init_cache(
    proposer: StandaloneProposer,
    num_trees: int,
    prompt_len: int = 512,
    response_len: int = 500,
):
    """
    Pre-initialize the suffix cache with trees.

    This simulates having processed previous requests and built up
    suffix trees with their prompt + response patterns.
    """
    print(f"Pre-initializing {num_trees} suffix trees...")

    for i in range(num_trees):
        req_id = f"preinit_req_{i}"

        # Generate unique prompt
        prompt = np.arange(i * 10000, i * 10000 + prompt_len, dtype=np.int32)

        # Start request (builds initial tree from prompt)
        proposer.suffix_cache.start_request(req_id, prompt)

        # Add response tokens to build up the tree
        # Add in chunks to simulate real token generation
        chunk_size = 50
        for start in range(0, response_len, chunk_size):
            end = min(start + chunk_size, response_len)
            tokens = np.arange(
                i * 10000 + prompt_len + start,
                i * 10000 + prompt_len + end,
                dtype=np.int32
            )
            proposer.suffix_cache.add_tokens(req_id, tokens)

    print(f"Pre-initialized {num_trees} trees")
    stats = proposer.get_stats()
    print(f"Cache stats: {stats}")


def benchmark_propose(
    proposer: StandaloneProposer,
    input_batch: MockInputBatch,
    num_warmup: int = 10,
    num_iterations: int = 100,
    tokens_per_step: int = 1,
) -> dict:
    """
    Benchmark the propose() method.

    Returns timing statistics.
    """
    print(f"\nBenchmarking propose() with batch_size={input_batch.batch_size}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iterations}")

    # Prepare sampled token IDs (simulating the model's output)
    def get_sampled_tokens():
        return [[np.random.randint(1, 10000)] for _ in range(input_batch.batch_size)]

    # Warmup
    print("  Running warmup...")
    for _ in range(num_warmup):
        sampled_token_ids = get_sampled_tokens()
        _ = proposer.propose(input_batch, sampled_token_ids)
        # Simulate adding tokens to batch
        input_batch.add_generated_tokens(tokens_per_step)

    # Reset batch for actual benchmark
    input_batch = MockInputBatch(
        batch_size=input_batch.batch_size,
        max_model_len=input_batch.max_model_len,
        prompt_len=input_batch.prompt_len,
        num_generated=input_batch.num_generated,
    )

    # Benchmark
    print("  Running benchmark...")
    latencies = []
    draft_lengths = []

    for _ in range(num_iterations):
        sampled_token_ids = get_sampled_tokens()

        start = time.perf_counter()
        draft_token_ids = proposer.propose(input_batch, sampled_token_ids)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms
        draft_lengths.append([len(d) for d in draft_token_ids])

        # Simulate adding tokens
        input_batch.add_generated_tokens(tokens_per_step)

    # Calculate statistics
    latencies = np.array(latencies)
    results = {
        "batch_size": input_batch.batch_size,
        "num_iterations": num_iterations,
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p90_latency_ms": np.percentile(latencies, 90),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_draft_length": np.mean([np.mean(dl) for dl in draft_lengths]),
    }

    return results


def profile_propose(
    proposer: StandaloneProposer,
    input_batch: MockInputBatch,
    num_warmup: int = 5,
    num_iterations: int = 20,
    output_dir: str = "./profiler_output",
):
    """
    Profile the propose() method using torch.profiler and export Chrome trace.
    """
    print(f"\nProfiling propose() with torch.profiler")
    print(f"  Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    def get_sampled_tokens():
        return [[np.random.randint(1, 10000)] for _ in range(input_batch.batch_size)]

    # Warmup outside profiler
    print("  Running warmup...")
    for _ in range(num_warmup):
        sampled_token_ids = get_sampled_tokens()
        _ = proposer.propose(input_batch, sampled_token_ids)
        input_batch.add_generated_tokens(1)

    # Reset batch
    input_batch = MockInputBatch(
        batch_size=input_batch.batch_size,
        max_model_len=input_batch.max_model_len,
        prompt_len=input_batch.prompt_len,
        num_generated=input_batch.num_generated,
    )

    # Profile with torch.profiler - don't use on_trace_ready so we can export manually
    print("  Running profiler...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_iterations):
            sampled_token_ids = get_sampled_tokens()
            _ = proposer.propose(input_batch, sampled_token_ids)
            input_batch.add_generated_tokens(1)

    # Export Chrome trace
    chrome_trace_path = os.path.join(output_dir, "chrome_trace.json")
    prof.export_chrome_trace(chrome_trace_path)
    print(f"  Chrome trace exported to: {chrome_trace_path}")

    # Print profiler summary
    print("\n" + "=" * 80)
    print("Profiler Summary (CPU time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    return chrome_trace_path


def run_batch_size_sweep(
    num_threads: int = 8,
    parallel_threshold: int = 8,
    num_trees: int = 100,
    prompt_len: int = 512,
    response_len: int = 500,
    batch_sizes: list = None,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Run benchmarks across different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    print("=" * 80)
    print("Batch Size Sweep Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  num_threads: {num_threads}")
    print(f"  parallel_threshold: {parallel_threshold}")
    print(f"  num_trees: {num_trees}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  response_len: {response_len}")
    print(f"  batch_sizes: {batch_sizes}")

    # Create proposer
    proposer = create_proposer(
        num_threads=num_threads,
        parallel_threshold=parallel_threshold,
    )

    # Pre-initialize cache
    pre_init_cache(proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        input_batch = MockInputBatch(
            batch_size=batch_size,
            prompt_len=512,
            num_generated=100,
        )

        result = benchmark_propose(
            proposer=proposer,
            input_batch=input_batch,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )
        results.append(result)

        # Clear active requests for this batch
        for req_id in list(proposer.suffix_cache.active_requests):
            if req_id.startswith("req_"):
                proposer.suffix_cache.stop_request(req_id)

    # Print results table
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Batch Size':>12} | {'Mean (ms)':>12} | {'P50 (ms)':>12} | {'P90 (ms)':>12} | {'P99 (ms)':>12} | {'Avg Draft':>12}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['batch_size']:>12} | "
            f"{r['mean_latency_ms']:>12.3f} | "
            f"{r['p50_latency_ms']:>12.3f} | "
            f"{r['p90_latency_ms']:>12.3f} | "
            f"{r['p99_latency_ms']:>12.3f} | "
            f"{r['avg_draft_length']:>12.2f}"
        )

    # Calculate per-request latency
    print("\n" + "=" * 80)
    print("Per-Request Latency")
    print("=" * 80)
    print(f"{'Batch Size':>12} | {'Per-Req (ms)':>14}")
    print("-" * 30)
    for r in results:
        per_req = r['mean_latency_ms'] / r['batch_size']
        print(f"{r['batch_size']:>12} | {per_req:>14.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ParallelSuffixDecodingProposer.propose()"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for benchmarking (default: 32)"
    )
    parser.add_argument(
        "--num-trees", type=int, default=100,
        help="Number of trees to pre-initialize (default: 100)"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=512,
        help="Prompt length for pre-initialized trees (default: 512)"
    )
    parser.add_argument(
        "--response-len", type=int, default=500,
        help="Response length for pre-initialized trees (default: 500)"
    )
    parser.add_argument(
        "--num-threads", type=int, default=8,
        help="Number of threads for parallel operations (default: 8)"
    )
    parser.add_argument(
        "--parallel-threshold", type=int, default=8,
        help="Minimum batch size for parallelization (default: 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable torch.profiler and export Chrome trace"
    )
    parser.add_argument(
        "--profile-output", type=str, default="./profiler_output",
        help="Output directory for profiler results (default: ./profiler_output)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run batch size sweep benchmark"
    )

    args = parser.parse_args()

    if args.sweep:
        # Run batch size sweep
        run_batch_size_sweep(
            num_threads=args.num_threads,
            parallel_threshold=args.parallel_threshold,
            num_trees=args.num_trees,
            prompt_len=args.prompt_len,
            response_len=args.response_len,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
        )
    else:
        # Single batch size benchmark
        print("=" * 80)
        print("ParallelSuffixDecodingProposer Benchmark")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  num_trees: {args.num_trees}")
        print(f"  prompt_len: {args.prompt_len}")
        print(f"  response_len: {args.response_len}")

        # Create proposer
        proposer = create_proposer(
            num_threads=args.num_threads,
            parallel_threshold=args.parallel_threshold,
        )

        # Pre-initialize cache
        pre_init_cache(proposer, num_trees=args.num_trees, prompt_len=args.prompt_len, response_len=args.response_len)

        # Create input batch
        input_batch = MockInputBatch(
            batch_size=args.batch_size,
            prompt_len=512,
            num_generated=100,
        )

        if args.profile:
            # Run profiling
            chrome_trace_path = profile_propose(
                proposer=proposer,
                input_batch=input_batch,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                output_dir=args.profile_output,
            )
            print(f"\nProfile complete. View trace at: {chrome_trace_path}")
            print("Open in Chrome: chrome://tracing/")
        else:
            # Run benchmark
            result = benchmark_propose(
                proposer=proposer,
                input_batch=input_batch,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
            )

            # Print results
            print("\n" + "=" * 80)
            print("Benchmark Results")
            print("=" * 80)
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
