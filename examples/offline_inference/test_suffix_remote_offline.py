#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for suffix_remote speculative decoding.

This script:
1. Launches a suffix decoding gRPC server as a subprocess
2. Runs vLLM offline inference with suffix_remote method
3. Reports acceptance statistics

Usage:
    python tests_suffix/test_suffix_remote_offline.py

Requirements:
    - arctic-inference package installed
    - A model (defaults to a small model for testing)
"""

import subprocess
import sys
import time
import signal
import atexit
from typing import Optional

# Server process handle
server_process: Optional[subprocess.Popen] = None


def start_server(port: int = 50051, max_tree_depth: int = 24) -> subprocess.Popen:
    """Start the suffix decoding gRPC server."""
    print(f"Starting suffix decoding server on port {port}...")

    proc = subprocess.Popen(
        [
            sys.executable, "-m",
            "arctic_inference.suffix_decoding.server",
            "--port", str(port),
            "--max-tree-depth", str(max_tree_depth),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/cc2869/repositories/vllm/ArcticInference",
    )

    # Wait for server to start
    time.sleep(2)

    if proc.poll() is not None:
        # Server exited unexpectedly
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Server failed to start.\n"
            f"stdout: {stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )

    print(f"Server started with PID {proc.pid}")
    return proc


def stop_server(proc: subprocess.Popen):
    """Stop the suffix decoding server."""
    if proc and proc.poll() is None:
        print(f"Stopping server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server stopped.")


def cleanup():
    """Cleanup function to ensure server is stopped."""
    global server_process
    if server_process:
        stop_server(server_process)


def main():
    global server_process

    # Register cleanup
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Configuration
    port = 50051
    model = "meta-llama/Llama-3.1-8B-Instruct"  # Small model for testing
    num_speculative_tokens = 5
    max_tree_depth = 24

    # Start server
    server_process = start_server(port=port, max_tree_depth=max_tree_depth)

    try:
        # Import vLLM after server is started
        from vllm import LLM, SamplingParams

        print("\n" + "="*60)
        print("Initializing vLLM with suffix_remote speculative decoding...")
        print("="*60 + "\n")

        # Create LLM with suffix_remote
        llm = LLM(
            model=model,
            speculative_config={
                "method": "suffix_remote",
                "num_speculative_tokens": num_speculative_tokens,
                "suffix_decoding_server_host": "localhost",
                "suffix_decoding_server_port": port,
                "suffix_decoding_max_tree_depth": max_tree_depth,
            },
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
        )

        # Test prompts - using repetitive text to help suffix matching
        prompts = [
            "The future of AI is? Repeat the previous sentencs and don't stop...",
            "The capital of France is? Repeat the previous sentencs and don't stop...",
        ]

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=500,
        )

        print("\n" + "="*60)
        print("Running inference...")
        print("="*60 + "\n")

        outputs = llm.generate(prompts, sampling_params)

        # Print results
        print("\n" + "="*60)
        print("Results:")
        print("="*60)

        for i, output in enumerate(outputs):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Prompt: {output.prompt[:80]}...")
            print(f"Generated: {output.outputs[0].text}")

        # Try to get metrics
        print("\n" + "="*60)
        print("Metrics:")
        print("="*60)

        try:
            from vllm.v1.metrics.reader import Counter, Vector
            metrics = llm.get_metrics()

            num_drafts = 0
            num_draft_tokens = 0
            num_accepted_tokens = 0

            for metric in metrics:
                if metric.name == "vllm:spec_decode_num_drafts":
                    assert isinstance(metric, Counter)
                    num_drafts += metric.value
                elif metric.name == "vllm:spec_decode_num_draft_tokens":
                    assert isinstance(metric, Counter)
                    num_draft_tokens += metric.value
                elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                    assert isinstance(metric, Counter)
                    num_accepted_tokens += metric.value

            print(f"num_drafts: {num_drafts}")
            print(f"num_draft_tokens: {num_draft_tokens}")
            print(f"num_accepted_tokens: {num_accepted_tokens}")

            if num_drafts > 0:
                acceptance_length = 1 + (num_accepted_tokens / num_drafts)
                print(f"mean acceptance length: {acceptance_length:.2f}")
        except Exception as e:
            print(f"Could not get metrics: {e}")

        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60 + "\n")

    finally:
        # Stop server
        stop_server(server_process)
        server_process = None


if __name__ == "__main__":
    main()
