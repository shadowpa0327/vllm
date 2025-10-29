#!/bin/bash
# Benchmark script for self-speculative decoding (self_spec_ngram)
# Based on the suffix/ngram benchmark pattern

set -e

# Configuration
spec_tokens=8  # Self-spec threshold (ACCUMULATING → VERIFYING)
ngram_draft_tokens=3  # N-gram draft tokens per step
concurrency=256

# Set environment variables for V1 engine with self-spec
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

# ============================================================
# SELF-SPECULATIVE DECODING CONFIGURATION
# ============================================================

# Baseline self_specs (no n-gram assistance)
# spec_config=$(cat <<EOF
# {
#     "method": "self_specs",
#     "num_speculative_tokens": ${spec_tokens}
# }
# EOF
# )

# Baseline self_specs (no n-gram assistance)
spec_config=$(cat <<EOF
{
    "method": "self_specs",
    "num_speculative_tokens": ${spec_tokens}
}
EOF
)


# spec_config=$(cat <<EOF
# {
#     "method": "ngram",
#     "num_speculative_tokens": ${spec_tokens},
#     "prompt_lookup_min": 3,
#     "prompt_lookup_max": 5
# }
# EOF
# )


# Alternative: self_spec_ngram with variable n-gram window
# spec_config=$(cat <<EOF
# {
#     "method": "self_spec_ngram",
#     "num_speculative_tokens": ${spec_tokens},
#     "num_ngram_draft_tokens": ${ngram_draft_tokens},
#     "prompt_lookup_min": 3,
#     "prompt_lookup_max": 5
# }
# EOF
# )

echo "============================================================"
echo "SELF-SPECULATIVE DECODING BENCHMARK"
echo "============================================================"
echo "Configuration:"
echo "  Method: self_specs (baseline, no n-gram)"
echo "  Self-spec threshold: ${spec_tokens} tokens"
echo "  Concurrency: ${concurrency}"
echo "============================================================"

# Start vLLM server with self-speculative decoding
echo "Starting vLLM server with self-speculative decoding..."


# spec
VLLM_TORCH_PROFILER_DIR=./vllm_profile vllm serve Qwen/Qwen3-8B \
    --trust-remote-code \
    ${spec_config:+--speculative-config "${spec_config}"} \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --block-size 1 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --disable-log-requests &

# # navie vllm
# VLLM_TORCH_PROFILER_DIR=./vllm_profile vllm serve Qwen/Qwen3-8B \
#     --trust-remote-code \
#     --max-model-len 40960 \
#     --max-num-batched-tokens 8192 \
#     --block-size 16 \
#     --gpu-memory-utilization 0.9 \
#     --no-enable-prefix-caching \
#     --disable-log-requests &

pid=$!

# Wait for server to start
echo "Waiting for the server to start..."
while ! curl -s http://localhost:8000/v1/models >/dev/null; do
    echo "Still waiting..."
    sleep 5
done
echo "Server started successfully!"

# Download spec_bench dataset if not exists
if [ ! -f "question.jsonl" ]; then
    echo "Downloading spec_bench dataset..."
    wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
fi

echo ""
echo "============================================================"
echo "Running benchmark with spec_bench dataset..."
echo "============================================================"

# Run benchmark with spec_bench dataset
vllm bench serve \
    --model Qwen/Qwen3-8B \
    --dataset-name custom \
    --dataset-path prompt_for_testing.jsonl \
    --num-prompts 1000 \
    --max-concurrency ${concurrency} \
    --no-oversample \
    --ignore-eos
    # --dataset-name spec_bench \
    # --dataset-path question.jsonl \
    # --spec-bench-output-len 256 \
    # --max-concurrency ${concurrency} \
    # --no-oversample \
    # --ignore-eos

echo ""
echo "============================================================"
echo "SPECULATIVE DECODING METRICS FROM PROMETHEUS"
echo "============================================================"

# Get speculative decoding metrics
echo ""
echo "Draft tokens proposed:"
curl -s http://localhost:8000/metrics 2>/dev/null | grep spec_decode_num_draft_tokens_total{

echo ""
echo "Tokens accepted:"
curl -s http://localhost:8000/metrics 2>/dev/null | grep spec_decode_num_accepted_tokens_total{

echo ""
echo "Per-position acceptance rates:"
curl -s http://localhost:8000/metrics 2>/dev/null | grep spec_decode_num_accepted_tokens_per_pos{

echo ""
echo "Number of drafts:"
curl -s http://localhost:8000/metrics 2>/dev/null | grep spec_decode_num_drafts_total{

echo ""
echo "============================================================"

# Kill the server
echo "Shutting down server..."
kill ${pid}
wait ${pid} 2>/dev/null || true

echo "Benchmark completed!"
