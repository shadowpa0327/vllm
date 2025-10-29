#!/bin/bash
# Math benchmark script with vLLM server
# Supports various speculative decoding methods: baseline, self_specs, ngram, self_spec_ngram

set -e

# ============================================================
# CONFIGURATION
# ============================================================

# Model configuration
#MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="Qwen/Qwen3-8B"

# Benchmark configuration
DATASETS="aime24"  # Comma-separated list: gsm8k,math,minerva_math
NUM_SAMPLES=100  # Number of test samples (-1 for all)
MAX_TOKENS=  # Max tokens per generation
PROMPT_TYPE="tool-integrated"  # Options: tool-integrated, cot, pal

# Request control
REQUEST_RATE="inf"  # Request rate (inf for unlimited, or number like 10)
MAX_CONCURRENCY=128  # Maximum concurrent requests

# Speculative decoding configuration
SPEC_METHOD="baseline"  # Options: baseline, self_specs, ngram, self_spec_ngram
SPEC_TOKENS=8  # Self-spec threshold or ngram draft count
NGRAM_DRAFT_TOKENS=3  # N-gram draft tokens per step (for self_spec_ngram)

# Server configuration
HOST="127.0.0.1"
PORT=8000
GPU_MEMORY_UTIL=0.9

# Output configuration
OUTPUT_DIR="./math_outputs"
SAVE_OUTPUTS=true

# Set environment variables for V1 engine (required for self_specs and self_spec_ngram)
if [[ "$SPEC_METHOD" == "self_specs" ]] || [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
    export VLLM_ENABLE_V1_MULTIPROCESSING=1
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASHINFER
fi

# ============================================================
# BUILD SPECULATIVE CONFIG
# ============================================================

spec_config=""
spec_args=""

case "$SPEC_METHOD" in
    baseline)
        echo "Running baseline (no speculative decoding)"
        ;;
    
    self_specs)
        spec_config=$(cat <<EOF
{
    "method": "self_specs",
    "num_speculative_tokens": ${SPEC_TOKENS}
}
EOF
)
        spec_args="--speculative-config"
        ;;
    
    ngram)
        spec_config=$(cat <<EOF
{
    "method": "ngram",
    "num_speculative_tokens": ${SPEC_TOKENS},
    "ngram_prompt_lookup_min": 1,
    "ngram_prompt_lookup_max": 4
}
EOF
)
        spec_args="--speculative-config"
        ;;
    
    self_spec_ngram)
        spec_config=$(cat <<EOF
{
    "method": "self_spec_ngram",
    "num_speculative_tokens": ${SPEC_TOKENS},
    "num_ngram_draft_tokens": ${NGRAM_DRAFT_TOKENS},
    "prompt_lookup_min": ${NGRAM_DRAFT_TOKENS},
    "prompt_lookup_max": ${NGRAM_DRAFT_TOKENS}
}
EOF
)
        spec_args="--speculative-config"
        ;;
    
    *)
        echo "Unknown spec method: $SPEC_METHOD"
        exit 1
        ;;
esac

# ============================================================
# PRINT CONFIGURATION
# ============================================================

echo "============================================================"
echo "MATH BENCHMARK WITH VLLM SERVER"
echo "============================================================"
echo "Model: ${MODEL}"
echo "Datasets: ${DATASETS}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Prompt type: ${PROMPT_TYPE}"
echo ""
echo "Speculative Method: ${SPEC_METHOD}"
if [[ "$SPEC_METHOD" != "baseline" ]]; then
    echo "  Spec tokens: ${SPEC_TOKENS}"
    if [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
        echo "  N-gram draft tokens: ${NGRAM_DRAFT_TOKENS}"
    fi
fi
echo ""
echo "Request rate: ${REQUEST_RATE} req/s"
echo "Max concurrency: ${MAX_CONCURRENCY}"
echo "============================================================"

# ============================================================
# START VLLM SERVER
# ============================================================

echo ""
echo "Starting vLLM server..."

# Build server command
server_cmd="vllm serve ${MODEL} --host ${HOST} --port ${PORT} --trust-remote-code"

# Add common server arguments
if [[ "$SPEC_METHOD" == "self_specs" ]] || [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
    # V1 engine specific settings
    server_cmd="${server_cmd} --max-model-len 40960"
    server_cmd="${server_cmd} --max-num-batched-tokens 8192"
    server_cmd="${server_cmd} --block-size 1"
    server_cmd="${server_cmd} --gpu-memory-utilization ${GPU_MEMORY_UTIL}"
    server_cmd="${server_cmd} --disable-log-requests"
else
    # V0 engine settings
    server_cmd="${server_cmd} --gpu-memory-utilization ${GPU_MEMORY_UTIL}"
    server_cmd="${server_cmd} --disable-log-requests"
fi

# Add speculative config if needed
if [[ -n "$spec_args" ]]; then
    server_cmd="${server_cmd} ${spec_args} '${spec_config}'"
fi

# Start server in background
echo "Command: ${server_cmd}"
eval "${server_cmd}" &
pid=$!

# Wait for server to start
echo ""
echo "Waiting for server to start..."
max_wait=300  # 5 minutes
waited=0
while ! curl -s http://${HOST}:${PORT}/v1/models >/dev/null; do
    if [ $waited -ge $max_wait ]; then
        echo "Error: Server failed to start within ${max_wait} seconds"
        kill ${pid} 2>/dev/null || true
        exit 1
    fi
    echo "Still waiting... (${waited}s)"
    sleep 5
    waited=$((waited + 5))
done

echo "Server started successfully!"

# ============================================================
# RUN MATH BENCHMARK
# ============================================================

echo ""
echo "============================================================"
echo "RUNNING MATH BENCHMARK"
echo "============================================================"

# Build benchmark command
benchmark_cmd="python math_eval_server.py"
benchmark_cmd="${benchmark_cmd} --data_names ${DATASETS}"
benchmark_cmd="${benchmark_cmd} --model_name_or_path ${MODEL}"
benchmark_cmd="${benchmark_cmd} --host ${HOST}"
benchmark_cmd="${benchmark_cmd} --port ${PORT}"
benchmark_cmd="${benchmark_cmd} --num_test_sample ${NUM_SAMPLES}"
benchmark_cmd="${benchmark_cmd} --max_tokens_per_call ${MAX_TOKENS}"
benchmark_cmd="${benchmark_cmd} --prompt_type ${PROMPT_TYPE}"
benchmark_cmd="${benchmark_cmd} --request_rate ${REQUEST_RATE}"
benchmark_cmd="${benchmark_cmd} --max_concurrency ${MAX_CONCURRENCY}"
benchmark_cmd="${benchmark_cmd} --output_dir ${OUTPUT_DIR}"
benchmark_cmd="${benchmark_cmd} --temperature 0.0"
benchmark_cmd="${benchmark_cmd} --seed 0"

if [[ "$SAVE_OUTPUTS" == true ]]; then
    benchmark_cmd="${benchmark_cmd} --save_outputs"
fi

echo "Running: ${benchmark_cmd}"
echo ""

# Run benchmark
eval "${benchmark_cmd}"
benchmark_exit_code=$?

# ============================================================
# COLLECT METRICS FROM SERVER
# ============================================================

if [[ "$SPEC_METHOD" != "baseline" ]]; then
    echo ""
    echo "============================================================"
    echo "SPECULATIVE DECODING METRICS FROM PROMETHEUS"
    echo "============================================================"
    
    echo ""
    echo "Draft tokens proposed:"
    curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "spec_decode_num_draft_tokens_total{" || echo "  (not available)"
    
    echo ""
    echo "Tokens accepted:"
    curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "spec_decode_num_accepted_tokens_total{" || echo "  (not available)"
    
    echo ""
    echo "Per-position acceptance rates:"
    curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "spec_decode_num_accepted_tokens_per_pos{" | head -10 || echo "  (not available)"
    
    echo ""
    echo "Number of drafts:"
    curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "spec_decode_num_drafts_total{" || echo "  (not available)"
    
    # For self_spec_ngram, also show self-spec metrics
    if [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
        echo ""
        echo "Self-spec draft tokens:"
        curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "self_spec_num_draft_tokens_total{" || echo "  (not available)"
        
        echo ""
        echo "Self-spec accepted tokens:"
        curl -s http://${HOST}:${PORT}/metrics 2>/dev/null | grep "self_spec_num_accepted_tokens_total{" || echo "  (not available)"
    fi
    
    echo ""
    echo "============================================================"
fi

# ============================================================
# CLEANUP
# ============================================================

echo ""
echo "Shutting down server..."
kill ${pid} 2>/dev/null || true
wait ${pid} 2>/dev/null || true

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETED!"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Exit code: ${benchmark_exit_code}"

exit ${benchmark_exit_code}
