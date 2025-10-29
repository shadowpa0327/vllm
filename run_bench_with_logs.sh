#!/bin/bash
# Run vLLM bench serve with server logs visible
# Server stays running so you can see all logs

set -e

# Configuration (same as run_sspec_test.sh)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PROMPT_FILE="${PROMPT_FILE:-/home/ubuntu/vllm/prompt_for_testing.jsonl}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"
TEMPERATURE="${TEMPERATURE:-0.65}"

# GPU configuration
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# CUDA graph configuration
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
# Space-separated list for cuda-graph-sizes (will be split into array)
CUDA_GRAPH_SIZES="${CUDA_GRAPH_SIZES:-1 2 4 8 16 32 64 128 192 256 320 384 448 512 768 1024 1536}"

# Self-spec parameters
SPEC_TOKENS="${SPEC_TOKENS:-8}"
NGRAM_TOKENS="${NGRAM_TOKENS:-1}"

# Streaming cache parameters (now supported as CLI args!)
SINK_SIZE="${SINK_SIZE:-32}"
RECENT_RATIO="${RECENT_RATIO:-0.05}"

# Build speculative config
SPEC_CONFIG=$(cat <<EOF
{
    "method": "self_spec_ngram",
    "model": null,
    "num_speculative_tokens": ${SPEC_TOKENS},
    "num_ngram_draft_tokens": ${NGRAM_TOKENS},
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 5
}
EOF
)

echo "============================================================"
echo "vLLM Bench Serve with Server Logs"
echo "============================================================"
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Prompt file: $PROMPT_FILE"
echo "  Temperature: $TEMPERATURE"
echo "  Request rate: $REQUEST_RATE"
echo ""
echo "GPU Configuration:"
echo "  CUDA devices: $CUDA_DEVICES"
echo "  Tensor parallel size: $TENSOR_PARALLEL"
echo "  GPU memory utilization: $GPU_MEMORY_UTIL"
echo "  Max num seqs: $MAX_NUM_SEQS"
echo "  Max num batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "  Enforce eager: $ENFORCE_EAGER"
echo "  CUDA graph sizes: $CUDA_GRAPH_SIZES"
echo ""
echo "Self-Spec Config:"
echo "  Method: self_spec_ngram"
echo "  Spec tokens: $SPEC_TOKENS"
echo "  N-gram draft tokens: $NGRAM_TOKENS"
echo ""
echo "Streaming Cache Config:"
echo "  Sink size: $SINK_SIZE"
echo "  Recent ratio: $RECENT_RATIO"
echo "============================================================"
echo ""

# # Check if prompt file exists
# if [ ! -f "$PROMPT_FILE" ]; then
#     echo "ERROR: Prompt file not found: $PROMPT_FILE"
#     echo "Run ./dump_sspec_prompts.sh first to generate prompts"
#     exit 1
# fi

# NUM_PROMPTS=$(wc -l < "$PROMPT_FILE")
# echo "Found $NUM_PROMPTS prompts in $PROMPT_FILE"
# echo ""

# Set environment variables
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_ATTENTION_BACKEND="FLASHINFER"

# Optional: Set profiler directory (empty = no profiling)
VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-}"
if [ -n "$VLLM_TORCH_PROFILER_DIR" ]; then
    export VLLM_TORCH_PROFILER_DIR="$VLLM_TORCH_PROFILER_DIR"
    echo "Torch profiler enabled, output to: $VLLM_TORCH_PROFILER_DIR"
fi

echo "Starting vLLM server (logs will be shown below)..."
echo "Press Ctrl+C to stop the server when benchmark is done"
echo "============================================================"
echo ""

# Start server in foreground (no background &)
# Server logs will be visible

# Build vllm serve command as array to properly handle JSON
VLLM_ARGS=(
    "vllm" "serve" "$MODEL_PATH"
    "--trust-remote-code"
    "--tensor-parallel-size" "$TENSOR_PARALLEL"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTIL"
    "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
    "--max-num-seqs" "$MAX_NUM_SEQS"
    "--block-size" "1"
    "--enable-chunked-prefill"
    #"--speculative-config" "${SPEC_CONFIG}"
    "--sink-size" "$SINK_SIZE"
    "--recent-ratio" "$RECENT_RATIO"
    "--no-enable-prefix-caching"
)

# Note: We removed --disable-log-stats false because:
# - --disable-log-stats is a flag (no value)
# - We want logs enabled, so we don't add this flag at all

# Add optional parameters
if [ "$ENFORCE_EAGER" = "true" ]; then
    VLLM_ARGS+=("--enforce-eager")
fi

if [ -n "$CUDA_GRAPH_SIZES" ]; then
    # Add --cuda-graph-sizes with each size as a separate argument
    VLLM_ARGS+=("--cuda-graph-sizes")
    # Split CUDA_GRAPH_SIZES by space and add each as separate argument
    for size in $CUDA_GRAPH_SIZES; do
        VLLM_ARGS+=("$size")
    done
fi

if [ -n "$MAX_MODEL_LEN" ]; then
    VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
fi

# Debug: Print command if DEBUG is set
if [ "${DEBUG:-false}" = "true" ]; then
    echo "DEBUG: Executing command:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
    printf '%q ' "${VLLM_ARGS[@]}"
    echo ""
    echo ""
fi

echo "Executing command: ${VLLM_ARGS[@]}"

# Execute command
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${VLLM_ARGS[@]}"

# Note: Script will stay here showing server logs
# Run the benchmark in a SEPARATE terminal with:
#
# vllm bench serve \
#     --backend openai \
#     --model $MODEL_PATH \
#     --dataset-name custom \
#     --dataset-path $PROMPT_FILE \
#     --temperature $TEMPERATURE \
#     --request-rate $REQUEST_RATE \
#     --save-result
