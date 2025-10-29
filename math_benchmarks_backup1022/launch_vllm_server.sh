#!/bin/bash
# Launch vLLM server with configurable speculative decoding
# Usage: ./launch_vllm_server.sh [method] [tensor_parallel_size]
#
# Arguments:
#   method: baseline|ngram|self_specs|self_spec_ngram (default: self_spec_ngram)
#   tensor_parallel_size: number of GPUs for tensor parallelism (default: 1)
#
# Examples:
#   ./launch_vllm_server.sh baseline                    # Single GPU, no spec decode
#   ./launch_vllm_server.sh self_spec_ngram             # Single GPU, self_spec_ngram
#   ./launch_vllm_server.sh self_spec_ngram 2           # 2 GPUs, self_spec_ngram
#   CUDA_VISIBLE_DEVICES=0,1 ./launch_vllm_server.sh self_specs 2
#   CUDA_VISIBLE_DEVICES=2 ./launch_vllm_server.sh baseline 1

set -e

# ============================================================
# CONFIGURATION
# ============================================================

# Parse command-line arguments
SPEC_METHOD="${1:-self_spec_ngram}"
TENSOR_PARALLEL_SIZE="${2:-${TENSOR_PARALLEL_SIZE:-1}}"

# Model configuration
MODEL="Qwen/Qwen3-8B"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Server configuration
HOST="127.0.0.1"
PORT=8000
GPU_MEMORY_UTIL=0.9

# Speculative decoding parameters
SPEC_TOKENS=8           # Number of speculative tokens
NGRAM_DRAFT_TOKENS=1    # N-gram draft tokens (for self_spec_ngram)
NGRAM_MIN=5             # Min n-gram window
NGRAM_MAX=5             # Max n-gram window

# Set environment variables for V1 engine (required for self_specs methods)
if [[ "$SPEC_METHOD" == "self_specs" ]] || [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
    export VLLM_ENABLE_V1_MULTIPROCESSING=1
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    echo "✓ V1 engine enabled for ${SPEC_METHOD}"
fi

# ============================================================
# BUILD SPECULATIVE CONFIG
# ============================================================

spec_config=""
spec_args=""

case "$SPEC_METHOD" in
    baseline)
        echo "Starting baseline vLLM server (no speculative decoding)"
        ;;
    
    ngram)
        spec_config=$(cat <<EOF
{
    "method": "ngram",
    "num_speculative_tokens": ${SPEC_TOKENS},
    "ngram_prompt_lookup_min": ${NGRAM_MIN},
    "ngram_prompt_lookup_max": ${NGRAM_MAX}
}
EOF
)
        spec_args="--speculative-config"
        echo "Starting vLLM server with N-gram speculative decoding"
        echo "  Spec tokens: ${SPEC_TOKENS}"
        echo "  N-gram window: [${NGRAM_MIN}, ${NGRAM_MAX}]"
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
        echo "Starting vLLM server with self-speculative decoding"
        echo "  Spec tokens: ${SPEC_TOKENS}"
        ;;
    
    self_spec_ngram)
        spec_config=$(cat <<EOF
{
    "method": "self_spec_ngram",
    "num_speculative_tokens": ${SPEC_TOKENS},
    "num_ngram_draft_tokens": ${NGRAM_DRAFT_TOKENS},
    "prompt_lookup_min": ${NGRAM_MIN},
    "prompt_lookup_max": ${NGRAM_MAX}
}
EOF
)
        spec_args="--speculative-config"
        echo "Starting vLLM server with self_spec + n-gram"
        echo "  Spec tokens: ${SPEC_TOKENS}"
        echo "  N-gram draft tokens: ${NGRAM_DRAFT_TOKENS}"
        echo "  N-gram window: [${NGRAM_MIN}, ${NGRAM_MAX}]"
        ;;
    
    *)
        echo "Error: Unknown spec method '${SPEC_METHOD}'"
        echo "Usage: $0 [baseline|ngram|self_specs|self_spec_ngram]"
        exit 1
        ;;
esac

# ============================================================
# BUILD SERVER COMMAND
# ============================================================

echo ""
echo "============================================================"
echo "LAUNCHING VLLM SERVER"
echo "============================================================"
echo "Model: ${MODEL}"
echo "Host: ${HOST}:${PORT}"
echo "Method: ${SPEC_METHOD}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "CUDA Visible Devices: ${CUDA_VISIBLE_DEVICES}"
else
    echo "CUDA Visible Devices: all available"
fi
echo "============================================================"
echo ""

# Base command
server_cmd="vllm serve ${MODEL}"
server_cmd="${server_cmd} --host ${HOST}"
server_cmd="${server_cmd} --port ${PORT}"
server_cmd="${server_cmd} --trust-remote-code"

# Add tensor parallel size
if [[ ${TENSOR_PARALLEL_SIZE} -gt 1 ]]; then
    server_cmd="${server_cmd} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
fi

# Add speculative config if not baseline
if [[ -n "$spec_args" ]]; then
    server_cmd="${server_cmd} ${spec_args} '${spec_config}'"
fi

# Engine-specific arguments
if [[ "$SPEC_METHOD" == "self_specs" ]] || [[ "$SPEC_METHOD" == "self_spec_ngram" ]]; then
    # V1 engine configuration
    server_cmd="${server_cmd} --max-model-len 40960"
    server_cmd="${server_cmd} --max-num-batched-tokens 2048"
    server_cmd="${server_cmd} --block-size 1"
    server_cmd="${server_cmd} --gpu-memory-utilization ${GPU_MEMORY_UTIL}"
    server_cmd="${server_cmd} --no-enable-prefix-caching"
    #server_cmd="${server_cmd} --disable-log-requests"
else
    # V0 engine configuration
    server_cmd="${server_cmd} --gpu-memory-utilization ${GPU_MEMORY_UTIL}"
    #server_cmd="${server_cmd} --disable-log-requests"
fi

# ============================================================
# LAUNCH SERVER
# ============================================================

echo "Executing command:"
echo "${server_cmd}"
echo ""
echo "Server will run in background..."
echo "PID will be saved to: vllm_server.pid"
echo ""
echo "To stop the server, run:"
echo "  kill \$(cat vllm_server.pid)"
echo ""
echo "To check server logs, run:"
echo "  tail -f vllm_server.log"
echo ""
echo "============================================================"
echo ""

# Launch server in background and save PID
eval "${server_cmd}" > vllm_server.log 2>&1 &
pid=$!
echo $pid > vllm_server.pid

echo "✓ Server launched with PID: ${pid}"
echo ""

# Wait for server to start
echo "Waiting for server to be ready..."
max_wait=300  # 5 minutes
waited=0
while ! curl -s http://${HOST}:${PORT}/v1/models >/dev/null 2>&1; do
    if [ $waited -ge $max_wait ]; then
        echo ""
        echo "✗ Error: Server failed to start within ${max_wait} seconds"
        echo "  Check logs: tail -f vllm_server.log"
        echo "  Kill process: kill ${pid}"
        exit 1
    fi
    
    # Check if process is still running
    if ! kill -0 ${pid} 2>/dev/null; then
        echo ""
        echo "✗ Error: Server process died"
        echo "  Check logs: tail -f vllm_server.log"
        exit 1
    fi
    
    printf "."
    sleep 5
    waited=$((waited + 5))
done

echo ""
echo ""
echo "============================================================"
echo "✓ SERVER READY!"
echo "============================================================"
echo "Endpoint: http://${HOST}:${PORT}"
echo "Health: http://${HOST}:${PORT}/health"
echo "Models: http://${HOST}:${PORT}/v1/models"
echo "Metrics: http://${HOST}:${PORT}/metrics"
echo ""
echo "Server is running with PID: ${pid}"
echo "Logs: tail -f vllm_server.log"
echo ""
echo "To stop: kill \$(cat vllm_server.pid)"
echo "============================================================"
