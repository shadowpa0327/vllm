#!/bin/bash
# run_sspec_suffix_test.sh
# Test vLLM self-speculative decoding with suffix decode assistance on math benchmarks

set -e

echo "================================================"
echo "Testing vLLM Self-Spec with Suffix Decoding"
echo "================================================"

# Configuration (using same defaults as other AIME24 scripts)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
REPEAT_DATASET="${REPEAT_DATASET:-10}"

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")

# Determine TP size from CUDA_VISIBLE_DEVICES
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    TP_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    TP_SIZE=1  # default
fi

# Construct output directory with model name, TP, and repeat info
OUTPUT_DIR="outputs/sspec_suffix_${MODEL_NAME}_tp${TP_SIZE}_repeat${REPEAT_DATASET}_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="${DATASETS:-aime24}"

# Self-spec with suffix parameters
SSPEC_SUFFIX_NUM_SPECULATIVE_TOKENS="${SSPEC_SUFFIX_NUM_SPECULATIVE_TOKENS:-6}"   # Threshold for ACCUMULATING -> VERIFYING
SSPEC_SUFFIX_NUM_SUFFIX_DRAFT_TOKENS="${SSPEC_SUFFIX_NUM_SUFFIX_DRAFT_TOKENS:-3}" # Drafts per ACCUMULATING step
SSPEC_SUFFIX_MAX_TREE_DEPTH="${SSPEC_SUFFIX_MAX_TREE_DEPTH:-24}"                  # Max suffix tree depth
SSPEC_SUFFIX_MAX_CACHED_REQUESTS="${SSPEC_SUFFIX_MAX_CACHED_REQUESTS:-10000}"     # Max cached requests
SSPEC_SUFFIX_MAX_SPEC_FACTOR="${SSPEC_SUFFIX_MAX_SPEC_FACTOR:-1.0}"               # Speculation length factor
SSPEC_SUFFIX_MIN_TOKEN_PROB="${SSPEC_SUFFIX_MIN_TOKEN_PROB:-0.1}"                 # Min token probability
SSPEC_SUFFIX_SINK_SIZE="${SSPEC_SUFFIX_SINK_SIZE:-32}"                            # Streaming cache sink blocks (larger for 14B)
SSPEC_SUFFIX_RECENT_RATIO="${SSPEC_SUFFIX_RECENT_RATIO:-0.05}"                    # Streaming cache recent ratio (lower for AIME24)
SSPEC_SUFFIX_BLOCK_SIZE="${SSPEC_SUFFIX_BLOCK_SIZE:-1}"                           # Block size (1 disables prefix caching)

# Profiling options
ENABLE_NSYS_PROFILING="${ENABLE_NSYS_PROFILING:-0}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Repeat dataset: $REPEAT_DATASET times"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  Output: $OUTPUT_DIR"
if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "  Nsight Systems profiling: enabled"
else
    echo "  Nsight Systems profiling: disabled"
fi
echo ""
echo "Self-Spec Suffix Parameters:"
echo "  Speculative tokens threshold (ACCUMULATING->VERIFYING): $SSPEC_SUFFIX_NUM_SPECULATIVE_TOKENS"
echo "  Suffix draft tokens per step: $SSPEC_SUFFIX_NUM_SUFFIX_DRAFT_TOKENS"
echo "  Max tree depth: $SSPEC_SUFFIX_MAX_TREE_DEPTH"
echo "  Max cached requests: $SSPEC_SUFFIX_MAX_CACHED_REQUESTS"
echo "  Max spec factor: $SSPEC_SUFFIX_MAX_SPEC_FACTOR"
echo "  Min token probability: $SSPEC_SUFFIX_MIN_TOKEN_PROB"
echo "  Sink size: $SSPEC_SUFFIX_SINK_SIZE blocks"
echo "  Recent ratio: $SSPEC_SUFFIX_RECENT_RATIO ($(echo "$SSPEC_SUFFIX_RECENT_RATIO * 100" | bc)% of computed tokens)"
echo "  Block size: $SSPEC_SUFFIX_BLOCK_SIZE"
echo ""

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_BENCH_ROOT="${MATH_BENCH_ROOT:-$(cd "$SCRIPT_DIR/../../math_benchmarks_backup1022" && pwd)}"
if [ ! -d "$MATH_BENCH_ROOT" ]; then
    echo "Error: math benchmarks directory not found at $MATH_BENCH_ROOT" >&2
    exit 1
fi
cd "$MATH_BENCH_ROOT"

# Build command with arguments
CMD_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --data_names "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --split test
    --prompt_type "$PROMPT_TYPE"
    --num_test_sample "$NUM_SAMPLES"
    --seed 0
    --start 0
    --end -1
    --temperature 0.65
    --use_vllm
    --apply_chat_template
    --enable_thinking
    --vllm_enable_sspec_suffix
    --vllm_sspec_suffix_num_speculative_tokens "$SSPEC_SUFFIX_NUM_SPECULATIVE_TOKENS"
    --vllm_sspec_suffix_num_suffix_draft_tokens "$SSPEC_SUFFIX_NUM_SUFFIX_DRAFT_TOKENS"
    --vllm_sspec_suffix_max_tree_depth "$SSPEC_SUFFIX_MAX_TREE_DEPTH"
    --vllm_sspec_suffix_max_cached_requests "$SSPEC_SUFFIX_MAX_CACHED_REQUESTS"
    --vllm_sspec_suffix_max_spec_factor "$SSPEC_SUFFIX_MAX_SPEC_FACTOR"
    --vllm_sspec_suffix_min_token_prob "$SSPEC_SUFFIX_MIN_TOKEN_PROB"
    --vllm_sspec_suffix_sink_size "$SSPEC_SUFFIX_SINK_SIZE"
    --vllm_sspec_suffix_recent_ratio "$SSPEC_SUFFIX_RECENT_RATIO"
    --vllm_sspec_suffix_block_size "$SSPEC_SUFFIX_BLOCK_SIZE"
    --save_outputs
    --overwrite
    --repeat_dataset "$REPEAT_DATASET"
)

# Compose run command
RUN_CMD=(python math_eval.py "${CMD_ARGS[@]}")

# Run self-spec suffix test
echo "Running self-spec suffix test..."
echo ""

if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "Profiling with Nsight Systems (delay=600s, duration=10s)"
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    TOKENIZERS_PARALLELISM=false \
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --delay=600 \
        --duration=10 \
        -o "sspec_suffix_${MODEL_NAME}_tp${TP_SIZE}" \
        "${RUN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    TOKENIZERS_PARALLELISM=false \
    "${RUN_CMD[@]}"
fi

echo ""
echo "================================================"
echo "Self-spec suffix test results saved to: $OUTPUT_DIR"
echo "================================================"
echo ""
echo "Results summary:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        acc=$(jq -r '.acc' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "N/A")
        echo "  $dataset: ${acc}%"
    fi
done
echo ""

# Show self-spec suffix metrics if available
echo "Self-spec suffix metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline or other methods:"
echo "  export SSPEC_SUFFIX_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with \$BASELINE_DIR and \$SSPEC_SUFFIX_DIR"
echo ""
