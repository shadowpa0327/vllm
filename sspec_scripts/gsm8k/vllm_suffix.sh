#!/bin/bash
# run_vllm_suffix_test.sh
# Test vLLM with suffix decoding on GSM8K benchmark

set -e

echo "================================================"
echo "Testing vLLM Suffix Decoding - GSM8K"
echo "================================================"

# Configuration (using same defaults as other gsm8k scripts)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
OUTPUT_DIR="outputs/vllm_suffix_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="gsm8k"

# Suffix decoding parameters
SUFFIX_NUM_SPECULATIVE_TOKENS="${SUFFIX_NUM_SPECULATIVE_TOKENS:-5}"        # Number of speculative tokens
SUFFIX_MAX_TREE_DEPTH="${SUFFIX_MAX_TREE_DEPTH:-24}"                       # Max suffix tree depth
SUFFIX_MAX_CACHED_REQUESTS="${SUFFIX_MAX_CACHED_REQUESTS:-10000}"          # Max cached requests (0=disable global cache)
SUFFIX_MAX_SPEC_FACTOR="${SUFFIX_MAX_SPEC_FACTOR:-1.0}"                    # Speculation length factor
SUFFIX_MIN_TOKEN_PROB="${SUFFIX_MIN_TOKEN_PROB:-0.1}"                      # Min token probability threshold

# Profiling options
ENABLE_NSYS_PROFILING="${ENABLE_NSYS_PROFILING:-0}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "  Nsight Systems profiling: enabled"
else
    echo "  Nsight Systems profiling: disabled"
fi
echo ""
echo "Suffix Decoding Parameters:"
echo "  Num speculative tokens: $SUFFIX_NUM_SPECULATIVE_TOKENS"
echo "  Max tree depth: $SUFFIX_MAX_TREE_DEPTH"
echo "  Max cached requests: $SUFFIX_MAX_CACHED_REQUESTS"
echo "  Max spec factor: $SUFFIX_MAX_SPEC_FACTOR"
echo "  Min token probability: $SUFFIX_MIN_TOKEN_PROB"
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
    --temperature 0.65
    --use_vllm
    --apply_chat_template
    --enable_thinking
    --vllm_enable_suffix
    --vllm_suffix_num_speculative_tokens "$SUFFIX_NUM_SPECULATIVE_TOKENS"
    --vllm_suffix_max_tree_depth "$SUFFIX_MAX_TREE_DEPTH"
    --vllm_suffix_max_cached_requests "$SUFFIX_MAX_CACHED_REQUESTS"
    --vllm_suffix_max_spec_factor "$SUFFIX_MAX_SPEC_FACTOR"
    --vllm_suffix_min_token_prob "$SUFFIX_MIN_TOKEN_PROB"
    --save_outputs
    --overwrite
)

# Compose run command
RUN_CMD=(python math_eval.py "${CMD_ARGS[@]}")

# Run suffix decoding test
echo "Running suffix decoding test..."
echo ""

if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "Profiling with Nsight Systems (delay=360s, duration=10s)"
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TOKENIZERS_PARALLELISM=false \
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --delay=360 \
        --duration=10 \
        "${RUN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TOKENIZERS_PARALLELISM=false \
    "${RUN_CMD[@]}"
fi

echo ""
echo "================================================"
echo "Suffix decoding test results saved to: $OUTPUT_DIR"
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

# Show suffix decoding metrics if available
echo "Suffix decoding metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline or other methods:"
echo "  export SUFFIX_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with \$BASELINE_DIR and \$SUFFIX_DIR"
echo ""
