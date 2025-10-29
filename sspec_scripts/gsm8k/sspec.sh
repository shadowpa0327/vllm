#!/bin/bash
# run_sspec_test.sh
# Test vLLM self-speculative decoding on math benchmarks

set -e

echo "================================================"
echo "Testing vLLM Self-Speculative Decoding"
echo "================================================"

# Configuration (using same defaults as run_baseline_snapshot.sh)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="outputs/sspec_$(date +%Y%m%d_%H%M%S)"

# Test datasets (quick ones first)
DATASETS="gsm8k"

# Self-spec parameters (streaming cache configuration)
SSPEC_NUM_TOKENS="${SSPEC_NUM_TOKENS:-6}"
SSPEC_SINK_SIZE="${SSPEC_SINK_SIZE:-32}"           # blocks (8 blocks = 128 tokens with block_size=16)
SSPEC_RECENT_RATIO="${SSPEC_RECENT_RATIO:-0.05}"  # ratio (10% of computed tokens)
SSPEC_BLOCK_SIZE="${SSPEC_BLOCK_SIZE:-1}"

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
echo "Self-Spec Parameters (Streaming Cache):"
echo "  Num speculative tokens: $SSPEC_NUM_TOKENS"
echo "  Sink size: $SSPEC_SINK_SIZE blocks"
echo "  Recent ratio: $SSPEC_RECENT_RATIO ($(echo "$SSPEC_RECENT_RATIO * 100" | bc)% of computed tokens)"
echo "  Block size: $SSPEC_BLOCK_SIZE"
echo ""

# Ensure we're in the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_BENCH_ROOT="${MATH_BENCH_ROOT:-$(cd "$SCRIPT_DIR/../../math_benchmarks_backup1022" && pwd)}"
if [ ! -d "$MATH_BENCH_ROOT" ]; then
    echo "Error: math benchmarks directory not found at $MATH_BENCH_ROOT" >&2
    exit 1
fi
cd "$MATH_BENCH_ROOT"

# Run self-spec test
echo "Running self-spec test..."
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
        python math_eval.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_names "$DATASETS" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample "$NUM_SAMPLES" \
        --seed 0 \
        --temperature 0.65 \
        --use_vllm \
        --apply_chat_template \
        --enable_thinking \
        --vllm_enable_sspec \
        --vllm_sspec_num_speculative_tokens "$SSPEC_NUM_TOKENS" \
        --vllm_sspec_sink_size "$SSPEC_SINK_SIZE" \
        --vllm_sspec_recent_ratio "$SSPEC_RECENT_RATIO" \
        --vllm_sspec_block_size "$SSPEC_BLOCK_SIZE" \
        --save_outputs \
        --overwrite
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TOKENIZERS_PARALLELISM=false \
    python math_eval.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_names "$DATASETS" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample "$NUM_SAMPLES" \
        --seed 0 \
        --temperature 0.65 \
        --use_vllm \
        --apply_chat_template \
        --enable_thinking \
        --vllm_enable_sspec \
        --vllm_sspec_num_speculative_tokens "$SSPEC_NUM_TOKENS" \
        --vllm_sspec_sink_size "$SSPEC_SINK_SIZE" \
        --vllm_sspec_recent_ratio "$SSPEC_RECENT_RATIO" \
        --vllm_sspec_block_size "$SSPEC_BLOCK_SIZE" \
        --save_outputs \
        --overwrite
fi

echo ""
echo "================================================"
echo "Self-spec test results saved to: $OUTPUT_DIR"
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

# Show self-spec metrics if available
echo "Self-spec metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline:"
echo "  export SSPEC_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with \$BASELINE_DIR and \$SSPEC_DIR"
echo ""
