#!/bin/bash
# run_sspec_ngram_test.sh
# Test vLLM self-speculative decoding with n-gram assistance on math benchmarks

set -e

echo "================================================"
echo "Testing vLLM Self-Spec with N-gram Assistance"
echo "================================================"

# Configuration (using same defaults as run_baseline_snapshot.sh)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="outputs/sspec_ngram_$(date +%Y%m%d_%H%M%S)"

# Test datasets (quick ones first)
DATASETS="${DATASETS:-aime24}"

# Self-spec with n-gram parameters
SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS="${SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS:-8}"  # Threshold for ACCUMULATING -> VERIFYING
SSPEC_NGRAM_NUM_DRAFT_TOKENS="${SSPEC_NGRAM_NUM_DRAFT_TOKENS:-1}"              # N-gram draft tokens per step
SSPEC_NGRAM_PROMPT_LOOKUP_MIN="${SSPEC_NGRAM_PROMPT_LOOKUP_MIN:-5}"              # Default: same as num_draft_tokens
SSPEC_NGRAM_PROMPT_LOOKUP_MAX="${SSPEC_NGRAM_PROMPT_LOOKUP_MAX:-5}"              # Default: same as num_draft_tokens
SSPEC_NGRAM_SINK_SIZE="${SSPEC_NGRAM_SINK_SIZE:-32}"                             # Streaming cache sink blocks
SSPEC_NGRAM_RECENT_RATIO="${SSPEC_NGRAM_RECENT_RATIO:-0.05}"                    # Streaming cache recent ratio
SSPEC_NGRAM_BLOCK_SIZE="${SSPEC_NGRAM_BLOCK_SIZE:-1}"                           # Block size (1 disables prefix caching)

# Profiling options
ENABLE_NSYS_PROFILING="${ENABLE_NSYS_PROFILING:-0}"
NSYS_PROFILE_OUTPUT="${NSYS_PROFILE_OUTPUT:-self_spec_ngram_qwen3_8b_tp1}"
NSYS_PROFILE_FORCE="${NSYS_PROFILE_FORCE:-true}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "  Nsight Systems profiling: enabled (output: $NSYS_PROFILE_OUTPUT)"
else
    echo "  Nsight Systems profiling: disabled"
fi
echo ""
echo "Self-Spec N-gram Parameters:"
echo "  Speculative tokens threshold (ACCUMULATING->VERIFYING): $SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS"
echo "  N-gram draft tokens per step: $SSPEC_NGRAM_NUM_DRAFT_TOKENS"
if [ -n "$SSPEC_NGRAM_PROMPT_LOOKUP_MIN" ]; then
    echo "  N-gram window min: $SSPEC_NGRAM_PROMPT_LOOKUP_MIN"
else
    echo "  N-gram window min: $SSPEC_NGRAM_NUM_DRAFT_TOKENS (default)"
fi
if [ -n "$SSPEC_NGRAM_PROMPT_LOOKUP_MAX" ]; then
    echo "  N-gram window max: $SSPEC_NGRAM_PROMPT_LOOKUP_MAX"
else
    echo "  N-gram window max: $SSPEC_NGRAM_NUM_DRAFT_TOKENS (default)"
fi
echo "  Sink size: $SSPEC_NGRAM_SINK_SIZE blocks"
echo "  Recent ratio: $SSPEC_NGRAM_RECENT_RATIO ($(echo "$SSPEC_NGRAM_RECENT_RATIO * 100" | bc)% of computed tokens)"
echo "  Block size: $SSPEC_NGRAM_BLOCK_SIZE"
echo ""

# Ensure we're in the correct directory
cd /home/ubuntu/vllm/math_benchmarks_backup1022

# Build command with optional arguments
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
    --vllm_enable_sspec_ngram
    --vllm_sspec_ngram_num_speculative_tokens "$SSPEC_NGRAM_NUM_SPECULATIVE_TOKENS"
    --vllm_sspec_ngram_num_ngram_draft_tokens "$SSPEC_NGRAM_NUM_DRAFT_TOKENS"
    --vllm_sspec_ngram_sink_size "$SSPEC_NGRAM_SINK_SIZE"
    --vllm_sspec_ngram_recent_ratio "$SSPEC_NGRAM_RECENT_RATIO"
    --vllm_sspec_ngram_block_size "$SSPEC_NGRAM_BLOCK_SIZE"
    --save_outputs
    --overwrite
    --repeat_dataset 10
)

# Add optional lookup window parameters if specified
if [ -n "$SSPEC_NGRAM_PROMPT_LOOKUP_MIN" ]; then
    CMD_ARGS+=(--vllm_sspec_ngram_prompt_lookup_min "$SSPEC_NGRAM_PROMPT_LOOKUP_MIN")
fi
if [ -n "$SSPEC_NGRAM_PROMPT_LOOKUP_MAX" ]; then
    CMD_ARGS+=(--vllm_sspec_ngram_prompt_lookup_max "$SSPEC_NGRAM_PROMPT_LOOKUP_MAX")
fi

# Compose run command
RUN_CMD=(python math_eval.py "${CMD_ARGS[@]}")

# Run self-spec n-gram test
echo "Running self-spec n-gram test..."
echo ""

if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "Profiling with Nsight Systems (delay=360s, duration=10s)"
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    TOKENIZERS_PARALLELISM=false \
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --delay=600 \
        --duration=10 \
        -o "$NSYS_PROFILE_OUTPUT" \
        -f "$NSYS_PROFILE_FORCE" \
        "${RUN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    TOKENIZERS_PARALLELISM=false \
    "${RUN_CMD[@]}"
fi

echo ""
echo "================================================"
echo "Self-spec n-gram test results saved to: $OUTPUT_DIR"
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

# Show self-spec n-gram metrics if available
echo "Self-spec n-gram metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline or regular self-spec:"
echo "  export SSPEC_NGRAM_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with $BASELINE_DIR, $SSPEC_DIR, and $SSPEC_NGRAM_DIR"
echo ""

