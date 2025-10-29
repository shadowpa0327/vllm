#!/bin/bash
# run_sspec_test.sh
# Test vLLM self-speculative decoding on AIME24 benchmark

set -e

echo "================================================"
echo "Testing vLLM Self-Speculative Decoding - AIME24"
echo "================================================"

# Configuration (using same defaults as run_baseline_snapshot.sh)
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
    TP_SIZE=4  # default
fi

# Construct output directory with model name, TP, and repeat info
OUTPUT_DIR="outputs/sspec_${MODEL_NAME}_tp${TP_SIZE}_repeat${REPEAT_DATASET}_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="aime24"

# Self-spec parameters (streaming cache configuration)
SSPEC_NUM_TOKENS="${SSPEC_NUM_TOKENS:-8}"
SSPEC_SINK_SIZE="${SSPEC_SINK_SIZE:-32}"           # blocks (32 blocks with block_size=1)
SSPEC_RECENT_RATIO="${SSPEC_RECENT_RATIO:-0.05}"  # ratio (5% of computed tokens)
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
cd /home/ubuntu/vllm/math_benchmarks_backup1022

# Run self-spec test
echo "Running self-spec test..."
echo ""

if [[ "$ENABLE_NSYS_PROFILING" == "1" || "$ENABLE_NSYS_PROFILING" == "true" ]]; then
    echo "Profiling with Nsight Systems (delay=360s, duration=10s)"
    VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    TOKENIZERS_PARALLELISM=false \
    nsys profile \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --delay=360 \
        --duration=10 \
        -o "sspec_${MODEL_NAME}_tp${TP_SIZE}" \
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
        --overwrite \
        --repeat_dataset "$REPEAT_DATASET"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
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
        --overwrite \
        --repeat_dataset "$REPEAT_DATASET"
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
