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
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="outputs/sspec_$(date +%Y%m%d_%H%M%S)"

# Test datasets (quick ones first)
DATASETS="${DATASETS:-aime24}"

# Self-spec parameters (aligned with examples/offline_inference/self_spec.py)
SSPEC_NUM_TOKENS="${SSPEC_NUM_TOKENS:-8}"
SSPEC_RECENT_SIZE="${SSPEC_RECENT_SIZE:-2048}"
SSPEC_SINK_SIZE="${SSPEC_SINK_SIZE:-64}"
SSPEC_BLOCK_SIZE="${SSPEC_BLOCK_SIZE:-1}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Self-Spec Parameters:"
echo "  Num speculative tokens: $SSPEC_NUM_TOKENS"
echo "  Recent size: $SSPEC_RECENT_SIZE"
echo "  Sink size: $SSPEC_SINK_SIZE"
echo "  Block size: $SSPEC_BLOCK_SIZE"
echo ""

# Ensure we're in the correct directory
cd /home/ccchang/repositories/vllm/math_benchmarks

# Run self-spec test
echo "Running self-spec test..."
echo ""

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
    --vllm_sspec_recent_size "$SSPEC_RECENT_SIZE" \
    --vllm_sspec_sink_size "$SSPEC_SINK_SIZE" \
    --vllm_sspec_block_size "$SSPEC_BLOCK_SIZE" \
    --save_outputs \
    --overwrite \
    --repeat_dataset 30

echo ""
echo "================================================"
echo "Self-spec test results saved to: $OUTPUT_DIR"
echo "================================================"
echo ""
echo "Results summary:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    metrics_file=$(find "$OUTPUT_DIR/$dataset" -name "*_metrics.json" 2>/dev/null | head -1)
    if [ -n "$metrics_file" ] && [ -f "$metrics_file" ]; then
        acc=$(jq -r '.acc' "$metrics_file" 2>/dev/null || echo "N/A")
        echo "  $dataset: ${acc}%"
    fi
done
echo ""

# Show self-spec metrics if available
echo "Self-spec metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    metrics_file=$(find "$OUTPUT_DIR/$dataset" -name "*_metrics.json" 2>/dev/null | head -1)
    if [ -n "$metrics_file" ] && [ -f "$metrics_file" ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$metrics_file" 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline:"
echo "  export SSPEC_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with \$BASELINE_DIR and \$SSPEC_DIR"
echo ""
