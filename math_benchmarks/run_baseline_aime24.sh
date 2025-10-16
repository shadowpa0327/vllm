#!/bin/bash
# run_baseline_aime24.sh
# Test vLLM baseline (without self-speculative decoding) on AIME24

set -e

echo "================================================"
echo "Testing vLLM Baseline on AIME24"
echo "================================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="outputs/baseline_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="${DATASETS:-aime24}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Ensure we're in the correct directory
cd /home/ccchang/repositories/vllm/math_benchmarks

# Run baseline test
echo "Running baseline test (no self-spec)..."
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
    --save_outputs \
    --overwrite \
    --repeat_dataset 30

echo ""
echo "================================================"
echo "Baseline test results saved to: $OUTPUT_DIR"
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

echo "To compare with self-spec results:"
echo "  export BASELINE_DIR='$OUTPUT_DIR'"
echo "  # Run run_sspec_test.sh to get SSPEC_DIR"
echo "  # Then compare the two directories"
echo ""