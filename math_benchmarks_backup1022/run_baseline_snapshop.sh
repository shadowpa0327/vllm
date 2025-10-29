#!/bin/bash
# run_baseline_snapshot.sh
# Creates a baseline accuracy snapshot before making vLLM modifications

set -e

echo "================================================"
echo "Creating Baseline Accuracy Snapshot"
echo "================================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
OUTPUT_DIR="outputs/baseline_$(date +%Y%m%d_%H%M%S)"

# Test datasets (quick ones first)
DATASETS="gsm8k"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Ensure we're on a clean state
cd /home/ubuntu/vllm/math_benchmarks_backup

# Run baseline test
echo "Running baseline test..."
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
    --overwrite

echo ""
echo "================================================"
echo "Baseline snapshot saved to: $OUTPUT_DIR"
echo "================================================"
echo ""
echo "Results summary:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        acc=$(jq -r '.acc' "$OUTPUT_DIR/$dataset"/*_metrics.json)
        echo "  $dataset: ${acc}%"
    fi
done
echo ""
echo "To use this as baseline for comparison:"
echo "  export BASELINE_DIR='$OUTPUT_DIR'"
