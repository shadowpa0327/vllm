#!/bin/bash
# run_vllm_ngram_test.sh
# Test vLLM with n-gram prompt lookup decoding on AIME24 benchmark

set -e

echo "================================================"
echo "Testing vLLM N-gram Prompt Lookup - AIME24"
echo "================================================"

# Configuration (using same defaults as sspec.sh)
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
OUTPUT_DIR="outputs/vllm_ngram_${MODEL_NAME}_tp${TP_SIZE}_repeat${REPEAT_DATASET}_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="aime24"

# N-gram prompt lookup parameters
NGRAM_NUM_DRAFT_TOKENS="${NGRAM_NUM_DRAFT_TOKENS:-5}"              # N-gram draft tokens per step
NGRAM_PROMPT_LOOKUP_MIN="${NGRAM_PROMPT_LOOKUP_MIN:-5}"            # Min n-gram window size
NGRAM_PROMPT_LOOKUP_MAX="${NGRAM_PROMPT_LOOKUP_MAX:-5}"            # Max n-gram window size

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
echo "N-gram Prompt Lookup Parameters:"
echo "  N-gram draft tokens per step: $NGRAM_NUM_DRAFT_TOKENS"
echo "  N-gram window min: $NGRAM_PROMPT_LOOKUP_MIN"
echo "  N-gram window max: $NGRAM_PROMPT_LOOKUP_MAX"
echo ""

# Ensure we're in the correct directory
cd /home/ubuntu/vllm/math_benchmarks_backup1022

# Run n-gram test
echo "Running vLLM n-gram test..."
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
        python math_eval_orignal.py \
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
        --vllm_enable_ngram \
        --vllm_num_speculative_tokens "$NGRAM_NUM_DRAFT_TOKENS" \
        --vllm_ngram_prompt_lookup_min "$NGRAM_PROMPT_LOOKUP_MIN" \
        --vllm_ngram_prompt_lookup_max "$NGRAM_PROMPT_LOOKUP_MAX" \
        --save_outputs \
        --overwrite \
        --repeat_dataset "$REPEAT_DATASET"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    TOKENIZERS_PARALLELISM=false \
    python math_eval_orignal.py \
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
        --vllm_enable_ngram \
        --vllm_num_speculative_tokens "$NGRAM_NUM_DRAFT_TOKENS" \
        --vllm_ngram_prompt_lookup_min "$NGRAM_PROMPT_LOOKUP_MIN" \
        --vllm_ngram_prompt_lookup_max "$NGRAM_PROMPT_LOOKUP_MAX" \
        --save_outputs \
        --overwrite \
        --repeat_dataset "$REPEAT_DATASET"
fi

echo ""
echo "================================================"
echo "vLLM n-gram test results saved to: $OUTPUT_DIR"
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

# Show n-gram metrics if available
echo "N-gram prompt lookup metrics:"
for dataset in $(echo $DATASETS | tr ',' ' '); do
    if [ -f "$OUTPUT_DIR/$dataset"/*_metrics.json ]; then
        echo "  $dataset:"
        jq -r '.speculative_decoding // "No spec metrics found"' "$OUTPUT_DIR/$dataset"/*_metrics.json 2>/dev/null || echo "    N/A"
    fi
done
echo ""

echo "To compare with baseline:"
echo "  export NGRAM_DIR='$OUTPUT_DIR'"
echo "  # Then run comparison script with \$BASELINE_DIR and \$NGRAM_DIR"
echo ""
