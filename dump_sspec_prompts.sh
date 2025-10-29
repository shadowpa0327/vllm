#!/bin/bash
# Dump prompts for vLLM bench serve using same config as run_sspec_test.sh

set -e

# Configuration (same as run_sspec_test.sh)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
DATASETS="${DATASETS:-aime24}"
OUTPUT_FILE="${OUTPUT_FILE:-bench_data/sspec_prompts.jsonl}"

echo "Dumping prompts for vLLM bench serve..."
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples: $NUM_SAMPLES"
echo "  Output: $OUTPUT_FILE"
echo ""

source .venv/bin/activate && python dump_math_eval_prompts.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_names "$DATASETS" \
    --data_dir ./math_benchmarks_backup1022/data \
    --output_file "$OUTPUT_FILE" \
    --split test \
    --prompt_type "$PROMPT_TYPE" \
    --num_test_sample "$NUM_SAMPLES" \
    --seed 0 \
    --apply_chat_template \
    --enable_thinking \
    --n_sampling 1 \
    --repeat_dataset 30

echo ""
echo "Prompts saved to: $OUTPUT_FILE"
echo ""
echo "To run vLLM bench serve:"
echo ""
echo "# 1. Start vLLM server:"
echo "vllm serve $MODEL_PATH \\"
echo "    --trust-remote-code \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --port 8000"
echo ""
echo "# 2. Run benchmark:"
echo "vllm bench serve \\"
echo "    --backend openai \\"
echo "    --model $MODEL_PATH \\"
echo "    --dataset-name custom \\"
echo "    --dataset-path $OUTPUT_FILE \\"
echo "    --temperature 0.65 \\"
echo "    --request-rate <rate> \\"
echo "    --save-result"
