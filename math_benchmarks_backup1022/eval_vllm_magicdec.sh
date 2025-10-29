#!/bin/bash
# run_sspec_test.sh
# Test vLLM self-speculative decoding on math benchmarks

set -e

echo "================================================"
echo "Testing vLLM Self-Speculative Decoding"
echo "================================================"



# Configuration
MODEL_PATH="${MODEL_PATH:-/nobackup/model/qwen3/Qwen3-8B}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen3-math-thinking}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
OUTPUT_DIR="outputs/sspec_$(date +%Y%m%d_%H%M%S)"

# Test datasets
DATASETS="${DATASETS:-aime24}"

# Self-spec parameters (streaming cache configuration)
SSPEC_NUM_TOKENS="${SSPEC_NUM_TOKENS:-8}"
SSPEC_SINK_SIZE="${SSPEC_SINK_SIZE:-32}"           # blocks (8 blocks = 128 tokens with block_size=16)
SSPEC_RECENT_RATIO="${SSPEC_RECENT_RATIO:-0.05}"  # ratio (5% of computed tokens)
SSPEC_BLOCK_SIZE="${SSPEC_BLOCK_SIZE:-1}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  Samples per dataset: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Self-Spec Parameters (Streaming Cache):"
echo "  Num speculative tokens: $SSPEC_NUM_TOKENS"
echo "  Sink size: $SSPEC_SINK_SIZE blocks"
echo "  Recent ratio: $SSPEC_RECENT_RATIO ($(echo "$SSPEC_RECENT_RATIO * 100" | bc)% of computed tokens)"
echo "  Block size: $SSPEC_BLOCK_SIZE"
echo ""

# Ensure we're in the correct directory
cd eval/benchmarks

# Run self-spec test
echo "Running self-spec test..."
echo ""

export CUDA_VISIBLE_DEVICES=0
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-1.7B \
    --data_names aime24 \
    --output_dir aime24/qwen3_1.7b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type qwen3-math-thinking \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 60 \

export CUDA_VISIBLE_DEVICES=0,1
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-8B \
    --data_names aime24 \
    --output_dir aime24/qwen3_8b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type qwen3-math-thinking \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 60 \
    --vllm_gpu_memory_utilization 0.85

export CUDA_VISIBLE_DEVICES=0,1,2,3
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-14B \
    --data_names aime24 \
    --output_dir aime24/qwen3_14b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type qwen3-math-thinking \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 60 \
    --vllm_gpu_memory_utilization 0.85

export CUDA_VISIBLE_DEVICES=0
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-1.7B \
    --data_names livecodebench \
    --output_dir livecodebench/qwen3_1.7b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 2 \

export CUDA_VISIBLE_DEVICES=0,1
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-8B \
    --data_names livecodebench \
    --output_dir livecodebench/qwen3_8b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 2 \
    --vllm_gpu_memory_utilization 0.85

export CUDA_VISIBLE_DEVICES=0,1,2,3
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-14B \
    --data_names livecodebench \
    --output_dir livecodebench/qwen3_14b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 2 \
    --vllm_gpu_memory_utilization 0.85

export CUDA_VISIBLE_DEVICES=0
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-1.7B \
    --data_names olympiadbench \
    --output_dir olympiadbench/qwen3_1.7b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 3 \

export CUDA_VISIBLE_DEVICES=0,1
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-8B \
    --data_names olympiadbench \
    --output_dir olympiadbench/qwen3_8b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 3 \
    --vllm_gpu_memory_utilization 0.85

export CUDA_VISIBLE_DEVICES=0,1,2,3
python math_eval_triforce.py \
    --model_name_or_path /nobackup/model/qwen3/Qwen3-14B \
    --data_names olympiadbench \
    --output_dir olympiadbench/qwen3_14b/vllm_magicdec_t06_new \
    --split test \
    --prompt_type cot \
    --seed 0 \
    --temperature 0.6 \
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
    --repeat_dataset 3 \
    --vllm_gpu_memory_utilization 0.85

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
