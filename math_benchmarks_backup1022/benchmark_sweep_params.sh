#!/bin/bash
# Sweep speculative token parameters for self_spec_ngram

set -e

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
DATASETS="gsm8k"
NUM_SAMPLES=50
OUTPUT_BASE="./param_sweep_results"
SPEC_METHOD="self_spec_ngram"

echo "============================================================"
echo "PARAMETER SWEEP FOR ${SPEC_METHOD}"
echo "============================================================"
echo "Model: ${MODEL}"
echo "Datasets: ${DATASETS}"
echo "Samples per configuration: ${NUM_SAMPLES}"
echo "============================================================"

# Sweep over spec_tokens values
spec_tokens_values=(4 8 16 32)
ngram_draft_values=(2 3 4 5)

for spec_tokens in "${spec_tokens_values[@]}"; do
    for ngram_draft in "${ngram_draft_values[@]}"; do
        config_name="spec${spec_tokens}_ngram${ngram_draft}"
        echo ""
        echo "============================================================"
        echo "Testing configuration: ${config_name}"
        echo "  Spec tokens: ${spec_tokens}"
        echo "  N-gram draft tokens: ${ngram_draft}"
        echo "============================================================"
        
        # Update the benchmark script configuration
        sed -i "s/^SPEC_METHOD=.*/SPEC_METHOD=\"${SPEC_METHOD}\"/" benchmark_math_server.sh
        sed -i "s/^SPEC_TOKENS=.*/SPEC_TOKENS=${spec_tokens}/" benchmark_math_server.sh
        sed -i "s/^NGRAM_DRAFT_TOKENS=.*/NGRAM_DRAFT_TOKENS=${ngram_draft}/" benchmark_math_server.sh
        sed -i "s|^OUTPUT_DIR=.*|OUTPUT_DIR=\"${OUTPUT_BASE}/${config_name}\"|" benchmark_math_server.sh
        sed -i "s/^MODEL=.*/MODEL=\"${MODEL}\"/" benchmark_math_server.sh
        sed -i "s/^DATASETS=.*/DATASETS=\"${DATASETS}\"/" benchmark_math_server.sh
        sed -i "s/^NUM_SAMPLES=.*/NUM_SAMPLES=${NUM_SAMPLES}/" benchmark_math_server.sh
        
        # Run benchmark
        ./benchmark_math_server.sh
        
        echo ""
        echo "Completed: ${config_name}"
        echo "============================================================"
        
        # Wait before next configuration
        sleep 5
    done
done

echo ""
echo "============================================================"
echo "PARAMETER SWEEP COMPLETED!"
echo "============================================================"
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Summary:"
printf "%-20s %-10s %-12s %-15s\n" "Configuration" "Accuracy" "Duration(s)" "Throughput"
echo "--------------------------------------------------------------------"
for spec_tokens in "${spec_tokens_values[@]}"; do
    for ngram_draft in "${ngram_draft_values[@]}"; do
        config_name="spec${spec_tokens}_ngram${ngram_draft}"
        metrics_file="${OUTPUT_BASE}/${config_name}/${DATASETS}/test_tool-integrated_${NUM_SAMPLES}_seed0_t0.0_s0_e-1_tool-integrated_metrics.json"
        if [ -f "${metrics_file}" ]; then
            acc=$(grep -oP '"acc":\s*\K[0-9.]+' "${metrics_file}" | head -1)
            duration=$(grep -oP '"duration_seconds":\s*\K[0-9.]+' "${metrics_file}" | head -1)
            throughput=$(grep -oP '"throughput_req_per_sec":\s*\K[0-9.]+' "${metrics_file}" | head -1)
            printf "%-20s %-10.2f %-12.2f %-15.2f\n" "${config_name}" "${acc}" "${duration}" "${throughput}"
        else
            printf "%-20s %-10s %-12s %-15s\n" "${config_name}" "N/A" "N/A" "N/A"
        fi
    done
done
