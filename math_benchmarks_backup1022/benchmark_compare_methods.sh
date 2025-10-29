#!/bin/bash
# Compare different speculative decoding methods on math benchmarks

set -e

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
DATASETS="gsm8k"
NUM_SAMPLES=50
OUTPUT_BASE="./comparison_results"

echo "============================================================"
echo "COMPARING SPECULATIVE DECODING METHODS"
echo "============================================================"
echo "Model: ${MODEL}"
echo "Datasets: ${DATASETS}"
echo "Samples per method: ${NUM_SAMPLES}"
echo "============================================================"

# Array of methods to compare
methods=("baseline" "ngram" "self_specs" "self_spec_ngram")

for method in "${methods[@]}"; do
    echo ""
    echo "============================================================"
    echo "Testing method: ${method}"
    echo "============================================================"
    
    # Update the benchmark script configuration
    sed -i "s/^SPEC_METHOD=.*/SPEC_METHOD=\"${method}\"/" benchmark_math_server.sh
    sed -i "s|^OUTPUT_DIR=.*|OUTPUT_DIR=\"${OUTPUT_BASE}/${method}\"|" benchmark_math_server.sh
    sed -i "s/^MODEL=.*/MODEL=\"${MODEL}\"/" benchmark_math_server.sh
    sed -i "s/^DATASETS=.*/DATASETS=\"${DATASETS}\"/" benchmark_math_server.sh
    sed -i "s/^NUM_SAMPLES=.*/NUM_SAMPLES=${NUM_SAMPLES}/" benchmark_math_server.sh
    
    # Run benchmark
    ./benchmark_math_server.sh
    
    echo ""
    echo "Completed: ${method}"
    echo "============================================================"
    
    # Wait a bit before starting next benchmark
    sleep 5
done

echo ""
echo "============================================================"
echo "ALL COMPARISONS COMPLETED!"
echo "============================================================"
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Summary:"
for method in "${methods[@]}"; do
    metrics_file="${OUTPUT_BASE}/${method}/${DATASETS}/test_tool-integrated_${NUM_SAMPLES}_seed0_t0.0_s0_e-1_tool-integrated_metrics.json"
    if [ -f "${metrics_file}" ]; then
        acc=$(grep -oP '"acc":\s*\K[0-9.]+' "${metrics_file}" | head -1)
        duration=$(grep -oP '"duration_seconds":\s*\K[0-9.]+' "${metrics_file}" | head -1)
        throughput=$(grep -oP '"throughput_req_per_sec":\s*\K[0-9.]+' "${metrics_file}" | head -1)
        echo "  ${method}: acc=${acc}%, duration=${duration}s, throughput=${throughput} req/s"
    else
        echo "  ${method}: metrics not found"
    fi
done
