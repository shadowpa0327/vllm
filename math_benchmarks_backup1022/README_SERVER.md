# Math Benchmark with vLLM Server

This directory contains scripts for evaluating math benchmarks by submitting requests to a vLLM server, similar to the pattern used in `vllm/benchmarks/serve.py`.

## Overview

The main components are:

1. **`math_eval_server.py`** - Core evaluation script that preprocesses math datasets and submits requests to a vLLM server
2. **`benchmark_math_server.sh`** - Automated benchmark script that starts server, runs evaluation, and collects metrics
3. **`benchmark_compare_methods.sh`** - Compare different speculative decoding methods
4. **`benchmark_sweep_params.sh`** - Sweep parameter values for optimization

## Quick Start

### Option 1: Using the Automated Benchmark Script (Recommended)

The easiest way is to use the automated benchmark script that handles server startup and shutdown:

```bash
# Edit configuration in benchmark_math_server.sh
vim benchmark_math_server.sh

# Run the benchmark
./benchmark_math_server.sh
```

### Option 2: Manual Server + Client

If you prefer manual control:

```bash
# Terminal 1: Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# Terminal 2: Run math evaluation
python math_eval_server.py \
    --data_names gsm8k,math \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --num_test_sample 100 \
    --max_tokens_per_call 2048 \
    --save_outputs
```

## Detailed Usage

### math_eval_server.py

Main evaluation script that processes math datasets and submits to vLLM server.

**Key Arguments:**

```bash
# Data configuration
--data_names gsm8k,math         # Datasets to evaluate
--data_dir ./data               # Data directory
--num_test_sample 100           # Number of samples (-1 for all)

# Server configuration
--host 127.0.0.1                # Server host
--port 8000                     # Server port
--model_name_or_path MODEL      # Model name

# Generation configuration
--max_tokens_per_call 2048      # Max tokens per request
--temperature 0.0               # Sampling temperature
--prompt_type tool-integrated   # Prompt type (tool-integrated, cot, pal)

# Request control
--request_rate inf              # Requests per second (inf or number)
--max_concurrency 128           # Max concurrent requests

# Output
--output_dir ./output           # Output directory
--save_outputs                  # Save detailed outputs
```

**Example:**

```bash
python math_eval_server.py \
    --data_names gsm8k \
    --model_name_or_path Qwen/Qwen2.5-Math-7B-Instruct \
    --num_test_sample 50 \
    --request_rate 10 \
    --max_concurrency 32 \
    --save_outputs
```

### benchmark_math_server.sh

Automated benchmark script that:
- Starts vLLM server with specified configuration
- Waits for server to be ready
- Runs math evaluation
- Collects Prometheus metrics
- Shuts down server

**Configuration (edit the script):**

```bash
# Model
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

# Benchmark settings
DATASETS="gsm8k,math"
NUM_SAMPLES=100
MAX_TOKENS=2048
PROMPT_TYPE="tool-integrated"

# Speculative decoding
SPEC_METHOD="baseline"          # baseline, ngram, self_specs, self_spec_ngram
SPEC_TOKENS=8                   # Speculative token count
NGRAM_DRAFT_TOKENS=3            # N-gram draft tokens

# Request control
REQUEST_RATE="inf"
MAX_CONCURRENCY=128
```

**Supported Speculative Methods:**

1. **baseline** - No speculative decoding
2. **ngram** - N-gram prompt lookup
3. **self_specs** - Self-speculative decoding (requires V1 engine)
4. **self_spec_ngram** - Self-spec with n-gram assistance (requires V1 engine)

**Example:**

```bash
# Test self-speculative with n-gram
sed -i 's/^SPEC_METHOD=.*/SPEC_METHOD="self_spec_ngram"/' benchmark_math_server.sh
sed -i 's/^NUM_SAMPLES=.*/NUM_SAMPLES=100/' benchmark_math_server.sh
./benchmark_math_server.sh
```

### benchmark_compare_methods.sh

Compare all speculative decoding methods side-by-side:

```bash
# Edit configuration at top of script
vim benchmark_compare_methods.sh

# Run comparison
./benchmark_compare_methods.sh
```

This will:
- Test baseline, ngram, self_specs, and self_spec_ngram
- Save results to separate directories
- Print comparison summary

**Output:**

```
Summary:
  baseline: acc=85.5%, duration=120s, throughput=0.83 req/s
  ngram: acc=85.5%, duration=95s, throughput=1.05 req/s
  self_specs: acc=85.5%, duration=88s, throughput=1.14 req/s
  self_spec_ngram: acc=85.5%, duration=75s, throughput=1.33 req/s
```

### benchmark_sweep_params.sh

Sweep over parameter values to find optimal configuration:

```bash
# Edit configuration
vim benchmark_sweep_params.sh

# Run sweep
./benchmark_sweep_params.sh
```

This tests combinations of:
- `SPEC_TOKENS` values (e.g., 4, 8, 16, 32)
- `NGRAM_DRAFT_TOKENS` values (e.g., 2, 3, 4, 5)

**Output:**

```
Configuration         Accuracy   Duration(s)  Throughput
--------------------------------------------------------------------
spec4_ngram2          85.50      92.45        0.54
spec4_ngram3          85.50      88.23        0.57
spec8_ngram3          85.50      75.12        0.67
...
```

## Output Files

After running benchmarks, you'll find:

```
output_dir/
├── gsm8k/
│   ├── test_tool-integrated_100_seed0_t0.0_s0_e-1.jsonl          # Detailed results
│   └── test_tool-integrated_100_seed0_t0.0_s0_e-1_tool-integrated_metrics.json  # Metrics
└── math/
    ├── test_tool-integrated_100_seed0_t0.0_s0_e-1.jsonl
    └── test_tool-integrated_100_seed0_t0.0_s0_e-1_tool-integrated_metrics.json
```

**Metrics JSON contains:**
- `acc` - Accuracy percentage
- `duration_seconds` - Total benchmark duration
- `throughput_req_per_sec` - Requests per second
- Plus detailed per-sample results

## Speculative Decoding Metrics

When using speculative methods, the server exposes Prometheus metrics:

```bash
# Check metrics while server is running
curl http://localhost:8000/metrics | grep spec_decode

# Key metrics:
# - spec_decode_num_draft_tokens_total - Total draft tokens
# - spec_decode_num_accepted_tokens_total - Accepted tokens
# - spec_decode_num_drafts_total - Number of verification steps
# - spec_decode_num_accepted_tokens_per_pos - Per-position acceptance
```

## Troubleshooting

### Server fails to start

```bash
# Check if port is in use
lsof -i :8000

# Kill existing server
pkill -f "vllm serve"
```

### Out of memory

Reduce batch size or model length:

```bash
# In benchmark_math_server.sh, modify server args:
--max-num-batched-tokens 4096   # Reduce from 8192
--gpu-memory-utilization 0.85   # Reduce from 0.9
```

### Import errors

Ensure you're in the correct directory:

```bash
cd /home/ubuntu/vllm/math_benchmarks_backup1022
python math_eval_server.py --help
```

### Benchmark takes too long

Reduce sample size for quick testing:

```bash
sed -i 's/^NUM_SAMPLES=.*/NUM_SAMPLES=10/' benchmark_math_server.sh
./benchmark_math_server.sh
```

## Advanced Examples

### High-throughput stress test

```bash
python math_eval_server.py \
    --data_names gsm8k \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --num_test_sample 500 \
    --request_rate inf \
    --max_concurrency 256 \
    --save_outputs
```

### Controlled rate with chat template

```bash
python math_eval_server.py \
    --data_names gsm8k \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --num_test_sample 100 \
    --request_rate 5 \
    --max_concurrency 16 \
    --apply_chat_template \
    --save_outputs
```

### Different prompt types

```bash
# Chain-of-thought
python math_eval_server.py --prompt_type cot ...

# Program-aided language (PAL)
python math_eval_server.py --prompt_type pal ...

# Tool-integrated (default)
python math_eval_server.py --prompt_type tool-integrated ...
```

## Comparison with Original math_eval.py

| Feature | math_eval.py | math_eval_server.py |
|---------|--------------|---------------------|
| Data preprocessing | ✅ | ✅ (extracted) |
| Local inference | ✅ | ❌ |
| Server-based inference | ❌ | ✅ |
| Request rate control | ❌ | ✅ |
| Concurrent requests | Limited | ✅ Full control |
| Async I/O | ❌ | ✅ |
| Live metrics | Limited | ✅ Prometheus |
| Automated benchmarking | Partial | ✅ Full automation |

## References

- Original math_eval.py: `/home/ubuntu/vllm/math_benchmarks_backup1022/math_eval.py`
- vLLM serve benchmark: `/home/ubuntu/vllm/vllm/benchmarks/serve.py`
- Self-spec benchmark example: `/home/ubuntu/vllm/benchmark_self_spec.sh`
