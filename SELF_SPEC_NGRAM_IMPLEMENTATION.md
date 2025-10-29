# Self-Spec N-gram Implementation Summary

## Overview

Successfully implemented `self_spec_ngram` method that combines self-speculative decoding with n-gram draft proposals to accelerate the ACCUMULATING phase.

## Changes Made

### 1. Configuration (`vllm/config/speculative.py`)

- Added `"self_spec_ngram"` to `SpeculativeMethod` Literal type
- Updated `use_self_specs()` to return `True` for both `"self_specs"` and `"self_spec_ngram"`
- Added handling in `__post_init__` to configure n-gram parameters for `self_spec_ngram`
- Updated `__repr__` to include `self_spec_ngram` in the no-model methods

**Key configuration logic:**
```python
# When method is "self_spec_ngram", it automatically configures ngram parameters
if self.method == "self_spec_ngram":
    self.model = "ngram"  # Set model to trigger ngram configuration

# Default prompt_lookup_min/max are set to 5 if not provided
# Can be customized via constructor parameters
```

### 2. Model Runner (`vllm/v1/worker/gpu_model_runner.py`)

#### Drafter Initialization (L296-298)
```python
elif self.speculative_config.method == "self_spec_ngram":
    # Self-spec with n-gram assistance during ACCUMULATING phase
    self.drafter = NgramProposer(self.vllm_config)
```

#### Draft Proposal (L2800-2809)
```python
elif self.speculative_config.method == "self_spec_ngram":
    # Self-spec with n-gram: propose drafts for all requests
    # Scheduler will override with pending_output_tokens when transitioning to VERIFYING
    assert isinstance(sampled_token_ids, list)
    assert isinstance(self.drafter, NgramProposer)
    draft_token_ids = self.drafter.propose(
        sampled_token_ids, self.input_batch.req_ids,
        self.input_batch.num_tokens_no_spec,
        self.input_batch.token_ids_cpu,
        self.input_batch.spec_decode_unsupported_reqs)
```

## How It Works

### State-Based Flow

```
ACCUMULATING Phase:
  1. NgramProposer generates draft tokens (e.g., [123, 456, 789])
  2. Engine updates request.spec_token_ids = [123, 456, 789]
  3. Model runner verifies with streaming cache
  4. Accepted tokens → pending_output_tokens
  5. Repeat until len(pending_output_tokens) >= threshold

Transition (L316 in scheduler.py):
  When len(pending_output_tokens) >= self_spec_threshold:
    - request.spec_token_ids = pending_output_tokens (OVERWRITES n-gram drafts!)
    - request.self_spec_state = VERIFYING

VERIFYING Phase:
  1. NgramProposer still generates drafts (gets overwritten anyway)
  2. Scheduler already set spec_token_ids to pending_output_tokens
  3. Model runner verifies with FULL KV cache
  4. Accepted tokens committed, rejected → back to ACCUMULATING
```

### Key Insight

**No need for state-aware filtering in the proposer!** The scheduler naturally handles this:
- During ACCUMULATING: N-gram drafts are used as-is
- During VERIFYING: Scheduler overwrites spec_token_ids at L316 with pending_output_tokens

## Usage

### Basic Usage
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "self_spec_ngram",
        "num_speculative_tokens": 8,  # Threshold for ACCUMULATING → VERIFYING
    }
)

prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=50))
```

### Custom N-gram Parameters
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "self_spec_ngram",
        "num_speculative_tokens": 8,
        "prompt_lookup_min": 2,  # Min n-gram window size
        "prompt_lookup_max": 4,  # Max n-gram window size
    }
)
```

## Testing

### Unit Test (Configuration)
```bash
source .venv/bin/activate
python test_self_spec_ngram_config.py
```

### Integration Test (Requires GPU)
```bash
source .venv/bin/activate

# Simple test with vllm command line
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-method self_spec_ngram \
    --num-speculative-tokens 8 \
    --prompt-lookup-max 3

# Or with Python API
python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='meta-llama/Llama-3.1-8B-Instruct',
    speculative_config={
        'method': 'self_spec_ngram',
        'num_speculative_tokens': 8,
    }
)
prompts = ['Write a poem about AI'] * 10
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=50))
for output in outputs[:2]:
    print(output.outputs[0].text)
"
```

## Expected Performance

With n-gram acceptance rate of ~50% and max_draft=3:

- **Without n-gram**: 8 ACCUMULATING steps (1 token per step)
- **With n-gram**: ~4 ACCUMULATING steps (avg 2 tokens per step)
- **Expected speedup**: 2x during accumulation phase

## Files Modified

1. `vllm/config/speculative.py`
   - Added `"self_spec_ngram"` to method types
   - Updated `use_self_specs()` method
   - Added configuration logic for n-gram parameters
   - Updated `__repr__` method

2. `vllm/v1/worker/gpu_model_runner.py`
   - Added drafter initialization for `self_spec_ngram`
   - Added draft proposal logic in `propose_draft_token_ids()`

3. `test_self_spec_ngram_config.py` (new)
   - Configuration validation tests

## Next Steps

### Optional Enhancements
1. **Adaptive draft length**: Adjust `max_draft_tokens` based on acceptance rate
2. **Performance benchmarking**: Compare speedup vs pure `self_specs`
3. **Integration tests**: End-to-end tests with actual model inference
4. **Documentation**: Update user-facing docs with new method

### Benchmarking Commands
```bash
# Benchmark self_specs (baseline)
python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-method self_specs \
    --num-speculative-tokens 8

# Benchmark self_spec_ngram (new)
python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-method self_spec_ngram \
    --num-speculative-tokens 8 \
    --prompt-lookup-max 3
```

## Design Principles

1. **Reuse existing components**: No new proposer class, just reuse `NgramProposer`
2. **Minimal code changes**: Only 3 files modified with ~30 lines of code
3. **Clean separation**: Model runner proposes, scheduler decides
4. **Backward compatible**: Doesn't affect existing `self_specs` or `ngram` methods

## Commit Message

```
feat: Add self_spec_ngram method for accelerated self-speculative decoding

Implement self_spec_ngram method that combines self-speculative decoding
with n-gram draft proposals to accelerate the ACCUMULATING phase.

Key changes:
- Add "self_spec_ngram" to SpeculativeMethod types
- Initialize NgramProposer for self_spec_ngram in gpu_model_runner
- Propose n-gram drafts during all states (scheduler handles override)
- Configure default n-gram parameters (prompt_lookup_min/max = 5)

During ACCUMULATING, n-gram drafts are verified with streaming cache,
allowing multiple tokens per step. During VERIFYING, scheduler
overwrites spec_token_ids with pending_output_tokens for full KV
verification.

Expected 2x speedup during accumulation phase with 50% n-gram acceptance.
```

## References

- Design discussion: [Previous conversation]
- Related PRs: Self-spec initial implementation
- Documentation: `SELF_SPEC_TOKEN_STATE_GUIDE.md`, `STREAMING_CACHE_INTEGRATION.md`
