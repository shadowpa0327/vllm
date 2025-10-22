# Self-Spec N-gram Usage Guide

## Overview

The `offline_inference_self_spec.py` script now supports two self-speculative decoding methods:

1. **`self_specs`** (baseline): Original self-speculative decoding
2. **`self_spec_ngram`** (new): Self-spec with n-gram draft proposals for faster accumulation

## Key Parameters

### Self-Spec Threshold
```bash
--num_speculative_tokens N
```
- **Purpose**: Number of tokens to accumulate before transitioning ACCUMULATING → VERIFYING
- **Default**: 8
- **Used by**: Both `self_specs` and `self_spec_ngram`
- **Example**: `--num_speculative_tokens 8` means accumulate 8 tokens, then verify all 8 with full KV

### N-gram Draft Tokens (NEW)
```bash
--ngram_draft_tokens N
```
- **Purpose**: Number of draft tokens proposed by n-gram per ACCUMULATING step
- **Default**: 3
- **Used by**: Only `self_spec_ngram`
- **Example**: `--ngram_draft_tokens 3` means n-gram proposes 3 tokens each step

### Advanced N-gram Window Size (Optional)
```bash
--prompt_lookup_max N   # Max n-gram window size (default: uses ngram_draft_tokens)
--prompt_lookup_min N   # Min n-gram window size (default: uses ngram_draft_tokens)
```

## Usage Examples

### Example 1: Baseline Self-Spec
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_specs \
    --num_speculative_tokens 8
```

**Behavior:**
- Accumulates 1 token per step (no n-gram)
- Takes ~8 steps to reach threshold
- Then verifies all 8 tokens with full KV

### Example 2: Self-Spec with N-gram (Default)
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 8 \
    --ngram_draft_tokens 3
```

**Behavior:**
- N-gram proposes 3 draft tokens per step
- With 50% acceptance: ~2 tokens accepted per step
- Takes ~4 steps to reach threshold (2x faster!)
- Then verifies all 8 tokens with full KV

### Example 3: Aggressive N-gram Drafting
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 16 \
    --ngram_draft_tokens 5
```

**Behavior:**
- Higher threshold (16 tokens)
- More aggressive drafting (5 tokens per step)
- Potentially reaches threshold in ~3-5 steps

### Example 4: Custom N-gram Window
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 8 \
    --prompt_lookup_min 2 \
    --prompt_lookup_max 4
```

**Behavior:**
- Variable n-gram window: 2-4 tokens
- Allows more flexible matching

## Parameter Relationship

```
┌─────────────────────────────────────────────────────────────┐
│ ACCUMULATING Phase                                           │
├─────────────────────────────────────────────────────────────┤
│ Step 1: N-gram proposes <ngram_draft_tokens> drafts         │
│         → Accept some (e.g., 2/3)                            │
│         → pending_output_tokens += accepted                  │
│                                                              │
│ Step 2: N-gram proposes <ngram_draft_tokens> drafts         │
│         → Accept some (e.g., 3/3)                            │
│         → pending_output_tokens += accepted                  │
│                                                              │
│ Step N: len(pending_output_tokens) >= <num_speculative_tokens>│
│         → TRANSITION TO VERIFYING                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ VERIFYING Phase                                              │
├─────────────────────────────────────────────────────────────┤
│ Verify all <num_speculative_tokens> with full KV            │
│ → Accept X tokens, reject rest                              │
│ → Back to ACCUMULATING with remaining tokens                │
└─────────────────────────────────────────────────────────────┘
```
