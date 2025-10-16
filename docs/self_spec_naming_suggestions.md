# Self-Spec Implementation: Comprehensive Naming Suggestions

**Author:** Claude Code Analysis
**Date:** 2025-10-15
**Related:** self_spec_code_quality_proposal.md

---

## Overview

This document provides comprehensive naming suggestions for the self-speculative decoding implementation. Good naming improves code readability, reduces cognitive load, and makes the codebase more maintainable.

---

## Naming Categories

### 1. Variable Names

#### A. Boolean Flags: Verbose and Unclear Intent

**Issue:** Flag variable name is too verbose and doesn't clearly indicate it's a boolean.

##### Current:
```python
# scheduler.py:897, 917, 933, 997
flip_from_normal_to_accumulating = False
# ... later ...
if self.use_self_specs:
    flip_from_normal_to_accumulating = True
# ... later ...
if not stopped and request.self_spec_state == SelfSpecState.NORMAL and flip_from_normal_to_accumulating:
    # Update sparse attention indices
```

**Problems:**
- "flip_from_X_to_Y" is verbose (32 characters)
- Doesn't follow boolean naming conventions (should_*, is_*, has_*, can_*)
- "flip" suggests an action, but this is a state check
- Easy to confuse with similar variables

##### Suggested:
```python
should_transition_to_accumulating = False  # 31 chars but clearer
# OR
will_start_accumulating = False  # 23 chars, even clearer
# OR
entering_accumulation = False  # 21 chars, most concise
```

**Rationale:**
- Uses "should_" prefix indicating boolean decision
- Shorter and more readable
- Clear intent: checking if state transition is needed

---

#### B. Token Count Variables: Inconsistent Naming

**Issue:** Multiple variables for token counts with unclear relationships.

##### Current Naming:
```python
# In Request class
self.num_computed_tokens          # Tokens that have been processed
self._pending_output_tokens       # Tokens waiting for verification
self.spec_token_ids               # Speculative tokens being verified
self._output_token_ids            # Final committed output tokens
self._all_token_ids               # Prompt + committed tokens

# In scheduler
num_new_tokens                    # New tokens to schedule
num_draft_tokens                  # Draft tokens for speculation
num_scheduled_tokens              # Scheduled tokens per request (dict)
num_scheduled_pending_output_tokens  # Pending tokens that were scheduled
```

**Problems:**
- Inconsistent prefixes: `num_*` vs `_*_tokens` vs `*_token_ids`
- Unclear distinction between "tokens" and "token_ids"
- Hard to understand the state flow
- `num_scheduled_pending_output_tokens` is extremely verbose (36 chars!)

##### Suggested Refactoring:

**1. Use consistent prefixes:**
- `num_*` for counts (integers)
- `*_token_ids` for token ID lists
- `*_tokens` avoided (ambiguous - count or list?)

**2. Clear state hierarchy:**

```python
# Request class - Token State
class Request:
    # === Computed/Verified Tokens ===
    num_verified_tokens: int = 0  # Renamed from num_computed_tokens
    # Tokens that have been computed and verified (committed to KV cache)

    verified_token_ids: list[int]  # Renamed from _output_token_ids
    # List of verified output token IDs

    # === Speculative Tokens (being drafted) ===
    draft_token_ids: list[int]  # Renamed from _pending_output_tokens
    # Draft tokens waiting for verification (in ACCUMULATING state)

    # === Verification Tokens ===
    verifying_token_ids: list[int]  # Renamed from spec_token_ids
    # Tokens currently being verified (in VERIFYING state)

    # === Full Token History ===
    all_token_ids: list[int]  # Keep as is
    # Prompt + verified output tokens
```

**3. Scheduler variables:**

```python
# In schedule() method
tokens_to_schedule: int  # Renamed from num_new_tokens
# Number of new tokens to schedule for this request

draft_tokens_count: int  # Renamed from num_draft_tokens
# Number of draft tokens in speculation

scheduled_tokens_per_request: dict[str, int]  # Renamed from num_scheduled_tokens
# Map of request_id -> number of tokens scheduled

scheduled_draft_count: int  # Renamed from num_scheduled_pending_output_tokens
# Number of draft tokens that were already scheduled
```

**Visual Comparison:**

| Current | Suggested | Chars Saved | Clarity |
|---------|-----------|-------------|---------|
| `num_computed_tokens` | `num_verified_tokens` | 0 | ✓ More accurate |
| `_pending_output_tokens` | `draft_token_ids` | 7 | ✓✓ Clearer purpose |
| `spec_token_ids` | `verifying_token_ids` | -9 | ✓✓✓ Self-documenting |
| `num_scheduled_pending_output_tokens` | `scheduled_draft_count` | 14 | ✓✓✓ Much shorter |
| `flip_from_normal_to_accumulating` | `entering_accumulation` | 11 | ✓✓ Clearer |

---

#### C. Dictionary Names: Non-Standard Naming

**Issue:** Dictionary names don't follow common Python conventions.

##### Current:
```python
# scheduler.py:167-169
self.req_to_sparse_selected_kv_indices: dict[str, list[int]] = {}
self.req_to_full_kv_start_offset: dict[str, int] = {}
```

**Problems:**
- `req_to_*` pattern is uncommon in Python (more Java-style)
- Extremely verbose (38 chars for first one!)
- Doesn't match other dictionaries in codebase (`self.requests: dict[str, Request]`)

##### Suggested:
```python
# Option 1: Follow existing pattern (preferred)
self.sparse_kv_indices: dict[str, list[int]] = {}  # request_id -> indices
self.full_kv_offsets: dict[str, int] = {}          # request_id -> offset

# Option 2: If disambiguation needed
self.sparse_kv_indices_by_request: dict[str, list[int]] = {}
self.full_kv_offsets_by_request: dict[str, int] = {}

# Option 3: More descriptive
self.streaming_kv_indices: dict[str, list[int]] = {}  # Sparse attention indices
self.full_kv_start_positions: dict[str, int] = {}     # Where full KV begins
```

**Recommended:** Option 1 with inline comments
- Follows existing pattern (`self.requests`)
- Shorter and cleaner
- Comment clarifies key type if needed

---

### 2. Method Names

#### A. Verification Methods: Redundant Prefixes

**Issue:** Method names have redundant "self_spec" prefix when they're already in a self-spec context.

##### Current:
```python
# request.py
def start_self_spec_verification(self) -> list[int]:
    """Start verification and return tokens to verify"""
    assert self.self_spec_state == SelfSpecState.ACCUMULATING
    self.self_spec_state = SelfSpecState.VERIFYING
    ...

def start_suffix_verification(self) -> list[int]:
    """Start verification and return tokens to verify"""
    # Identical implementation!
    ...

# scheduler.py
def should_start_self_spec_verification(self, request: Request) -> bool:
    """Check if a request should start verification based on scheduler's threshold"""
    assert self.use_self_specs
    return (request.self_spec_state == SelfSpecState.ACCUMULATING and
            len(request._pending_output_tokens) >= self.self_spec_threshold)
```

**Problems:**
- `start_self_spec_verification` is redundant - we know it's self-spec from `self.self_spec_state`
- Two methods with identical implementations
- Scheduler method name is too verbose (37 chars)

##### Suggested Refactoring:

```python
# request.py - Single unified method
def start_verification(self) -> list[int]:
    """Transition from ACCUMULATING to VERIFYING state.

    Moves draft tokens to verification buffer and returns them.
    Used by both self-spec and suffix decoding modes.

    Returns:
        List of token IDs to verify

    Raises:
        AssertionError: If not in ACCUMULATING state
    """
    assert self.self_spec_state == SelfSpecState.ACCUMULATING
    self.self_spec_state = SelfSpecState.VERIFYING
    self.verifying_token_ids = self.draft_token_ids.copy()  # Updated names
    self.draft_token_ids.clear()
    return self.verifying_token_ids

# scheduler.py - Shorter, clearer name
def should_verify_drafts(self, request: Request) -> bool:
    """Check if request has enough draft tokens to start verification.

    Args:
        request: Request to check

    Returns:
        True if request should transition to VERIFYING state
    """
    if not self.use_self_specs:
        return False
    return (request.self_spec_state == SelfSpecState.ACCUMULATING and
            len(request.draft_token_ids) >= self.self_spec_threshold)
```

**Benefits:**
- Single implementation instead of duplicates
- Shorter names (easier to read in call sites)
- Clear intent from context

---

#### B. State Check Methods: Missing Property Pattern

**Issue:** State check is method instead of property.

##### Current:
```python
# request.py:131-133
@property
def is_in_self_spec_mode(self) -> bool:
    """Check if request is in any self-spec state"""
    return self.self_spec_state != SelfSpecState.NORMAL
```

**Problems:**
- Name has redundant "self_spec" - the property name already tells us the context
- Not consistent with other boolean properties in the class

##### Suggested:
```python
@property
def is_drafting(self) -> bool:
    """Check if request is drafting or verifying tokens.

    Returns True if in ACCUMULATING or VERIFYING state.
    Returns False if in NORMAL state.
    """
    return self.self_spec_state != SelfSpecState.NORMAL

# OR more explicit:
@property
def is_speculating(self) -> bool:
    """Check if request is in speculative execution mode."""
    return self.self_spec_state != SelfSpecState.NORMAL
```

**Additional useful properties:**

```python
@property
def is_accumulating_drafts(self) -> bool:
    """Check if currently accumulating draft tokens."""
    return self.self_spec_state == SelfSpecState.ACCUMULATING

@property
def is_verifying_drafts(self) -> bool:
    """Check if currently verifying draft tokens."""
    return self.self_spec_state == SelfSpecState.VERIFYING

@property
def num_draft_tokens(self) -> int:
    """Number of draft tokens waiting for verification."""
    return len(self.draft_token_ids)
```

---

#### C. Private Helper Methods: Inconsistent Prefix

**Issue:** Some private methods use underscore prefix inconsistently.

##### Current:
```python
# scheduler.py - Mix of naming styles
def _free_request(self, request: Request) -> Optional[dict[str, Any]]:  # ✓ Private
def _free_blocks(self, request: Request):  # ✓ Private
def _make_cached_request_data(...)  # ✓ Private

# But then (proposed new method):
def _prepare_verification(self, request: Request) -> tuple[...]:  # NEW: ✓ Private
```

**This is actually GOOD** - just ensure new methods follow the pattern:
- Public API methods: no prefix
- Internal helpers: `_` prefix

##### Suggested naming for new extraction:

```python
# scheduler.py
def _prepare_verification(self, request: Request) -> tuple[list[int], int, int]:
    """Prepare request for token verification.

    Common logic for self-spec and suffix verification:
    1. Adjust num_verified_tokens to exclude pending drafts
    2. Transition request to VERIFYING state
    3. Reset to full KV cache for verification
    4. Set spec_token_ids for verification

    Args:
        request: Request entering verification

    Returns:
        tuple of (tokens_to_verify, tokens_to_schedule, draft_count)
    """
    # Implementation...
```

**Rationale:**
- Clear that it's internal helper (`_` prefix)
- Verb "prepare" indicates setup/transformation
- Concise yet descriptive

---

### 3. Class and Enum Names

#### A. Enum State Names: Good, Could Be Better

##### Current:
```python
# request.py:18-22
class SelfSpecState(enum.Enum):
    """State for self-speculative decoding"""
    NORMAL = "normal"            # Regular token generation
    ACCUMULATING = "accumulating"  # Collecting tokens for verification
    VERIFYING = "verifying"        # Verifying accumulated tokens
```

**Analysis:**
- ✓ Good: Descriptive state names
- ✓ Good: Clear documentation
- ⚠️ Consider: "SelfSpecState" might be too specific

##### Suggested Alternative (if suffix/other modes share):
```python
class DraftingState(enum.Enum):
    """State for draft-then-verify decoding modes (self-spec, suffix, etc.)"""
    NORMAL = "normal"            # Regular token generation
    DRAFTING = "drafting"        # Collecting draft tokens (was ACCUMULATING)
    VERIFYING = "verifying"      # Verifying draft tokens
```

**OR keep as-is if truly self-spec specific:**
```python
class SpeculationState(enum.Enum):  # Slightly more general than SelfSpecState
    """State machine for speculative token generation."""
    NORMAL = "normal"
    ACCUMULATING = "accumulating"
    VERIFYING = "verifying"
```

**Recommendation:** Keep `SelfSpecState` for now, but consider renaming to `SpeculationState` or `DraftingState` if suffix decoding is implemented to avoid confusion.

---

#### B. Dataclass Field Names: Good Structure

##### Current:
```python
# gpu_input_batch.py:45-53
@dataclass
class CachedRequestState:
    # Selective KV indices for this request
    selective_kv_indices: Optional[list[int]] = None

    # Number of selective KV indices for this request
    num_selective_kv_indices: int = 0

    # The offset point where full KV caching starts for all subsequent tokens
    full_kv_start_offset: int = 0  # 0 means all tokens are used (full KV cache)
```

**Analysis:**
- ✓ Good: Descriptive field names
- ✓ Good: Inline comments
- ⚠️ Slight redundancy: `num_selective_kv_indices` is len(selective_kv_indices)

##### Suggested:
```python
@dataclass
class CachedRequestState:
    """Cached state for a request in the input batch."""

    # === Sparse Attention (Streaming LLM) ===
    streaming_kv_indices: Optional[list[int]] = None
    """Sparse KV indices for streaming attention (sink + recent tokens).
    None means full KV cache (no sparsity)."""

    full_kv_start_pos: int = 0
    """Token position where full KV caching begins.
    0 means full KV cache from start (no sparse attention).
    N means tokens 0..N-1 use sparse indices, N+ use full cache."""

    @property
    def num_streaming_kv_indices(self) -> int:
        """Number of sparse KV indices."""
        return len(self.streaming_kv_indices) if self.streaming_kv_indices else 0
```

**Benefits:**
- "streaming" is more descriptive than "selective"
- Property eliminates redundant field
- Better documentation with docstrings

---

### 4. Configuration Parameter Names

#### A. Config Field Names: Inconsistent Terminology

##### Current:
```python
# config.py (CacheConfig)
recent_size: int = 128
"""NOTE (siqi) These are parameters for streaming llm in self speculative decoding"""

sink_size: int = 32
"""NOTE (siqi) These are parameters for streaming llm in self speculative decoding"""

# scheduler.py initialization
self.recent_size = self.cache_config.recent_size
self.sink_size = self.cache_config.sink_size

# speculative_config usage
speculative_config.num_speculative_tokens
```

**Problems:**
- Not grouped together conceptually
- Comments are notes rather than proper docs
- Mix of `*_size` and `num_*` naming

##### Suggested Refactoring:

**Option 1: Create nested config (preferred for organization)**
```python
# config.py
@dataclass
class StreamingAttentionConfig:
    """Configuration for streaming attention (sparse KV cache).

    Implements the pattern from StreamingLLM paper:
    Keep sink tokens (attention anchors) + recent tokens (recency bias).
    """

    num_sink_tokens: int = 32
    """Number of initial tokens to keep (attention sink).
    These tokens are kept in KV cache to maintain attention stability."""

    num_recent_tokens: int = 128
    """Number of recent tokens to keep (recency bias).
    Recent tokens are important for next-token prediction."""

    @property
    def total_tokens_kept(self) -> int:
        """Total KV cache entries per request."""
        return self.num_sink_tokens + self.num_recent_tokens

    def validate(self, max_model_len: int) -> None:
        """Validate configuration against model constraints."""
        if self.total_tokens_kept > max_model_len:
            raise ValueError(
                f"Streaming attention requires {self.total_tokens_kept} tokens "
                f"but max_model_len is only {max_model_len}"
            )

@dataclass
class CacheConfig:
    # ... existing fields ...

    streaming_attention: Optional[StreamingAttentionConfig] = None
    """Configuration for sparse attention (streaming LLM pattern).
    If None, uses full dense attention."""
```

**Option 2: Keep flat but rename for consistency**
```python
# config.py
@dataclass
class CacheConfig:
    # === Streaming Attention (Sparse KV Cache) ===
    num_sink_tokens: int = 32
    """Number of initial tokens to keep as attention sinks.
    Used in self-speculative decoding with streaming attention."""

    num_recent_tokens: int = 128
    """Number of recent tokens to keep in KV cache.
    Used in self-speculative decoding with streaming attention."""
```

**Rationale for `num_*` prefix:**
- Consistent with other config fields (`num_gpu_blocks`, `num_speculative_tokens`)
- Clearly indicates integer count
- More discoverable with autocomplete

---

### 5. Local Variable Names in Complex Logic

#### A. Verification Logic: Overly Technical Names

##### Current:
```python
# scheduler.py:247-249
num_scheduled_pending_output_tokens = request.num_tokens - request.num_computed_tokens
request.num_computed_tokens += (num_scheduled_pending_output_tokens - len(request._pending_output_tokens))
request.num_computed_tokens -= 1 # Fall back one token
```

**Problems:**
- `num_scheduled_pending_output_tokens` is 36 characters!
- Complex calculation split across 3 lines with unclear purpose
- "scheduled pending output" is confusing terminology

##### Suggested with better names:

```python
# Calculate how many draft tokens were already scheduled
already_scheduled_drafts = request.num_tokens - request.num_verified_tokens
unscheduled_drafts = len(request.draft_token_ids)

# Adjust verified count: account for drafts, then step back one token
request.num_verified_tokens += already_scheduled_drafts - unscheduled_drafts
request.num_verified_tokens -= 1  # Recompute last token for verification
```

**OR with extracted helper method:**

```python
# Add to Request class
def prepare_for_verification(self) -> int:
    """Adjust token counts when transitioning to verification.

    When starting verification:
    1. Some draft tokens may have been scheduled already
    2. We need to recompute the last verified token
    3. Return the adjustment needed for num_verified_tokens

    Returns:
        Number of tokens to subtract from num_verified_tokens
    """
    already_scheduled = self.num_tokens - self.num_verified_tokens
    unscheduled = len(self.draft_token_ids)
    return unscheduled - already_scheduled + 1

# In scheduler:
tokens_to_recompute = request.prepare_for_verification()
request.num_verified_tokens -= tokens_to_recompute
```

---

#### B. Sparse Attention Logic: Non-Descriptive Names

##### Current:
```python
# scheduler.py:935-942
all_kv_indices = self.kv_cache_manager.get_block_ids(request.request_id)[0][:request.num_computed_tokens]
if self.sink_size + self.recent_size >= len(all_kv_indices):
    selective_kv_indices = all_kv_indices
else:
    selective_kv_indices = all_kv_indices[:self.sink_size] + all_kv_indices[-self.recent_size:]
self.req_to_sparse_selected_kv_indices[req_id] = selective_kv_indices
self.req_to_full_kv_start_offset[req_id] = request.num_computed_tokens
```

**Problems:**
- `all_kv_indices` is vague
- `selective_kv_indices` doesn't explain the pattern
- Logic is inline, hard to understand streaming pattern

##### Suggested:

```python
# Extract to helper method
def _compute_streaming_kv_indices(
    self,
    request: Request,
    req_id: str
) -> list[int]:
    """Compute sparse KV indices using streaming LLM pattern.

    Keeps sink tokens (first N) + recent tokens (last M) to reduce memory
    while maintaining attention quality.

    Args:
        request: Request to compute indices for
        req_id: Request identifier

    Returns:
        List of KV indices to keep (may be full or sparse)
    """
    # Get all verified KV indices for this request
    verified_kv_indices = (
        self.kv_cache_manager
        .get_block_ids(req_id)[0]  # First cache group
        [:request.num_verified_tokens]  # Only verified tokens
    )

    # If sequence is short, keep all tokens (no benefit to sparsity)
    total_kept = self.num_sink_tokens + self.num_recent_tokens
    if total_kept >= len(verified_kv_indices):
        return verified_kv_indices

    # Apply streaming pattern: sink + recent
    sink_indices = verified_kv_indices[:self.num_sink_tokens]
    recent_indices = verified_kv_indices[-self.num_recent_tokens:]
    return sink_indices + recent_indices

# In the calling code:
streaming_indices = self._compute_streaming_kv_indices(request, req_id)
self.sparse_kv_indices[req_id] = streaming_indices
self.full_kv_offsets[req_id] = request.num_verified_tokens
```

---

## Summary Table: Key Renamings

| Category | Current Name | Suggested Name | Impact |
|----------|-------------|----------------|---------|
| **Variable** | `flip_from_normal_to_accumulating` | `entering_accumulation` | High - used 4 times |
| **Variable** | `num_computed_tokens` | `num_verified_tokens` | High - core concept |
| **Variable** | `_pending_output_tokens` | `draft_token_ids` | High - clearer purpose |
| **Variable** | `spec_token_ids` | `verifying_token_ids` | Medium - clarifies state |
| **Variable** | `num_scheduled_pending_output_tokens` | `scheduled_draft_count` | Low - local var |
| **Dict** | `req_to_sparse_selected_kv_indices` | `sparse_kv_indices` | Medium - used 5+ times |
| **Dict** | `req_to_full_kv_start_offset` | `full_kv_offsets` | Medium - used 5+ times |
| **Method** | `start_self_spec_verification()` | `start_verification()` | High - remove duplicate |
| **Method** | `should_start_self_spec_verification()` | `should_verify_drafts()` | Medium - clearer |
| **Method** | `is_in_self_spec_mode` | `is_drafting` or `is_speculating` | Low - property |
| **Config** | `recent_size` | `num_recent_tokens` | Medium - consistency |
| **Config** | `sink_size` | `num_sink_tokens` | Medium - consistency |
| **Field** | `selective_kv_indices` | `streaming_kv_indices` | Low - more descriptive |
| **Field** | `full_kv_start_offset` | `full_kv_start_pos` | Low - shorter |

---

## Implementation Priority

### Phase 1: High Impact Renames (Do First)
1. ✓ `flip_from_normal_to_accumulating` → `entering_accumulation`
2. ✓ `num_computed_tokens` → `num_verified_tokens`
3. ✓ `_pending_output_tokens` → `draft_token_ids`
4. ✓ Merge `start_self_spec_verification()` + `start_suffix_verification()` → `start_verification()`
5. ✓ `req_to_*` dictionaries → remove prefix

**Estimated time:** 2-3 hours with tests

### Phase 2: Medium Impact Renames (Do Second)
1. `spec_token_ids` → `verifying_token_ids`
2. `should_start_self_spec_verification()` → `should_verify_drafts()`
3. Config: `recent_size` → `num_recent_tokens`, `sink_size` → `num_sink_tokens`
4. Extract `_compute_streaming_kv_indices()` helper method

**Estimated time:** 3-4 hours with tests

### Phase 3: Low Impact Renames (Optional)
1. Dataclass fields: `selective_kv_indices` → `streaming_kv_indices`
2. Properties: `is_in_self_spec_mode` → `is_drafting`
3. Local variable improvements in complex logic
4. Create `StreamingAttentionConfig` nested config

**Estimated time:** 2-3 hours

---

## Naming Conventions Reference

For future additions, follow these conventions:

### Boolean Variables
- Prefix with: `is_`, `has_`, `should_`, `can_`, `will_`
- Examples: `is_drafting`, `should_verify`, `has_pending_tokens`

### Counts
- Prefix with: `num_`
- Examples: `num_verified_tokens`, `num_draft_tokens`

### Lists of IDs
- Suffix with: `_ids`
- Examples: `draft_token_ids`, `verified_token_ids`

### Dictionaries (Mappings)
- Use descriptive noun (plural if mapping to collections)
- Add comment indicating key → value types
- Examples: `requests: dict[str, Request]  # request_id → Request`

### Private Methods
- Prefix with: `_`
- Use verbs: `_prepare_`, `_compute_`, `_validate_`

### Configuration Fields
- Use `num_` for counts
- Use descriptive nouns for other types
- Group related fields with comments

---

## Migration Strategy

### Gradual Refactoring
1. **Introduce aliases** (if possible) to maintain backward compatibility
2. **Update internal code** to use new names
3. **Deprecate old names** (add warnings)
4. **Remove old names** after 1-2 releases

### Example:
```python
# Phase 1: Add alias
@property
def draft_token_ids(self) -> list[int]:
    """Draft tokens awaiting verification (new name for _pending_output_tokens)."""
    return self._pending_output_tokens

# Phase 2: Update all internal usages to draft_token_ids

# Phase 3: Make _pending_output_tokens an alias
@property
def _pending_output_tokens(self) -> list[int]:
    """Deprecated: Use draft_token_ids instead."""
    warnings.warn("_pending_output_tokens is deprecated, use draft_token_ids",
                  DeprecationWarning, stacklevel=2)
    return self.draft_token_ids

# Phase 4: Remove _pending_output_tokens
```

---

## Related Work

- **Code Quality Proposal:** See `self_spec_code_quality_proposal.md` for full refactoring plan
- **Architecture Docs:** See `CLAUDE.md` for V1 architecture overview
- **StreamingLLM Paper:** https://arxiv.org/abs/2309.17453

---

**End of Document**
