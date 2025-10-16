# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for Large Language Models (LLMs). This document focuses on the **V1 architecture**, specifically the scheduling system in `/home/cc2869/repositories/vllm/vllm/v1`.

**V1 Architecture Status:** Alpha release with 1.7x speedup over V0, cleaner code, zero-overhead prefix caching, and enhanced multimodal support.

## V1 Architecture - Core Components

The V1 architecture is located in `vllm/v1/` and follows a **multi-layered design** with clear separation of concerns:

### Architectural Layers

```
1. User-Facing Layer
   ├── LLM (vllm/entrypoints/llm.py) - Offline batch inference
   ├── AsyncLLM (vllm/v1/engine/async_llm.py) - Online serving with asyncio
   └── LLMEngine (vllm/v1/engine/llm_engine.py) - Legacy synchronous API

2. Configuration Layer
   ├── EngineArgs - Command-line parameter definitions
   └── VllmConfig - Consolidated configuration settings

3. Processing Layer
   ├── Processor (vllm/v1/engine/processor.py) - Converts inputs to EngineCoreRequest
   └── OutputProcessor (vllm/v1/engine/output_processor.py) - Transforms outputs to user format

4. IPC (Inter-Process Communication) Layer
   ├── EngineCoreClient (vllm/v1/engine/core_client.py)
   │   ├── InprocClient - Direct in-process calls
   │   ├── SyncMPClient - Blocking ZMQ socket communication
   │   └── AsyncMPClient - Non-blocking ZMQ socket communication
   └── Uses ZMQ sockets + shared memory queues

5. Engine Core (vllm/v1/engine/core.py)
   ├── Scheduler (vllm/v1/core/sched/scheduler.py) ⭐ KEY COMPONENT
   │   ├── KVCacheManager (vllm/v1/core/kv_cache_manager.py)
   │   │   ├── BlockPool (vllm/v1/core/block_pool.py) - LRU block allocation
   │   │   └── SingleTypeKVCacheManager - Attention-aware caching
   │   └── EncoderCacheManager - Multimodal encoder cache
   └── StructuredOutputManager - Grammar-based output constraints

6. Execution Layer
   ├── Executor (vllm/v1/executor/)
   │   ├── MultiprocExecutor - Multi-GPU with multiprocessing
   │   └── RayDistributedExecutor - Multi-node with Ray
   └── Worker (vllm/v1/worker/)
       ├── Worker - Device-level operations and model initialization
       └── ModelRunner (vllm/v1/worker/gpu_model_runner.py) - Forward pass + sampling
```

### Request Lifecycle in V1

```
User Input → Processor → EngineCoreRequest → Scheduler (waiting queue)
                                                  ↓
                                        Scheduling Decision
                                                  ↓
                            SchedulerOutput (batched requests)
                                                  ↓
                                    Executor → Workers → ModelRunner
                                                  ↓
                                         Model Forward Pass
                                                  ↓
                                    ModelRunnerOutput (tokens)
                                                  ↓
                            Scheduler.update_from_output()
                                                  ↓
                                    OutputProcessor → User Output
```

## Key File Locations

### Scheduler (⭐ Primary Focus)
- **`vllm/v1/core/sched/scheduler.py`**: Main scheduler implementation (`Scheduler` class)
- **`vllm/v1/core/sched/interface.py`**: Abstract interface (`SchedulerInterface`)
- **`vllm/v1/core/sched/output.py`**: Scheduler output data structures (`SchedulerOutput`, `NewRequestData`, `CachedRequestData`)
- **`vllm/v1/core/sched/utils.py`**: Utility functions (e.g., `check_stop`)

### Request Management
- **`vllm/v1/request.py`**: Request class with state management, including:
  - `Request`: Main request object
  - `SelfSpecState`: State machine for self-speculative decoding (NORMAL, ACCUMULATING, VERIFYING)
  - `RequestStatus`: Request lifecycle states (WAITING, RUNNING, FINISHED, etc.)

### KV Cache Management
- **`vllm/v1/core/kv_cache_manager.py`**: High-level KV cache manager
- **`vllm/v1/core/block_pool.py`**: Memory block pool for KV cache
- **`vllm/v1/core/kv_cache_utils.py`**: Utilities for KV cache operations, hashing, and prefix matching
- **`vllm/v1/core/single_type_kv_cache_manager.py`**: Type-specific KV cache management

### Engine
- **`vllm/v1/engine/core.py`**: EngineCore - inner loop of V1 engine
- **`vllm/v1/engine/llm_engine.py`**: High-level LLMEngine interface
- **`vllm/v1/engine/async_llm.py`**: Async engine implementation

### Worker & Model Runner
- **`vllm/v1/worker/gpu_worker.py`**: GPU worker implementation
- **`vllm/v1/worker/gpu_model_runner.py`**: Model execution logic

## Scheduler Deep Dive

### Scheduler Responsibilities

The `Scheduler` class (vllm/v1/core/sched/scheduler.py) is responsible for:

1. **Request Queue Management**: Maintains `waiting` (deque) and `running` (list) queues
2. **Token Budget Allocation**: Distributes `max_num_batched_tokens` across requests
3. **KV Cache Slot Allocation**: Coordinates with `KVCacheManager` to allocate memory blocks
4. **Prefix Cache Matching**: Identifies cached prompt prefixes to avoid recomputation
5. **Chunked Prefill**: Breaks large prompts into chunks (`long_prefill_token_threshold`)
6. **Speculative Decoding**: Handles draft tokens and verification
7. **Self-Speculative Decoding**: Custom speculation mode with accumulation and verification
8. **Sparse Attention**: Manages selective KV indices for sparse attention patterns
9. **Encoder Cache**: Manages multimodal encoder inputs (vision, etc.)
10. **Preemption**: Evicts lower-priority requests when out of memory

### Scheduling Algorithm (schedule() method)

The scheduler uses a **unified token-centric approach** - no separate "prefill" and "decode" phases. This design enables:
- **Continuous Batching**: New requests can join ongoing batches at any time
- **Chunked Prefill**: Long prompts processed in smaller chunks
- **Mixed Batches**: Prefill and decode requests in the same batch

```python
def schedule(self) -> SchedulerOutput:
    # Each request has:
    # - num_computed_tokens: tokens already processed
    # - num_tokens_with_spec: total tokens including prompt, output, and spec tokens
    #
    # Goal: Assign tokens to requests so num_computed_tokens catches up to num_tokens_with_spec
```

**Scheduling Policies:**
- **Default**: FCFS (First-Come-First-Served) for `waiting` queue
- **Extensible**: Can implement priority-based or custom policies

**Key scheduling steps:**

1. **Schedule RUNNING requests first** (lines 236-405):
   - Check for self-spec verification trigger
   - Calculate `num_new_tokens` to schedule (respects token budget)
   - Handle chunked prefills (max `long_prefill_token_threshold` tokens)
   - Allocate KV cache slots via `kv_cache_manager.allocate_slots()`
   - Preempt lowest-priority request if out of memory
   - Track encoder inputs, spec tokens, sparse attention indices

2. **Schedule WAITING requests** (lines 419-587):
   - Skip if max running requests reached
   - Handle special states: `WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_FSM`
   - Check LoRA constraints (`max_loras`)
   - **Prefix cache matching** via `kv_cache_manager.get_computed_blocks()`
   - Handle external KV transfers (P/D, offloading)
   - Allocate KV cache and move to RUNNING queue

3. **Return SchedulerOutput** (lines 639-657):
   - Lists of new/resumed/running requests
   - Token counts per request
   - Block IDs for KV cache
   - Spec tokens, encoder inputs, sparse attention indices
   - Grammar bitmasks for structured output

### Request State Machine

```
WAITING → RUNNING → FINISHED_*
           ↓  ↑
        PREEMPTED
```

States in `RequestStatus` (vllm/v1/request.py):
- `WAITING`: In waiting queue
- `WAITING_FOR_REMOTE_KVS`: Waiting for KV transfer (disaggregated serving)
- `WAITING_FOR_FSM`: Waiting for structured output grammar compilation
- `RUNNING`: Currently being processed
- `PREEMPTED`: Evicted due to memory pressure (returns to WAITING)
- `FINISHED_STOPPED`: Finished with EOS or stop token
- `FINISHED_LENGTH_CAPPED`: Hit max tokens
- `FINISHED_ABORTED`: Aborted by user
- `FINISHED_IGNORED`: Ignored (e.g., decode-only request)

### Self-Speculative Decoding State Machine

```
NORMAL → ACCUMULATING → VERIFYING → NORMAL
  ↓                                     ↑
  └─────────────────────────────────────┘
```

States in `SelfSpecState` (vllm/v1/request.py):
- `NORMAL`: Regular token generation (tokens committed to `output_token_ids`)
- `ACCUMULATING`: Collecting tokens in `_pending_output_tokens` buffer (not yet committed)
- `VERIFYING`: Verifying accumulated tokens (moved to `spec_token_ids` for verification)

**Key scheduling behavior:**
- **Accumulating phase** (lines 325-332):
  - Uses `delay_cache_blocks=True` in `allocate_slots()` to avoid caching speculative tokens
  - Builds up sparse attention indices (sink + recent tokens)
  - Scheduler increments `num_computed_tokens` but tokens stay in pending buffer

- **Verification trigger** (lines 240-276):
  - Checks `should_start_self_spec_verification()` when threshold reached
  - Adjusts `num_computed_tokens` to exclude pending tokens
  - Moves pending tokens to `spec_token_ids` for verification
  - Clears sparse indices (use full KV during verification)

- **Verification phase** (lines 860-873 in update_from_output):
  - Model runner verifies tokens (rejection sampling)
  - Rejected tokens reduce `num_computed_tokens`
  - State resets to NORMAL after verification

### update_from_output() method

After model execution, the scheduler updates state based on `ModelRunnerOutput`:

1. **Token acceptance** (lines 832-918):
   - Extract sampled tokens for each request
   - Handle spec token verification and rejection
   - Append tokens to request's output (or pending buffer if ACCUMULATING)
   - Check stop conditions

2. **Sparse attention updates** (lines 929-938):
   - Update `req_to_sparse_selected_kv_indices` with sink + recent tokens
   - Track `full_kv_start_offset` for transition to full KV

3. **State transitions**:
   - `ACCUMULATING → VERIFYING`: When verification starts
   - `VERIFYING → NORMAL`: After verification completes
   - `NORMAL → ACCUMULATING`: After committing a token (if self-spec enabled)
   - `RUNNING → FINISHED_*`: When request completes

4. **Return outputs** (lines 976-997):
   - Skip `EngineCoreOutput` during ACCUMULATING (no committed tokens yet)
   - Generate output only when tokens are committed or request finishes

### KV Cache Slot Allocation

Key method: `kv_cache_manager.allocate_slots()` (vllm/v1/core/kv_cache_manager.py)

Parameters:
- `num_new_tokens`: Number of tokens to allocate slots for
- `num_draft_tokens`: Draft tokens for speculative decoding
- `num_lookahead_tokens`: Lookahead slots for EAGLE
- `delay_cache_blocks`: If True, don't cache blocks yet (used for self-spec accumulating)

The method:
1. Computes required blocks: `num_required_blocks = ceil(num_new_tokens / block_size)`
2. Tries to allocate from block pool
3. Returns `None` if allocation fails (triggers preemption)
4. Returns `KVCacheBlocks` object with allocated block IDs

### Prefix Caching

**Purpose**: Reuse computed KV values for requests with shared token prefixes to avoid redundant computation.

**When a new request arrives:**
1. Hash prompt tokens in blocks (`hash_request_tokens()` in kv_cache_utils.py)
2. Look up hashes in block pool (`get_computed_blocks()`)
3. Reuse cached blocks and set `num_computed_tokens`
4. Only allocate/compute remaining tokens

**Block Sharing Mechanism:**
- Uses **reference counting** for safe block sharing across requests
- Blocks remain cached until all references are released
- **LRU (Least Recently Used)** eviction policy when memory is full
- Tracks prefix cache statistics (hits/misses) for monitoring

**Attention-Type-Aware Caching:**
The prefix cache system supports different attention patterns:

1. **Full Attention** (default):
   - Sequential prefix matching
   - All tokens can be cached and reused

2. **Sliding Window Attention**:
   - Reverse lookup within window size
   - Automatically evicts blocks outside window
   - Respects `attention_window_size` constraint

3. **Chunked Local Attention**:
   - Window-aware lookup respecting chunk boundaries
   - Ensures cache hits align with attention chunks

**Important:** Prefix caching is skipped if:
- `enable_prefix_caching=False`
- Request asks for `prompt_logprobs` (need to recompute for logprobs)
- Different attention patterns may limit cache reuse

## Multiprocessing Architecture

The V1 engine uses a **multi-process architecture** for performance and isolation:

```
Main Process (Client Interface)
    ↓ ZMQ Socket / Shared Memory
Engine Core Process
  ├── Input Thread (receives requests)
  ├── Output Thread (returns results)
  └── Main Loop (scheduling + coordination)
      ↓ ZMQ Socket / Shared Memory
Worker Processes (1 per GPU)
  └── ModelRunner (model execution)
```

**Communication Modes:**
1. **InprocClient**: Direct function calls (single process, fastest)
2. **SyncMPClient**: Blocking ZMQ sockets (multiprocess, synchronous)
3. **AsyncMPClient**: Non-blocking ZMQ sockets (multiprocess, async)

**Benefits:**
- Process isolation prevents GPU memory leaks
- Efficient inter-process communication via shared memory
- Supports distributed execution across nodes with Ray
- Clean separation of scheduling and execution

## Development Commands

### Building & Testing

```bash
# Install V1 dependencies
pip install -e .

# Run V1-specific tests
pytest tests/ --optional  # Some V1 tests are marked optional
pytest tests/ -m "not skip_v1"  # Skip tests that don't support V1

# Run scheduler-specific tests
pytest tests/v1/  # If they exist, or:
pytest tests/ -k "scheduler"

# Run with V1 engine (different communication modes)
VLLM_USE_V1=1 python examples/offline_inference/basic.py  # Auto-detect mode
VLLM_USE_V1=1 VLLM_USE_INPROC_CLIENT=1 python script.py  # In-process (no IPC)
VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model <model>  # API server
```

### Debugging Scheduler

```bash
# Enable debug logging
export VLLM_LOGGING_LEVEL=DEBUG

# Trace scheduler decisions
# Add print statements in scheduler.py schedule() method
# Key locations:
# - Line 237: Start of RUNNING request scheduling
# - Line 420: Start of WAITING request scheduling
# - Line 639: SchedulerOutput construction

# Check KV cache usage
# In scheduler.py, access: self.kv_cache_manager.usage
```

## Making Scheduler Modifications

### Common Modification Scenarios

#### 1. Changing Scheduling Policy

**File:** `vllm/v1/core/sched/scheduler.py`

Current policy: FCFS (First-Come-First-Served) for WAITING queue

To implement priority-based scheduling:
```python
# Line 420: Replace
while self.waiting and token_budget > 0:
    request = self.waiting[0]  # FCFS

# With:
while self.waiting and token_budget > 0:
    request = max(self.waiting, key=lambda r: r.priority)  # Priority-based
```

**Note:** Need to add `priority` field to `Request` class in `vllm/v1/request.py`

#### 2. Adjusting Token Budget Allocation

**File:** `vllm/v1/core/sched/scheduler.py`

Current: Equal opportunity for all requests (up to token budget)

Lines 278-285: Controls how many tokens to schedule per request
```python
num_new_tokens = (request.num_tokens_with_spec - request.num_computed_tokens)
if (0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens):
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
```

To prioritize decode over prefill:
```python
is_prefill = request.num_computed_tokens < request.num_prompt_tokens
if is_prefill and len(scheduled_running_reqs) > 0:
    # If there are decode requests, limit prefill to smaller chunks
    num_new_tokens = min(num_new_tokens, 256)
```

#### 3. Custom Preemption Policy

**File:** `vllm/v1/core/sched/scheduler.py`

Current: Preempts lowest-priority (last in running queue)

Lines 339-361: Preemption logic
```python
if new_blocks is None:
    # Preempt the lowest-priority request.
    preempted_req = self.running.pop()  # Last = lowest priority
```

To preempt based on custom criteria:
```python
if new_blocks is None:
    # Preempt request with most tokens (least work lost)
    preempted_req = max(self.running, key=lambda r: r.num_computed_tokens)
    self.running.remove(preempted_req)
```

#### 4. Modifying Self-Spec Verification Threshold

**File:** `vllm/v1/core/sched/scheduler.py`

Current: Fixed threshold from config

Line 182-186: Verification check
```python
def should_start_self_spec_verification(self, request: Request) -> bool:
    assert self.use_self_specs
    return (request.self_spec_state == SelfSpecState.ACCUMULATING and
            len(request._pending_output_tokens) >= self.self_spec_threshold)
```

To use dynamic threshold based on confidence:
```python
def should_start_self_spec_verification(self, request: Request) -> bool:
    assert self.use_self_specs
    if request.self_spec_state != SelfSpecState.ACCUMULATING:
        return False

    # Dynamic threshold based on average token probability
    avg_prob = sum(request.pending_token_probs) / len(request.pending_token_probs)
    dynamic_threshold = int(self.self_spec_threshold * (1.0 / avg_prob))

    return len(request._pending_output_tokens) >= dynamic_threshold
```

**Note:** Would need to add `pending_token_probs` tracking to `Request` class

#### 5. Adding New Scheduling Constraints

**File:** `vllm/v1/core/sched/scheduler.py`

Example: Limit number of requests per user

1. Add user tracking to `Request` in `vllm/v1/request.py`:
```python
class Request:
    def __init__(self, ..., user_id: Optional[str] = None):
        self.user_id = user_id
```

2. Track per-user request counts in `Scheduler.__init__`:
```python
self.user_request_counts: dict[str, int] = defaultdict(int)
self.max_requests_per_user = 5  # Config
```

3. Check constraint in scheduling loop (line ~424):
```python
if request.user_id and self.user_request_counts[request.user_id] >= self.max_requests_per_user:
    self.waiting.popleft()
    skipped_waiting_requests.appendleft(request)
    continue
```

4. Update counts in `update_from_output` when requests finish

### Testing Scheduler Changes

1. **Unit tests**: Add to `tests/v1/core/test_scheduler.py` (create if doesn't exist)
2. **Integration tests**: Test with full engine in `tests/v1/`
3. **Benchmarks**: Run benchmarks in `benchmarks/` to measure impact

Example unit test structure:
```python
# tests/v1/core/test_scheduler.py
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

def test_custom_scheduling_policy():
    scheduler = Scheduler(...)

    # Add requests with different priorities
    req1 = Request(..., priority=1)
    req2 = Request(..., priority=10)

    scheduler.add_request(req1)
    scheduler.add_request(req2)

    # Schedule and check that high priority goes first
    output = scheduler.schedule()
    assert req2.request_id in output.scheduled_new_reqs[0].req_id
```

## Important Configuration Parameters

Located in `VllmConfig.scheduler_config` (vllm/config.py):

- `max_num_seqs`: Max concurrent requests (default: 256)
- `max_num_batched_tokens`: Token budget per step (default: depends on model)
- `long_prefill_token_threshold`: Max tokens per prefill chunk (default: no limit)
- `scheduler_cls`: Custom scheduler class (default: `V1Scheduler`)

Accessed in scheduler as:
```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
```

## Environment Variables

V1-specific environment variables (vllm/envs.py):

- `VLLM_USE_V1`: Enable V1 engine (default: auto-detect)
- `VLLM_LOGGING_LEVEL`: Logging level (DEBUG for scheduler debugging)
- `VLLM_TRACE_FUNCTION`: Trace function calls (performance overhead)

## Key Data Structures

### Request Objects

**EngineCoreRequest** (vllm/v1/engine/__init__.py):
- User-facing request format from Processor
- Contains: `request_id`, `prompt_token_ids`, `mm_inputs`, `sampling_params`, etc.

**Request** (vllm/v1/request.py):
- Internal scheduler representation
- Tracks: `num_computed_tokens`, `num_tokens`, `output_token_ids`, `self_spec_state`, etc.
- Created from `EngineCoreRequest` in scheduler

### Scheduling Data Structures

**SchedulerOutput** (vllm/v1/core/sched/output.py):
```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]  # First-time scheduled
    scheduled_cached_reqs: list[CachedRequestData]  # Previously scheduled
    num_scheduled_tokens: dict[str, int]  # req_id -> num_tokens
    total_num_scheduled_tokens: int  # Sum of all tokens in batch
    scheduled_spec_decode_tokens: dict[str, list[int]]  # Draft tokens
    sparse_selected_kv_indices_of_scheduled_reqs: dict[str, list[int]]  # Sparse attention
    full_kv_start_offset: dict[str, int]  # Transition point to full KV
    scheduled_encoder_inputs: dict[str, list[int]]  # Multimodal inputs
    num_common_prefix_blocks: list[int]  # Common prefix for cascade attention
    finished_req_ids: set[str]  # Finished since last step
    free_encoder_input_ids: list[tuple[str, int]]  # Encoder cache to free
    structured_output_request_ids: dict[str, int]  # Grammar-constrained requests
    grammar_bitmask: Optional[NDArray]  # Token validity mask
    kv_connector_metadata: Optional[...]  # P/D metadata
```

**NewRequestData** (vllm/v1/core/sched/output.py):
- Sent to workers for first-time scheduled requests
- Contains full request data: prompt tokens, multimodal inputs, sampling params, block IDs

**CachedRequestData** (vllm/v1/core/sched/output.py):
- Sent to workers for already-cached requests (reduces IPC overhead)
- Contains only diffs: new token IDs, new block IDs, computed tokens count

### Execution Data Structures

**ModelRunnerOutput** (vllm/v1/outputs.py):
```python
@dataclass
class ModelRunnerOutput:
    sampled_token_ids: list[list[int]]  # Generated tokens per request
    spec_token_ids: Optional[list[list[int]]]  # Draft tokens for next step
    logprobs: Optional[...]  # Token logprobs
    prompt_logprobs_dict: dict[str, ...]  # Prompt logprobs per request
    req_id_to_index: dict[str, int]  # Mapping request IDs to batch indices
    finished_recving: Optional[set[str]]  # P/D: finished receiving KV
    finished_sending: Optional[set[str]]  # P/D: finished sending KV
```

**EngineCoreOutputs** (vllm/v1/engine/__init__.py):
- Returned from `scheduler.update_from_output()`
- Contains: list of `EngineCoreOutput` per request, `SchedulerStats`
- Sent back through IPC layer to user

### KV Cache Data Structures

**KVCacheBlocks** (vllm/v1/core/kv_cache_manager.py):
```python
@dataclass
class KVCacheBlocks:
    blocks: list[KVCacheBlock]  # List of allocated blocks

    # Methods:
    def get_block_ids() -> list[list[int]]  # Convert to 2D list
    def get_unhashed_block_ids() -> list[int]  # Blocks not yet cached
```

**KVCacheBlock** (vllm/v1/core/kv_cache_utils.py):
- Represents a single KV cache block
- Contains: `block_id`, `block_hash`, `num_tokens`, `ref_count`

## Documentation & References

**V1 Design Docs** (docs/design/v1/):
- `prefix_caching.md`: Zero-overhead prefix caching design
- `metrics.md`: Metrics and monitoring
- `torch_compile.md`: Torch compilation support

**Architecture Overview**: `docs/design/arch_overview.md`

**Blog Post**: [V1 Alpha Release](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

## V1 Optimization Strategies

The V1 architecture achieves 1.7x speedup over V0 through several key optimizations:

### 1. Continuous Batching
- **What**: Dynamically add new requests to ongoing batches without waiting for batch completion
- **Benefit**: Higher GPU utilization, lower latency for new requests
- **Implementation**: Scheduler checks `waiting` queue every step and adds requests if token budget allows

### 2. Chunked Prefill
- **What**: Break large prompt processing into smaller chunks
- **Benefit**: Prevents blocking decode requests, better fairness
- **Configuration**: Set `long_prefill_token_threshold` (e.g., 4096 tokens per chunk)
- **Scheduler Logic**: Lines 281-284 limit `num_new_tokens` per step

### 3. Zero-Overhead Prefix Caching
- **What**: Reuse KV cache for shared prompt prefixes with no additional latency
- **Benefit**: Massive speedup for requests with common prefixes (e.g., system prompts)
- **Implementation**: Block-level hashing + reference counting + LRU eviction
- **Use Cases**:
  - System prompts reused across requests
  - Multi-turn conversations
  - Few-shot examples in prompts

### 4. Efficient IPC with CachedRequestData
- **What**: Send only diffs for cached requests, not full data
- **Benefit**: Reduces IPC overhead by ~90% for running requests
- **Implementation**: `NewRequestData` (full) vs `CachedRequestData` (diff-only)

### 5. Attention-Aware Memory Management
- **What**: Tailor KV cache strategy to attention pattern
- **Benefit**: More efficient caching for sliding window / chunked attention
- **Patterns**:
  - Full Attention: Cache everything
  - Sliding Window: Auto-evict old blocks
  - Chunked Local: Respect chunk boundaries

### 6. Block Pool with LRU Eviction
- **What**: Centralized memory pool with smart eviction
- **Benefit**: Better memory utilization, automatic cleanup
- **Implementation**: `BlockPool` in vllm/v1/core/block_pool.py

### 7. Lazy Block Cleanup
- **What**: Delay freeing cached blocks until memory pressure
- **Benefit**: Increases prefix cache hit rate
- **Trade-off**: Uses more memory but improves performance

## Performance Monitoring

Track these metrics to understand scheduler behavior:

**From `SchedulerStats`** (vllm/v1/metrics/stats.py):
- `num_running_reqs`: Active requests in batch
- `num_waiting_reqs`: Queued requests
- `gpu_cache_usage`: KV cache utilization (0.0-1.0)
- `total_num_scheduled_tokens`: Tokens in current batch
- `prefix_cache_stats.hit_rate`: Fraction of cache hits

**Enable with:**
```python
scheduler = Scheduler(..., log_stats=True)
stats = scheduler.make_stats()
```

## Common Gotchas

1. **Don't modify `num_computed_tokens` directly in schedule()**: It's updated at the end (line 682)
2. **KV cache allocation can fail**: Always check if `allocate_slots()` returns `None`
3. **Self-spec accumulating doesn't cache**: Uses `delay_cache_blocks=True` to avoid caching speculative tokens
4. **Sparse attention indices are per-request**: Stored in `req_to_sparse_selected_kv_indices` dict
5. **SchedulerOutput is reused**: Don't modify in-place, create new objects
6. **Request queues are different types**: `waiting` is deque, `running` is list (different APIs)
7. **Prefix caching benefits vary by workload**: Most effective when requests share long common prefixes
8. **Chunked prefill adds steps**: Large prompts take multiple scheduling steps to complete
9. **IPC overhead matters**: Use `InprocClient` for single-process benchmarking to eliminate IPC noise

## V1 Design Philosophy & Extension Points

The V1 architecture is designed for **modularity and extensibility**. Here's where to make different types of modifications:

### Scheduling Policy Changes
**File**: `vllm/v1/core/sched/scheduler.py`
- Modify `schedule()` method (lines 195-685)
- Change how requests are selected from `waiting` queue
- Adjust token budget allocation per request
- Customize preemption policy

### Request State Management
**File**: `vllm/v1/request.py`
- Add new fields to `Request` class
- Extend state machines (e.g., new `SelfSpecState` values)
- Track additional metadata per request

### KV Cache Strategy
**Files**: `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`
- Modify block allocation policy
- Change eviction strategy (current: LRU)
- Add new caching patterns for different attention types

### Custom Scheduler Implementation
**Approach**: Implement `SchedulerInterface` (vllm/v1/core/sched/interface.py)
```python
from vllm.v1.core.sched.interface import SchedulerInterface

class MyCustomScheduler(SchedulerInterface):
    def schedule(self) -> SchedulerOutput:
        # Your custom scheduling logic
        pass

    def update_from_output(self, ...) -> EngineCoreOutputs:
        # Your custom output handling
        pass
    # ... implement other abstract methods
```

Then configure:
```python
from vllm import EngineArgs
args = EngineArgs(
    model="your-model",
    scheduler_cls="path.to.MyCustomScheduler"
)
```

### Model Runner Modifications
**Files**: `vllm/v1/worker/gpu_model_runner.py`
- Change how inputs are prepared for the model
- Modify sampling strategy
- Add custom attention mechanisms

### IPC Communication Changes
**Files**: `vllm/v1/engine/core_client.py`, `vllm/v1/serial_utils.py`
- Optimize serialization (currently uses msgpack)
- Add new communication modes
- Reduce IPC overhead

## Next Steps

After making scheduler modifications:
1. **Add unit tests** for new logic
2. **Run integration tests** with V1 engine
3. **Benchmark performance** impact (measure throughput, latency, memory)
4. **Profile with metrics** enabled (`log_stats=True`)
5. **Update documentation** if adding new features
6. **Consider backward compatibility** with V0 if applicable
7. **Test edge cases**: OOM scenarios, preemption, prefix cache misses

## Debugging Tips

### Common Issues When Modifying Scheduler

1. **Memory Leaks**:
   - Ensure `_free_request()` is called for finished requests
   - Check that `_cached_reqs_data` is properly cleaned up
   - Verify block reference counts are decremented

2. **Incorrect Token Counts**:
   - `num_computed_tokens` should only be updated at end of `schedule()`
   - `num_tokens` includes pending tokens for self-spec
   - `num_tokens_with_spec` includes both spec and pending tokens

3. **KV Cache Corruption**:
   - Don't modify `block_ids` after allocation
   - Respect `delay_cache_blocks=True` for speculative tokens
   - Ensure blocks are freed in correct order

4. **Deadlocks in Multiprocess**:
   - Don't block on ZMQ sockets in scheduler
   - Use timeouts for IPC operations
   - Check that worker processes are still alive

### Useful Debug Prints

Add to `scheduler.py` for debugging:

```python
# In schedule() method
logger.debug(f"Step: waiting={len(self.waiting)}, running={len(self.running)}, "
             f"token_budget={token_budget}, cache_usage={self.kv_cache_manager.usage:.2f}")

# In update_from_output()
logger.debug(f"Request {req_id}: computed={request.num_computed_tokens}, "
             f"total={request.num_tokens}, state={request.self_spec_state}")

# For KV cache debugging
logger.debug(f"Allocated blocks for {req_id}: {new_blocks.get_block_ids()}")
```

## Getting Help

- **Code location**: `/home/cc2869/repositories/vllm/vllm/v1/`
- **Main scheduler**: `vllm/v1/core/sched/scheduler.py` (~1221 lines)
- **Request state**: `vllm/v1/request.py` (~200 lines, focus on lines 18-195)
- **Tests**: `tests/v1/` and `tests/` with `-m "not skip_v1"`
- **DeepWiki**: https://deepwiki.com/vllm-project/vllm/ for architecture docs
- **GitHub Issues**: https://github.com/vllm-project/vllm/issues (use `[Core]` tag for scheduler)
- **Slack**: https://slack.vllm.ai (developer community)
