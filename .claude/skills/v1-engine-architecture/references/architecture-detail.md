# vLLM v1 Engine Architecture — Complete Reference

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [End-to-End Request Flow](#2-end-to-end-request-flow)
3. [Frontend: AsyncLLM & EngineCoreClient](#3-frontend-asyncllm--enginecoreclient)
4. [EngineCore: The Main Loop](#4-enginecore-the-main-loop)
5. [Scheduler](#5-scheduler)
6. [Executor → Worker → GPUModelRunner Chain](#6-executor--worker--gpumodelrunner-chain)
7. [GPUModelRunner: execute_model & sample_tokens](#7-gpumodelrunner-execute_model--sample_tokens)
8. [InputBatch: Persistent Batching](#8-inputbatch-persistent-batching)
9. [Output Return Path](#9-output-return-path)
10. [Request Lifecycle & Data Transformations](#10-request-lifecycle--data-transformations)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Architecture Overview

```
User API Call (generate / encode)
    |
AsyncLLM                           (v1/engine/async_llm.py)
    |  InputProcessor: tokenize, build EngineCoreRequest
    |
EngineCoreClient (AsyncMPClient)   (v1/engine/core_client.py)
    |  ZMQ ROUTER→DEALER (msgpack serialized)
    |
EngineCoreProc                     (v1/engine/core.py)
    |  input socket thread → input_queue → busy loop
    |
    +-- EngineCore                 (v1/engine/core.py:81)
    |     |
    |     +-- Scheduler            (v1/core/sched/scheduler.py:63)
    |     |     +-- KVCacheManager (v1/core/kv_cache_manager.py)
    |     |     +-- EncoderCacheManager
    |     |     +-- KVConnector (optional, for P/D disagg)
    |     |
    |     +-- Executor             (v1/executor/abstract.py)
    |           |
    |           +-- UniProcExecutor    (single process)
    |           +-- MultiprocExecutor  (multi-process, MessageQueue)
    |           +-- RayDistributedExecutor
    |                 |
    |                 +-- WorkerWrapperBase  (v1/worker/worker_base.py)
    |                       |
    |                       +-- GPUWorker   (v1/worker/gpu_worker.py)
    |                             |
    |                             +-- GPUModelRunner (v1/worker/gpu_model_runner.py)
    |                                   +-- InputBatch (v1/worker/gpu_input_batch.py)
    |                                   +-- Model (transformer layers)
    |                                   +-- Sampler
    |
    +-- output_queue → output socket thread
          |  ZMQ PUSH→PULL (msgpack serialized)
          |
EngineCoreClient receives EngineCoreOutputs
    |
AsyncLLM.output_handler()
    |  OutputProcessor: detokenize, build RequestOutput
    |
    +-- per-request RequestOutputCollector queue
          |
User receives RequestOutput via async generator
```

---

## 2. End-to-End Request Flow

### The Happy Path (one iteration)

```
1. User calls AsyncLLM.generate(prompt)
2. InputProcessor.process_inputs() → EngineCoreRequest (tokenized)
3. AsyncMPClient._send_input() → ZMQ send to EngineCore process
4. EngineCoreProc.process_input_sockets() thread receives → preprocess_add_request()
   → Request.from_engine_core_request() → input_queue.put()
5. EngineCoreProc.run_busy_loop() → _process_input_queue()
   → _handle_client_request(ADD, request) → scheduler.add_request()
6. _process_engine_step() → step_fn()

   EngineCore.step():
     a. scheduler.schedule()           → SchedulerOutput
     b. executor.execute_model()       → Future (non-blocking)
     c. scheduler.get_grammar_bitmask() (CPU, overlaps with GPU forward)
     d. future.result()                → None (deferred sampling)
     e. executor.sample_tokens()       → ModelRunnerOutput
     f. scheduler.update_from_output() → dict[int, EngineCoreOutputs]

7. output_queue.put(EngineCoreOutputs)
8. process_output_sockets() thread → ZMQ send to client
9. AsyncMPClient.process_outputs_socket() → outputs_queue.put()
10. AsyncLLM.output_handler() → OutputProcessor.process_outputs()
    → detokenize → RequestOutput → req_state.queue.put()
11. AsyncLLM.generate() → await q.get() → yield RequestOutput to user
```

---

## 3. Frontend: AsyncLLM & EngineCoreClient

### AsyncLLM (`v1/engine/async_llm.py`)

**Class:** `AsyncLLM` (line 70)

| Method | Lines | What it does |
|--------|-------|-------------|
| `generate()` | 528-637 | User-facing async generator; calls add_request, yields from queue |
| `add_request()` | 287-402 | Validates params, calls InputProcessor.process_inputs(), creates queue |
| `_add_request()` | 404-419 | Adds to OutputProcessor + calls engine_core.add_request_async() |
| `output_handler()` | 654-705 | Background task: pulls from engine_core, pushes to per-request queues |

### AsyncMPClient (`v1/engine/core_client.py`)

**Class:** `AsyncMPClient` (line 829), extends `MPClient` (line 449)

**ZMQ Topology:**
- Input: ROUTER socket (client) ↔ DEALER socket (per engine)
- Output: PUSH socket (engine) → PULL socket (client)

| Method | Lines | What it does |
|--------|-------|-------------|
| `add_request_async()` | 977-980 | Encodes EngineCoreRequest, sends via ZMQ input socket |
| `_send_input()` | 920-930 | Encodes with MsgpackEncoder |
| `_send_input_message()` | 932-955 | ZMQ multipart send: (engine_id, request_type, data) |
| `get_output_async()` | 909-918 | Pulls from outputs_queue (populated by socket task) |
| `process_outputs_socket()` | 880-907 | Async task: ZMQ recv → decode → outputs_queue.put() |

**Message Protocol:**
- Request: `(engine_identity_2bytes, request_type_1byte, msgpack_data...)`
- Request types: ADD=`\x00`, ABORT=`\x01`, START_DP_WAVE=`\x02`, UTILITY=`\x03`
- Response: msgpack-encoded `EngineCoreOutputs` frames

---

## 4. EngineCore: The Main Loop

### EngineCoreProc (`v1/engine/core.py:757`)

Wraps `EngineCore` with ZMQ sockets and threading. Runs in a separate process.

**Threads:**
1. **Input socket thread** (`process_input_sockets`, line 1237): ZMQ recv → decode → `preprocess_add_request()` → `input_queue.put()`
2. **Main thread** (`run_busy_loop`, line 1099): `_process_input_queue()` + `_process_engine_step()`
3. **Output socket thread** (`process_output_sockets`, line 1318): `output_queue.get()` → encode → ZMQ send

### EngineCore (`v1/engine/core.py:81`)

| Method | Lines | What it does |
|--------|-------|-------------|
| `__init__()` | 84-224 | Creates executor, KV caches, scheduler, batch queue |
| `_initialize_kv_caches()` | 226-284 | Profiles GPU, computes block count |
| `add_request()` | 289-320 | Adds Request to scheduler |
| `step()` | 376-405 | **Main loop**: schedule → execute → grammar → sample → update |
| `step_with_batch_queue()` | 417-532 | Pipeline parallelism variant with async batch queue |
| `post_step()` | 407-415 | Updates draft token IDs for spec decode |
| `preprocess_add_request()` | 732-754 | EngineCoreRequest → Request conversion |

**step() — The Core Loop** (line 376):
```python
def step(self):
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**Key insight:** `execute_model()` returns `None` (common path) to enable deferred sampling. Grammar bitmask is computed on CPU while the GPU runs the forward pass. Then `sample_tokens()` applies the bitmask and samples.

---

## 5. Scheduler

### Class: `Scheduler` (`v1/core/sched/scheduler.py:63`)

**Key state:**
- `self.requests: dict[str, Request]` — all known requests
- `self.waiting` — priority queue of waiting requests
- `self.running: list[Request]` — currently running requests
- `self.finished_req_ids: set[str]` — finished since last step

### schedule() (line 322)

**Design:** No separate "prefill" or "decode" phase. Each request has `num_computed_tokens` and `num_tokens_with_spec`; the scheduler closes the gap.

**Phase 1 — Schedule RUNNING requests** (lines 357-524):
```
For each running request:
  1. num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
  2. Cap by token_budget, max_model_len, long_prefill_token_threshold
  3. Schedule encoder inputs (multimodal)
  4. kv_cache_manager.allocate_slots(request, num_new_tokens)
     - If fails → preempt lowest-priority running request (free KV, move to waiting)
  5. Track: req_to_new_blocks, num_scheduled_tokens, token_budget
```

**Phase 2 — Schedule WAITING requests** (lines 535-816):
```
Only if no preemptions. For each waiting request:
  1. Skip if WAITING_FOR_REMOTE_KVS, WAITING_FOR_FSM, WAITING_FOR_STREAMING_REQ
  2. Check LoRA constraints
  3. kv_cache_manager.get_computed_blocks() → prefix cache hits
  4. KVConnector.get_num_new_matched_tokens() → external cache hits
  5. num_new_tokens = request.num_tokens - num_computed_tokens
  6. Cap by token_budget (chunked prefill)
  7. kv_cache_manager.allocate_slots()
  8. Move: waiting → running, status = RUNNING
```

**Output construction** (lines 841-910):
- `NewRequestData.from_request()` for new requests
- `_make_cached_request_data()` for running/resumed requests (delta updates only)
- Compute `num_common_prefix_blocks` for cascade attention
- Build KVConnectorMetadata if P/D disagg

**_update_after_schedule()** (line 934):
- Eagerly advances `request.num_computed_tokens += num_scheduled_tokens`
- Allows next scheduling step to run immediately (for chunked prefill)
- If spec tokens are rejected later, adjusted backward in `update_from_output`

### update_from_output() (line 1258)

For each scheduled request:
1. Get `sampled_token_ids` from ModelRunnerOutput
2. Handle spec decode: compare draft vs actual, adjust `num_computed_tokens` for rejections
3. `_update_request_with_output()`: append tokens, check stop conditions
4. If stopped: `_handle_stopped_request()`, `_free_request()` (free KV blocks)
5. Build `EngineCoreOutput` with new_token_ids, finish_reason, logprobs
6. Remove stopped requests from running queue
7. Return `dict[client_index, EngineCoreOutputs]`

---

## 6. Executor → Worker → GPUModelRunner Chain

### Complete Invocation Chain

```
EngineCore.step()                              (core.py:376)
  |
  v
Executor.execute_model(scheduler_output)       (abstract.py:202)
  |  calls collective_rpc("execute_model", args=(scheduler_output,))
  v
UniProcExecutor.collective_rpc()               (uniproc_executor.py:62)
  |  run_method(driver_worker, "execute_model", (scheduler_output,))
  |
  |  --- OR for multi-GPU ---
  |
MultiprocExecutor.collective_rpc()             (multiproc_executor.py:303)
  |  MessageQueue.enqueue((method, args, kwargs, output_rank))
  |  → worker process picks up, run_method()
  v
WorkerWrapperBase.execute_model()              (worker_base.py:356)
  |  self._apply_mm_cache(scheduler_output)
  |  return self.worker.execute_model(scheduler_output)
  v
GPUWorker.execute_model()                      (gpu_worker.py:636)
  |  Handle PP intermediate tensors
  |  return self.model_runner.execute_model(scheduler_output, intermediate_tensors)
  v
GPUModelRunner.execute_model()                 (gpu_model_runner.py:3357)
  |  → returns None (saves state for deferred sampling)
  |
  v  (back in EngineCore.step())
  |
Executor.sample_tokens(grammar_output)         (abstract.py:222)
  → same chain → GPUWorker.sample_tokens() (gpu_worker.py:630)
  → GPUModelRunner.sample_tokens()             (gpu_model_runner.py:3671)
  → returns ModelRunnerOutput
```

### Executor Types

| Executor | Communication | Use Case |
|----------|-------------|----------|
| `UniProcExecutor` | Direct method call | Single GPU |
| `MultiprocExecutor` | Shared memory MessageQueue | Multi-GPU, same node |
| `RayDistributedExecutor` | Ray remote calls | Multi-node |
| `ExecutorWithExternalLauncher` | External orchestration | torchrun |

---

## 7. GPUModelRunner: execute_model & sample_tokens

### execute_model() (line 3357)

```
1. PREPROCESS (line 3386):
   _update_states(scheduler_output)
     - Remove finished requests from InputBatch
     - Remove unscheduled (preempted) requests
     - Add new requests (create CachedRequestState, add to InputBatch)
     - Update running requests (tokens, blocks, computed counts)
     - Condense batch, refresh SamplingMetadata

2. PREPARE INPUTS (line 3428):
   _prepare_inputs(scheduler_output, num_scheduled_tokens_np)
     - Compute logits_indices, positions, token IDs
     - Build SpecDecodeMetadata if applicable

3. DETERMINE EXECUTION (line 3449):
   _determine_batch_execution_and_padding()
     - CUDAGraphMode: FULL / PARTIAL / NONE
     - Micro-batching (ubatching) decision
     - Cascade attention prefix lengths

4. BUILD ATTENTION METADATA (line 3524):
   _build_attention_metadata()
     - Slot mappings for KV cache writes
     - Attention backend metadata

5. PREPROCESS INPUTS (line 3540):
   _preprocess() → input_ids, positions, inputs_embeds, intermediate_tensors

6. FORWARD PASS (line 3588):
   _model_forward(input_ids, positions, inputs_embeds, ...)
     → hidden_states

7. COMPUTE LOGITS (line 3624):
   sample_hidden_states = hidden_states[logits_indices]
   logits = model.compute_logits(sample_hidden_states)

8. SAVE STATE & RETURN None (line 3655):
   self.execute_model_state = ExecuteModelState(logits, metadata, ...)
   return None  ← signals deferred sampling
```

### sample_tokens() (line 3671)

```
1. Unpack execute_model_state (line 3694)
2. Apply grammar bitmask to logits (line 3711)
3. _sample(logits, spec_decode_metadata) → sampler_output (line 3716)
4. _update_states_after_model_execute() (line 3718)
5. PP broadcast if async scheduling (line 3721)
6. Propose draft tokens for spec decode (line 3735+)
7. Return ModelRunnerOutput
```

---

## 8. InputBatch: Persistent Batching

### Class: `InputBatch` (`v1/worker/gpu_input_batch.py:81`)

The InputBatch persists across inference steps. Only deltas are applied each step.

### Core Data Structures

| Field | Type | Purpose |
|-------|------|---------|
| `token_ids_cpu` | np.ndarray (max_reqs, max_model_len) | All token IDs (prompt + output) |
| `num_computed_tokens_cpu` | np.ndarray (max_reqs,) | How far each req has been computed |
| `num_tokens_no_spec` | np.ndarray (max_reqs,) | Tokens excluding speculative |
| `num_prompt_tokens` | np.ndarray (max_reqs,) | Prompt length per request |
| `block_table` | MultiGroupBlockTable | KV cache block assignments |
| `req_id_to_index` | dict[str, int] | Request ID → batch index |
| `req_prompt_embeds` | dict[int, Tensor] | Sparse prompt embeddings |
| `temperature_cpu` | np.ndarray | Sampling temperature per req |
| `top_p_cpu`, `top_k_cpu` | np.ndarray | Sampling params |
| `sampling_metadata` | SamplingMetadata | GPU-side sampling tensors |
| `generators` | dict[int, Generator] | Per-request RNG |

### Key Methods

| Method | Lines | What it does |
|--------|-------|-------------|
| `add_request()` | 304-425 | Assign index, copy tokens/params/blocks |
| `remove_request()` | 469-520 | Clear slot, track for condensing |
| `condense()` | 632-754 | Compact batch by filling gaps |
| `swap_states()` | 522-630 | Swap two request slots |
| `refresh_metadata()` | 756-772 | Rebuild SamplingMetadata for GPU |

### Per-Step Update Flow (in `_update_states`)

```
1. Remove finished requests → input_batch.remove_request()
2. Remove unscheduled requests → input_batch.remove_request()
3. Add new requests → CachedRequestState → input_batch.add_request()
4. Update running requests:
   - input_batch.token_ids_cpu[idx] ← new output tokens
   - input_batch.num_computed_tokens_cpu[idx] ← updated count
   - input_batch.block_table ← new block IDs
5. input_batch.condense()  ← compact gaps
6. input_batch.refresh_metadata()  ← rebuild GPU sampling tensors
```

### CachedRequestState (`gpu_input_batch.py:29`)

Worker-side per-request cache:
- `req_id`, `prompt_token_ids`, `prompt_embeds`, `mm_features`
- `sampling_params`, `pooling_params`, `generator`
- `block_ids: tuple[list[int], ...]` — KV cache blocks per group
- `num_computed_tokens`, `output_token_ids`
- `lora_request`, `mrope_positions`, `xdrope_positions`

---

## 9. Output Return Path

```
GPUModelRunner.sample_tokens() → ModelRunnerOutput
  |
GPUWorker.sample_tokens() returns it
  |
Executor.sample_tokens() returns output[0]
  |
EngineCore.step():
  scheduler.update_from_output(scheduler_output, model_output)
  → dict[client_index, EngineCoreOutputs]
  |
EngineCoreProc._process_engine_step():
  output_queue.put_nowait((client_index, outputs))
  |
process_output_sockets() thread:
  output_queue.get() → MsgpackEncoder.encode_into() → ZMQ PUSH send
  |
AsyncMPClient.process_outputs_socket() async task:
  ZMQ PULL recv → MsgpackDecoder.decode() → outputs_queue.put()
  |
AsyncLLM.output_handler():
  engine_core.get_output_async() → OutputProcessor.process_outputs()
    For each EngineCoreOutput:
      - Detokenize new_token_ids → text
      - Build RequestOutput (with text, logprobs, finish_reason)
      - req_state.queue.put(request_output)
  |
AsyncLLM.generate():
  await q.get() → yield RequestOutput to user
```

### RequestOutputCollector (`output_processor.py:45`)

Per-request async queue with backpressure:
- `put()`: If producer is ahead of consumer, merges/aggregates outputs
- `get()`: Async wait, clears slot after consumption
- Supports DELTA mode (only new tokens) vs FINAL_ONLY mode

---

## 10. Request Lifecycle & Data Transformations

### Status Lifecycle

```
                    ┌─────────────────────────────┐
                    v                             |
START → WAITING → RUNNING → FINISHED_STOPPED ─(resumable)─→ WAITING_FOR_STREAMING_REQ → WAITING
            |         |        FINISHED_LENGTH_CAPPED
            |         |        FINISHED_ABORTED
            |         |        FINISHED_IGNORED
            |         |        FINISHED_ERROR
            |         |
            |         └──(preempted)──→ PREEMPTED → (back to WAITING)
            |
            ├──→ WAITING_FOR_FSM (grammar compilation) → WAITING
            └──→ WAITING_FOR_REMOTE_KVS (KV transfer) → WAITING
```

### Data Type Transformations

```
User Input (prompt string or token_ids)
  ↓  InputProcessor.process_inputs()
EngineCoreRequest (msgspec.Struct, serializable)
  |  Fields: request_id, prompt_token_ids, sampling_params, mm_features, ...
  ↓  Request.from_engine_core_request()
Request (scheduler-internal, v1/request.py:59)
  |  Added: status, _output_token_ids, _all_token_ids, block_hashes, ...
  ↓  Scheduler.schedule()
SchedulerOutput (v1/core/sched/output.py)
  |  NewRequestData: first-time requests (full data)
  |  CachedRequestData: running requests (delta only: new blocks, token counts)
  |  num_scheduled_tokens: dict[req_id, int]
  ↓  GPUModelRunner._update_states()
CachedRequestState (gpu_input_batch.py:29)
  |  Worker-side cache: block_ids, output_token_ids, sampling_params
  ↓  InputBatch.add_request() / update
InputBatch GPU tensors (token_ids, positions, block_table, sampling_metadata)
  ↓  Forward pass + sampling
ModelRunnerOutput (v1/outputs.py)
  |  sampled_token_ids, logprobs, pooler_output
  ↓  scheduler.update_from_output()
EngineCoreOutput (v1/engine/__init__.py:141)
  |  new_token_ids, finish_reason, logprobs, events
  ↓  OutputProcessor.process_outputs()
RequestOutput (vllm/outputs.py)
  |  User-facing: text, token_ids, logprobs, finished flag
  ↓  yield to user
```

---

## 11. Key Design Decisions

1. **Unified scheduling** — No prefill/decode phases. Each request has a gap between `num_computed_tokens` and `num_tokens_with_spec`; scheduler closes the gap regardless of whether it's prefill or decode.

2. **Persistent batch** — `InputBatch` survives across steps. Only deltas applied via `_update_states()`. Avoids re-creating GPU tensors every iteration.

3. **Deferred sampling** — `execute_model()` returns `None`, saves logits in `execute_model_state`. Grammar bitmask computed on CPU while GPU runs forward pass. Then `sample_tokens()` applies mask and samples. This overlaps CPU and GPU work.

4. **Eager num_computed_tokens advance** — Scheduler advances token counts in `_update_after_schedule()` right after scheduling, before model actually runs. If spec tokens are rejected, adjusted backward in `update_from_output()`. Enables immediate re-scheduling of chunked prefill requests.

5. **Two-phase output** — `SchedulerOutput` distinguishes `NewRequestData` (full state for first scheduling) vs `CachedRequestData` (deltas for ongoing requests). Minimizes serialization overhead.

6. **ZMQ + msgpack IPC** — Frontend and EngineCore run in separate processes, communicating via ZMQ with msgpack serialization. Three threads in EngineCore: input socket, main loop, output socket.

7. **Preemption** — When KV cache is exhausted, scheduler preempts lowest-priority running request: frees its KV blocks, resets `num_computed_tokens=0`, moves to waiting queue. Request will be fully recomputed when rescheduled.

8. **Prefix caching** — `kv_cache_manager.get_computed_blocks()` finds cached KV blocks via block hashing. Computed tokens are "free" — scheduler skips them. Block hashes computed incrementally per request.

---

## Key File Locations

| Component | File | Key Class/Line |
|-----------|------|---------------|
| AsyncLLM (frontend) | `v1/engine/async_llm.py` | `AsyncLLM:70` |
| EngineCoreClient | `v1/engine/core_client.py` | `AsyncMPClient:829`, `MPClient:449` |
| EngineCore | `v1/engine/core.py` | `EngineCore:81`, `EngineCoreProc:757` |
| Scheduler | `v1/core/sched/scheduler.py` | `Scheduler:63` |
| SchedulerOutput | `v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |
| Executor (abstract) | `v1/executor/abstract.py` | `Executor` |
| UniProcExecutor | `v1/executor/uniproc_executor.py` | `UniProcExecutor:26` |
| MultiprocExecutor | `v1/executor/multiproc_executor.py` | `MultiprocExecutor:93` |
| GPUWorker | `v1/worker/gpu_worker.py` | `Worker:105` |
| WorkerWrapperBase | `v1/worker/worker_base.py` | `WorkerWrapperBase:175` |
| GPUModelRunner | `v1/worker/gpu_model_runner.py` | `GPUModelRunner:375` |
| InputBatch | `v1/worker/gpu_input_batch.py` | `InputBatch:81`, `CachedRequestState:29` |
| Request | `v1/request.py` | `Request:59`, `RequestStatus` |
| EngineCoreRequest | `v1/engine/__init__.py` | `EngineCoreRequest:56` |
| EngineCoreOutput | `v1/engine/__init__.py` | `EngineCoreOutput:141`, `EngineCoreOutputs:187` |
| ModelRunnerOutput | `v1/outputs.py` | `ModelRunnerOutput` |
| OutputProcessor | `v1/engine/output_processor.py` | `OutputProcessor` |
| KVCacheManager | `v1/core/kv_cache_manager.py` | `KVCacheManager` |
