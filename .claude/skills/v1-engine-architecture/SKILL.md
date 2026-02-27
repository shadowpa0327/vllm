---
name: v1-engine-architecture
description: >
  This skill should be used when the user asks about "vLLM v1 architecture",
  "engine execution flow", "how the scheduler works", "GPUModelRunner",
  "InputBatch", "EngineCore main loop", "request lifecycle", "deferred sampling",
  "execute_model flow", "AsyncLLM", "EngineCoreClient", "KV cache management",
  "persistent batching", "output return path", or needs to understand v1 engine
  internals before designing or modifying features. Also trigger when the user
  mentions "v1 engine", "scheduler schedule()", "SchedulerOutput", "sample_tokens",
  or asks how requests flow through vLLM.
---

# vLLM v1 Engine Architecture

Reference skill for the vLLM v1 inference engine internals. Load this to understand
the architecture before designing features, debugging issues, or navigating the codebase.

## Architecture at a Glance

```
User API Call
  → AsyncLLM (v1/engine/async_llm.py)
    → EngineCoreClient / AsyncMPClient (ZMQ IPC)
      → EngineCoreProc (separate process, 3 threads)
        → EngineCore.step():
            1. Scheduler.schedule()        → SchedulerOutput
            2. Executor.execute_model()    → None (deferred)
            3. Grammar bitmask (CPU)       → overlaps GPU forward
            4. Executor.sample_tokens()    → ModelRunnerOutput
            5. Scheduler.update_from_output() → EngineCoreOutputs
      → ZMQ back to client
    → OutputProcessor (detokenize)
  → yield RequestOutput to user
```

## Key Components

### Frontend: AsyncLLM + EngineCoreClient
- `AsyncLLM` (`v1/engine/async_llm.py`) — User-facing async generator API
- `AsyncMPClient` (`v1/engine/core_client.py`) — ZMQ ROUTER/DEALER + PUSH/PULL IPC
- Message protocol: msgpack-encoded, request types: ADD, ABORT, START_DP_WAVE, UTILITY

### EngineCore (`v1/engine/core.py`)
- `EngineCoreProc` — 3 threads: input socket, main busy loop, output socket
- `EngineCore.step()` — The core loop: schedule → execute → grammar → sample → update

### Scheduler (`v1/core/sched/scheduler.py`)
- Unified scheduling — no separate prefill/decode phases
- Phase 1: schedule RUNNING requests (allocate KV slots, preempt if needed)
- Phase 2: schedule WAITING requests (prefix cache hits, chunked prefill)
- Eager `num_computed_tokens` advance after scheduling, adjusted on spec decode rejection

### Executor Chain
```
Executor.execute_model() → collective_rpc("execute_model")
  → WorkerWrapperBase → GPUWorker → GPUModelRunner.execute_model()
```
- `UniProcExecutor` (single GPU), `MultiprocExecutor` (multi-GPU), `RayDistributedExecutor` (multi-node)

### GPUModelRunner (`v1/worker/gpu_model_runner.py`)
- `execute_model()` — preprocess → prepare inputs → build attention metadata → forward → compute logits → save state → return None
- `sample_tokens()` — apply grammar bitmask → sample → return ModelRunnerOutput
- Deferred sampling enables CPU/GPU overlap

### InputBatch (`v1/worker/gpu_input_batch.py`)
- Persistent batch that survives across steps — only deltas applied
- Per-step: remove finished → add new → update running → condense → refresh metadata

## Key Design Decisions

1. **Unified scheduling** — Gap between `num_computed_tokens` and `num_tokens_with_spec` closed regardless of prefill/decode
2. **Persistent batch** — `InputBatch` avoids re-creating GPU tensors every iteration
3. **Deferred sampling** — `execute_model()` returns None; grammar bitmask overlaps with GPU forward pass
4. **Eager token count advance** — Enables immediate re-scheduling for chunked prefill
5. **Two-phase output** — `NewRequestData` (full) vs `CachedRequestData` (delta) minimizes serialization
6. **ZMQ + msgpack IPC** — Frontend ↔ EngineCore in separate processes
7. **Preemption** — Lowest-priority request evicted when KV cache exhausted; fully recomputed on reschedule
8. **Prefix caching** — Block hashing finds cached KV blocks; computed tokens are "free"

## Key File Locations

| Component | File |
|-----------|------|
| AsyncLLM | `v1/engine/async_llm.py` |
| EngineCoreClient | `v1/engine/core_client.py` |
| EngineCore | `v1/engine/core.py` |
| Scheduler | `v1/core/sched/scheduler.py` |
| SchedulerOutput | `v1/core/sched/output.py` |
| Executor (abstract) | `v1/executor/abstract.py` |
| GPUWorker | `v1/worker/gpu_worker.py` |
| GPUModelRunner | `v1/worker/gpu_model_runner.py` |
| InputBatch | `v1/worker/gpu_input_batch.py` |
| Request | `v1/request.py` |
| EngineCoreRequest/Output | `v1/engine/__init__.py` |
| ModelRunnerOutput | `v1/outputs.py` |

## Detailed Reference

For complete architecture documentation including:
- Full invocation chains with line numbers
- Detailed scheduler algorithm (Phase 1 & 2)
- Complete `execute_model()` step-by-step breakdown
- Data type transformation chain (User Input → RequestOutput)
- Request status lifecycle diagram
- InputBatch data structures and per-step update flow

Consult: **`references/architecture-detail.md`**
