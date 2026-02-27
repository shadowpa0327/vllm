# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual E2E acceptance test for drafter weight sync.

This script measures speculative decoding acceptance length across 3 phases:
1) Baseline
2) After randomizing drafter weights
3) After restoring original drafter weights

Expected behavior: acceptance drops after randomization and recovers after
restoration.
"""

import os

# collective_rpc callables require insecure serialization in v1.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter


@dataclass
class SpecDecodeCounters:
    num_drafts: int = 0
    num_accepted_tokens: int = 0

    def __sub__(self, other: "SpecDecodeCounters") -> "SpecDecodeCounters":
        return SpecDecodeCounters(
            num_drafts=self.num_drafts - other.num_drafts,
            num_accepted_tokens=self.num_accepted_tokens - other.num_accepted_tokens,
        )


def _read_spec_decode_counters(llm: LLM) -> SpecDecodeCounters:
    counters = SpecDecodeCounters()
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            counters.num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            counters.num_accepted_tokens += metric.value
    return counters


def _run_and_measure_acceptance(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> float:
    before = _read_spec_decode_counters(llm)
    llm.generate(prompts, sampling_params=sampling_params)
    after = _read_spec_decode_counters(llm)
    delta = after - before

    if delta.num_drafts <= 0:
        raise RuntimeError(
            "No speculative drafts were recorded. "
            "Ensure speculative decoding is enabled and active."
        )
    return 1 + (delta.num_accepted_tokens / delta.num_drafts)


def _snapshot_drafter_weights(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    if drafter_model is None:
        raise RuntimeError("No drafter model found on this worker.")

    worker._weight_sync_test_drafter_snapshot = {
        name: tensor.detach().cpu().clone()
        for name, tensor in drafter_model.state_dict().items()
    }
    return {
        "ok": True,
        "num_tensors": len(worker._weight_sync_test_drafter_snapshot),
    }


def _randomize_drafter_weights(worker, seed: int):
    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    if drafter_model is None:
        raise RuntimeError("No drafter model found on this worker.")

    torch.manual_seed(seed + worker.rank)
    with torch.no_grad():
        for param in drafter_model.parameters():
            param.copy_(torch.randn_like(param))
    return {"ok": True}


def _restore_drafter_weights(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    if drafter_model is None:
        raise RuntimeError("No drafter model found on this worker.")

    snapshot = getattr(worker, "_weight_sync_test_drafter_snapshot", None)
    if snapshot is None:
        raise RuntimeError("No saved drafter snapshot found on this worker.")

    drafter_model.load_state_dict(snapshot, strict=True)
    return {"ok": True}


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.set_defaults(
        dataset_name="hf",
        dataset_path="philschmid/mt-bench",
        num_prompts=80,
    )

    parser.add_argument("--method", choices=["eagle", "eagle3"], default="eagle")
    parser.add_argument(
        "--model-dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-padded-drafter-batch", action="store_true")
    parser.add_argument("--parallel-drafting", action="store_true")
    parser.add_argument("--randomize-seed", type=int, default=2026)
    parser.add_argument("--drop-ratio", type=float, default=0.7)
    parser.add_argument("--recover-ratio", type=float, default=0.9)
    return parser.parse_args()


def _resolve_eagle_model(args) -> str:
    if args.eagle_dir is not None:
        return args.eagle_dir
    return (
        "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        if args.method == "eagle"
        else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    )


def _build_text_prompts(args, tokenizer) -> list[str]:
    samples = get_samples(args, tokenizer)
    prompts: list[str] = []
    for sample in samples:
        if not isinstance(sample.prompt, str):
            raise TypeError(
                "This test expects text prompts. "
                f"Got prompt type: {type(sample.prompt)}."
            )
        prompts.append(sample.prompt)
    return prompts


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = _build_text_prompts(args, tokenizer)

    speculative_config = {
        "method": args.method,
        "model": _resolve_eagle_model(args),
        "num_speculative_tokens": args.num_spec_tokens,
        "disable_padded_drafter_batch": args.disable_padded_drafter_batch,
        "parallel_drafting": args.parallel_drafting,
    }

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
    )

    print("\n=== Step 1: Baseline inference with configured drafter ===")
    baseline = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Baseline acceptance length: {baseline:.4f}")

    print("\n=== Step 2: Snapshot and randomize drafter weights ===")
    llm.collective_rpc(_snapshot_drafter_weights)
    llm.collective_rpc(_randomize_drafter_weights, args=(args.randomize_seed,))
    randomized = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Randomized acceptance length: {randomized:.4f}")

    print("\n=== Step 3: Restore drafter weights from snapshot ===")
    llm.collective_rpc(_restore_drafter_weights)
    restored = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Restored acceptance length: {restored:.4f}")

    drop_ok = randomized <= baseline * args.drop_ratio
    recover_ok = restored >= baseline * args.recover_ratio
    monotonic_ok = restored > randomized

    print("\n=== Result ===")
    print(f"Drop check (<= baseline * {args.drop_ratio:.2f}): {drop_ok}")
    print(f"Recovery check (>= baseline * {args.recover_ratio:.2f}): {recover_ok}")
    print(f"Ordering check (restored > randomized): {monotonic_ok}")

    assert drop_ok and recover_ok and monotonic_ok, (
        "Drafter weight sync acceptance checks failed. "
        f"baseline={baseline:.4f}, randomized={randomized:.4f}, "
        f"restored={restored:.4f}"
    )

    print(
        "Test passed: acceptance dropped after randomization and recovered "
        "after restore."
    )


if __name__ == "__main__":
    main(parse_args())
