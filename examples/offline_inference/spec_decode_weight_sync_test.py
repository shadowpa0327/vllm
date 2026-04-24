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


def _inspect_drafter_sharing(worker):
    """Report whether drafter shares embed_tokens / lm_head with the target model."""
    from vllm.model_executor.models.interfaces import supports_multimodal

    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    target_model = worker.model_runner.model

    if drafter_model is None:
        return {"ok": False, "reason": "no drafter / drafter has no .model"}

    if supports_multimodal(target_model):
        target_lm = target_model.get_language_model()
    else:
        target_lm = target_model

    # Resolve inner target embed_tokens (same logic as _maybe_share_embeddings).
    target_inner = getattr(target_lm, "model", None)
    target_embed = None
    if target_inner is not None:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
    target_lm_head = getattr(target_lm, "lm_head", None)

    drafter_inner = getattr(drafter_model, "model", None)
    drafter_embed = (
        getattr(drafter_inner, "embed_tokens", None)
        if drafter_inner is not None
        else None
    )
    drafter_lm_head = getattr(drafter_model, "lm_head", None)

    def _ptr(mod):
        if mod is None:
            return None
        w = getattr(mod, "weight", None)
        return None if w is None else w.data_ptr()

    report = {
        "has_own_embed_tokens": getattr(drafter_model, "has_own_embed_tokens", "N/A"),
        "has_own_lm_head": getattr(drafter_model, "has_own_lm_head", "N/A"),
        "embed_module_is_same": drafter_embed is target_embed,
        "embed_weight_ptr_equal": (
            _ptr(drafter_embed) is not None
            and _ptr(drafter_embed) == _ptr(target_embed)
        ),
        "lm_head_module_is_same": drafter_lm_head is target_lm_head,
        "lm_head_weight_ptr_equal": (
            _ptr(drafter_lm_head) is not None
            and _ptr(drafter_lm_head) == _ptr(target_lm_head)
        ),
        "drafter_embed_shape": (
            tuple(drafter_embed.weight.shape) if drafter_embed is not None else None
        ),
        "target_embed_shape": (
            tuple(target_embed.weight.shape) if target_embed is not None else None
        ),
        "drafter_lm_head_shape": (
            tuple(drafter_lm_head.weight.shape) if drafter_lm_head is not None else None
        ),
        "target_lm_head_shape": (
            tuple(target_lm_head.weight.shape) if target_lm_head is not None else None
        ),
        "drafter_class": type(drafter_model).__name__,
        "rank": worker.rank,
    }

    # MTP-specific: each draft layer has shared_head.head.
    mtp_shared_heads = []
    if drafter_inner is not None:
        layers = getattr(drafter_inner, "layers", None)
        if layers is not None:
            iterable = layers.values() if hasattr(layers, "values") else layers
            for i, layer in enumerate(iterable):
                sh = getattr(layer, "shared_head", None)
                head = getattr(sh, "head", None) if sh is not None else None
                if head is not None:
                    mtp_shared_heads.append(
                        {
                            "layer": i,
                            "is_target_lm_head": head is target_lm_head,
                            "weight_ptr_equal": _ptr(head) == _ptr(target_lm_head),
                        }
                    )
    report["mtp_shared_heads"] = mtp_shared_heads

    return report


def _probe_target_param_norms(worker):
    """Return L2 norms of the TARGET model's embed_tokens and lm_head weights.

    Useful for detecting whether an update to the drafter has written through
    to the target via shared modules (embed_tokens and/or lm_head).
    """
    from vllm.model_executor.models.interfaces import supports_multimodal

    target = worker.model_runner.model
    if supports_multimodal(target):
        target = target.get_language_model()

    inner = getattr(target, "model", None)
    embed = getattr(inner, "embed_tokens", None) if inner is not None else None
    lm_head = getattr(target, "lm_head", None)

    def _norm(mod):
        if mod is None:
            return None
        w = getattr(mod, "weight", None)
        if w is None:
            return None
        return float(w.detach().float().norm().item())

    return {
        "target_embed_norm": _norm(embed),
        "target_lm_head_norm": _norm(lm_head),
        "rank": worker.rank,
    }


def _snapshot_drafter_weights(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    if drafter_model is None:
        raise RuntimeError("No drafter model found on this worker.")

    worker._weight_sync_test_drafter_snapshot = {
        name: tensor.detach().cpu().clone()
        for name, tensor in drafter_model.state_dict().items()
    }
    # print the name of all tensors
    for name in drafter_model.state_dict().keys():
        print(name)

    return {
        "ok": True,
        "num_tensors": len(worker._weight_sync_test_drafter_snapshot),
    }


def _randomize_drafter_weights(worker, seed: int):
    """Randomize drafter params, skipping anything shared with the main model.

    Parameters are identified by their tensor data_ptr(). Any drafter
    parameter whose storage coincides with a target parameter is left alone —
    that storage belongs to the main model, and overwriting it would silently
    corrupt production inference.
    """
    from vllm.model_executor.models.interfaces import supports_multimodal

    drafter = getattr(worker.model_runner, "drafter", None)
    drafter_model = getattr(drafter, "model", None) if drafter is not None else None
    if drafter_model is None:
        raise RuntimeError("No drafter model found on this worker.")

    target = worker.model_runner.model
    if supports_multimodal(target):
        target = target.get_language_model()
    target_ptrs = {
        p.data_ptr() for p in target.parameters() if p.is_floating_point()
    }

    torch.manual_seed(seed + worker.rank)
    skipped_non_float = []
    skipped_shared = []
    randomized = []
    with torch.no_grad():
        for name, param in drafter_model.named_parameters():
            if not param.is_floating_point():
                skipped_non_float.append((name, str(param.dtype)))
                continue
            if param.data_ptr() in target_ptrs:
                skipped_shared.append(name)
                continue
            param.copy_(torch.randn_like(param))
            randomized.append(name)
    return {
        "ok": True,
        "num_randomized": len(randomized),
        "skipped_non_float": skipped_non_float,
        "skipped_shared_with_target": skipped_shared,
    }


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

    print("\n=== Step 0: Inspect drafter/target weight sharing ===")
    sharing_reports = llm.collective_rpc(_inspect_drafter_sharing)
    for report in sharing_reports:
        print(f"[rank {report.get('rank')}] drafter={report.get('drafter_class')}")
        print(
            f"  has_own_embed_tokens={report.get('has_own_embed_tokens')}  "
            f"has_own_lm_head={report.get('has_own_lm_head')}"
        )
        print(
            f"  embed_tokens:  module_is={report.get('embed_module_is_same')}  "
            f"weight_ptr_eq={report.get('embed_weight_ptr_equal')}  "
            f"drafter_shape={report.get('drafter_embed_shape')}  "
            f"target_shape={report.get('target_embed_shape')}"
        )
        print(
            f"  lm_head:       module_is={report.get('lm_head_module_is_same')}  "
            f"weight_ptr_eq={report.get('lm_head_weight_ptr_equal')}  "
            f"drafter_shape={report.get('drafter_lm_head_shape')}  "
            f"target_shape={report.get('target_lm_head_shape')}"
        )
        if report.get("mtp_shared_heads"):
            print(f"  mtp_shared_heads: {report['mtp_shared_heads']}")

    sample_prompt = prompts[:1]
    sample_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=48
    )

    def _phase_report(tag: str):
        norms = llm.collective_rpc(_probe_target_param_norms)
        for n in norms:
            print(
                f"  [rank {n['rank']}] target_embed_norm={n['target_embed_norm']:.4f}  "
                f"target_lm_head_norm={n['target_lm_head_norm']:.4f}"
            )
        out = llm.generate(sample_prompt, sampling_params=sample_params)
        text = out[0].outputs[0].text.replace("\n", " ")
        print(f"  [{tag}] sample output: {text[:200]!r}")

    print("\n=== Step 1: Baseline inference with configured drafter ===")
    baseline = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Baseline acceptance length: {baseline:.4f}")
    _phase_report("baseline")

    print("\n=== Step 2: Snapshot and randomize drafter weights ===")
    llm.collective_rpc(_snapshot_drafter_weights)
    randomize_reports = llm.collective_rpc(
        _randomize_drafter_weights, args=(args.randomize_seed,)
    )
    for r in randomize_reports:
        print(
            f"  [randomize] randomized={r['num_randomized']}  "
            f"skipped_shared={r['skipped_shared_with_target']}  "
            f"skipped_non_float={[n for n, _ in r['skipped_non_float']]}"
        )
    randomized = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Randomized acceptance length: {randomized:.4f}")
    _phase_report("randomized")

    print("\n=== Step 3: Restore drafter weights from snapshot ===")
    llm.collective_rpc(_restore_drafter_weights)
    restored = _run_and_measure_acceptance(llm, prompts, sampling_params)
    print(f"Restored acceptance length: {restored:.4f}")
    _phase_report("restored")

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
