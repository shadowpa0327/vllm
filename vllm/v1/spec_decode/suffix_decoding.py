# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer implementing Suffix Decoding
    (arxiv.org/pdf/2411.04975) via Arctic Inference integration.
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config

        # For 'suffix' method: use num_speculative_tokens
        # For 'self_spec_suffix' method: use num_suffix_draft_tokens
        if config.method == "self_spec_suffix":
            self.draft_size = config.num_suffix_draft_tokens
            assert self.draft_size is not None, \
                "num_suffix_draft_tokens should be set by __post_init__"
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info(
                f"[SELF_SPEC_SUFFIX] SuffixDecodingProposer for self_spec_suffix | "
                f"draft_size_per_step={self.draft_size} | "
                f"threshold={config.num_speculative_tokens}")
        else:
            self.draft_size = config.num_speculative_tokens

        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import for optional dependency
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """Generate speculative tokens with dynamic lengths per request."""
        draft_token_ids: list[list[int]] = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            # Skip partial prefills
            if not sampled_ids:
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]

            # Skip unsupported requests
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]

            # Skip max length requests
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]

            # Initialize request if new
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    self.suffix_cache.evict_cached_response(req_id)

                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[
                    index, :num_prompt_tokens
                ]
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Update cache with new tokens
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Extract pattern (limited to max_tree_depth)
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]

            # Query suffix tree for candidates
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.draft_size,
                    self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # Cleanup inactive requests
        for req_id in (
            self.suffix_cache.active_requests
            - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load
        pass
