# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group-isolated slot management for compressed CPU offloading."""

from collections import defaultdict
from collections.abc import Collection, Iterable

import numpy as np
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class GroupedCPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Address CPU slots in independent group arenas.

    Args:
        block_ids: Arena-local slot IDs in key order.
        group_indices: KV cache group for every corresponding slot ID.

    Raises:
        ValueError: If the two arrays have different lengths.
    """

    def __init__(
        self,
        block_ids: list[int],
        group_indices: list[int],
    ) -> None:
        if len(block_ids) != len(group_indices):
            raise ValueError("CPU slot and group arrays must have the same length")
        super().__init__(block_ids)
        self.group_indices = np.array(group_indices, dtype=np.int64)


class GroupedCPUOffloadingManager(OffloadingManager):
    """Compose one fixed-slot CPU manager per KV cache group.

    Args:
        num_blocks_per_group: Number of independently addressable slots in
            every group arena.
        cache_policy: Cache policy name passed to each child manager.
        cache_policy_module_path: Optional out-of-tree cache-policy module.
        enable_events: Whether child managers should emit KV events.
        store_threshold: Minimum lookup count before a key is stored.
        max_tracker_size: Maximum admission-frequency tracker size.

    Notes:
        Local slot IDs intentionally overlap across groups. The group index in
        :class:`GroupedCPULoadStoreSpec` disambiguates the physical arena.
    """

    def __init__(
        self,
        num_blocks_per_group: tuple[int, ...],
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ) -> None:
        if not num_blocks_per_group or any(count < 0 for count in num_blocks_per_group):
            raise ValueError("group slot counts must be a non-empty non-negative tuple")
        self.medium = Medium.CPU
        self._managers = tuple(
            CPUOffloadingManager(
                num_blocks=count,
                cache_policy=cache_policy,
                cache_policy_module_path=cache_policy_module_path,
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
            for count in num_blocks_per_group
        )

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """Look up a key in its group's arena.

        Args:
            key: Group-qualified offload key.
            req_context: Per-request offloading context.

        Returns:
            The selected child manager's lookup result.
        """
        return self._manager_for_key(key).lookup(key, req_context)

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """Protect keys and return group-qualified local CPU slots.

        Args:
            keys: Group-qualified keys in transfer order.
            req_context: Per-request offloading context.

        Returns:
            CPU slot IDs and their group indices in the original key order.
        """
        specs = self._prepare_group_loads(keys, req_context)
        return self._merge_specs(keys, specs)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """Touch keys in their respective child policies.

        Args:
            keys: Group-qualified keys.
            req_context: Per-request offloading context.
        """
        for group_idx, group_keys in self._partition_keys(keys).items():
            self._managers[group_idx].touch(group_keys, req_context)

    @override
    def complete_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        """Release load references in each group.

        Args:
            keys: Previously prepared keys.
            req_context: Per-request offloading context.
        """
        for group_idx, group_keys in self._partition_keys(keys).items():
            self._managers[group_idx].complete_load(group_keys, req_context)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """Allocate local slots independently in every selected group.

        Args:
            keys: Group-qualified keys in transfer order.
            req_context: Per-request offloading context.

        Returns:
            A merged group-qualified store allocation, or ``None`` if any
            group cannot allocate its requested objects.
        """
        grouped_keys = self._partition_keys(keys)
        outputs: dict[int, PrepareStoreOutput] = {}
        for group_idx, group_keys in grouped_keys.items():
            output = self._managers[group_idx].prepare_store(
                group_keys,
                req_context,
            )
            if output is None:
                for rollback_idx, prepared in outputs.items():
                    self._managers[rollback_idx].complete_store(
                        prepared.keys_to_store,
                        req_context,
                        success=False,
                    )
                return None
            outputs[group_idx] = output

        keys_to_store_set = {
            key for output in outputs.values() for key in output.keys_to_store
        }
        keys_to_store = [key for key in keys if key in keys_to_store_set]
        specs = {group_idx: output.store_spec for group_idx, output in outputs.items()}
        evicted_keys = [
            key for output in outputs.values() for key in output.evicted_keys
        ]
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=self._merge_specs(keys_to_store, specs),
            evicted_keys=evicted_keys,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """Complete or roll back stores in each group.

        Args:
            keys: Previously prepared group-qualified keys.
            req_context: Per-request offloading context.
            success: Whether the physical transfer succeeded.
        """
        for group_idx, group_keys in self._partition_keys(keys).items():
            self._managers[group_idx].complete_store(
                group_keys,
                req_context,
                success,
            )

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Initialize every child and return their common request policy.

        Args:
            req_context: New request context.

        Returns:
            The request offloading policy shared by all group managers.

        Raises:
            ValueError: If child managers select incompatible policies.
        """
        contexts = tuple(
            manager.on_new_request(req_context) for manager in self._managers
        )
        if any(context.policy != contexts[0].policy for context in contexts[1:]):
            raise ValueError("group CPU managers selected incompatible policies")
        return contexts[0]

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Yield events owned by every group arena.

        Yields:
            Events emitted by the child CPU managers.
        """
        for manager in self._managers:
            yield from manager.take_events()

    @override
    def reset_cache(self) -> None:
        """Clear every group arena."""
        for manager in self._managers:
            manager.reset_cache()

    @override
    def get_stats(self) -> OffloadingConnectorStats:
        """Aggregate metrics emitted by all group managers.

        Returns:
            A combined offloading metric payload.
        """
        stats = OffloadingConnectorStats()
        for manager in self._managers:
            child_stats = manager.get_stats()
            if child_stats is not None:
                stats.aggregate(child_stats)
        return stats

    @override
    def shutdown(self) -> None:
        """Release every child manager."""
        for manager in self._managers:
            manager.shutdown()

    def _manager_for_key(self, key: OffloadKey) -> CPUOffloadingManager:
        group_idx = get_offload_group_idx(key)
        if group_idx >= len(self._managers):
            raise ValueError(f"offload key references unknown group {group_idx}")
        return self._managers[group_idx]

    def _partition_keys(
        self,
        keys: Collection[OffloadKey],
    ) -> dict[int, list[OffloadKey]]:
        grouped: dict[int, list[OffloadKey]] = defaultdict(list)
        for key in keys:
            group_idx = get_offload_group_idx(key)
            if group_idx >= len(self._managers):
                raise ValueError(f"offload key references unknown group {group_idx}")
            grouped[group_idx].append(key)
        return grouped

    def _prepare_group_loads(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> dict[int, LoadStoreSpec]:
        return {
            group_idx: self._managers[group_idx].prepare_load(
                group_keys,
                req_context,
            )
            for group_idx, group_keys in self._partition_keys(keys).items()
        }

    def _merge_specs(
        self,
        keys: Collection[OffloadKey],
        specs: dict[int, LoadStoreSpec],
    ) -> GroupedCPULoadStoreSpec:
        cursors: dict[int, int] = defaultdict(int)
        block_ids: list[int] = []
        group_indices: list[int] = []
        for key in keys:
            group_idx = get_offload_group_idx(key)
            spec = specs[group_idx]
            if not isinstance(spec, CPULoadStoreSpec):
                raise TypeError("group CPU manager returned an incompatible spec")
            cursor = cursors[group_idx]
            block_ids.append(int(spec.block_ids[cursor]))
            group_indices.append(group_idx)
            cursors[group_idx] += 1
        return GroupedCPULoadStoreSpec(block_ids, group_indices)
