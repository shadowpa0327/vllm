# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for checking module availability."""

import importlib.util


def _has_module(module_name: str) -> bool:
    """Check if a module is available."""
    return importlib.util.find_spec(module_name) is not None


def has_arctic_inference() -> bool:
    """Check optional arctic_inference package availability."""
    return _has_module("arctic_inference")

