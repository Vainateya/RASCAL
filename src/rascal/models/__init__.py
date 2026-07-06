"""Model wiring for RASCAL: the differentiable KV handoff primitive."""

from .handoff import (
    build_tiny_qwen3,
    alice_forward,
    bob_forward_with_prefix,
    run_handoff,
)

__all__ = [
    "build_tiny_qwen3",
    "alice_forward",
    "bob_forward_with_prefix",
    "run_handoff",
]
