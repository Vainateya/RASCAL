"""The differentiable KV-cache handoff (the load-bearing mechanism).

Reimplements only LatentMAS's KV-handoff primitive in plain HuggingFace
``transformers`` -- but, unlike LatentMAS (``@torch.no_grad``, vLLM, inference
only), this path is fully differentiable so gradients flow from Bob's loss back
into Alice's parameters.

Two details copied from LatentMAS ``models.py`` (~lines 230-246):

1. ``cache_position = arange(past_len, past_len + new_tokens)`` for Bob's tokens.
2. Left-concat an attention mask of width ``past_len`` (ones over the real prefix)
   in front of Bob's own mask.

Rules honoured here:

- Alice's cache is passed as Bob's ``past_key_values``.
- ``use_cache=True``; **no** ``.generate()``, **no** ``torch.no_grad()``,
  **no** ``.detach()`` on the cache.
- A fresh :class:`DynamicCache` is constructed per Alice forward.
- Teacher-forced cross-entropy on Bob's targets; no autoregressive decode in the
  training path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache


def build_tiny_qwen3(
    vocab_size: int,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 512,
    seed: Optional[int] = None,
) -> Qwen3ForCausalLM:
    """A tiny randomly-initialised Qwen3 for CP0 (same architecture, small)."""
    if seed is not None:
        torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(cfg)


def alice_forward(
    alice: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[DynamicCache, torch.Tensor]:
    """Run Alice, returning her (grad-carrying) KV cache and logits.

    A **fresh** ``DynamicCache`` is created here and populated in-place by the
    forward pass. It is never detached, so it stays attached to Alice's compute
    graph.
    """
    cache = DynamicCache()
    seq_len = input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=input_ids.device)
    out = alice(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return cache, out.logits


def bob_forward_with_prefix(
    bob: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    alice_cache: DynamicCache,
    alice_attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
):
    """Run Bob with Alice's cache as its prefix.

    ``alice_attention_mask`` (width ``past_len``) is left-concatenated in front of
    Bob's own mask, and Bob's ``cache_position`` starts at ``past_len`` -- the two
    LatentMAS details. When Alice is unpadded this prefix mask is all ones, exactly
    as the LatentMAS primitive specifies.
    """
    past_len = alice_attention_mask.shape[1]
    new_tokens = input_ids.shape[1]

    cache_position = torch.arange(
        past_len, past_len + new_tokens, device=input_ids.device
    )
    full_attention_mask = torch.cat([alice_attention_mask, attention_mask], dim=1)

    return bob(
        input_ids=input_ids,
        attention_mask=full_attention_mask,
        past_key_values=alice_cache,
        use_cache=True,
        cache_position=cache_position,
        labels=labels,
    )


def run_handoff(
    alice: Qwen3ForCausalLM,
    bob: Qwen3ForCausalLM,
    batch,
    detach_cache: bool = False,
):
    """Full Alice->Bob handoff on a :class:`~rascal.tokenization.HandoffBatch`.

    Returns Bob's ``CausalLMOutput`` (with ``.loss`` when ``batch.bob_labels`` is
    set). ``detach_cache=True`` is a **negative control**: it severs the cache
    from Alice's graph, which must zero out Alice's gradients -- proving the signal
    path is genuinely the KV handoff and not some other leak.
    """
    cache, _ = alice_forward(
        alice, batch.alice_input_ids, batch.alice_attention_mask
    )

    if detach_cache:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].detach()
            cache.value_cache[i] = cache.value_cache[i].detach()

    return bob_forward_with_prefix(
        bob,
        input_ids=batch.bob_input_ids,
        attention_mask=batch.bob_attention_mask,
        alice_cache=cache,
        alice_attention_mask=batch.alice_attention_mask,
        labels=batch.bob_labels,
    )
