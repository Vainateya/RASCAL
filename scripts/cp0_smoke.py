"""CP0 -- differentiable-handoff smoke test.

Go/no-go gate for the whole project: pass Alice's KV cache into Bob, compute a
loss on Bob, ``.backward()``, and assert gradients land on **Alice's** params.
If this fails, nothing downstream can work.

What it does:
  1. Build the arithmetic cover task (SplitArithmeticStream).
  2. Wrap it with the steganographic signal scheme (SteganographicStream).
  3. Tokenise + collate one batch (CharTokenizer).
  4. Build two tiny Qwen3 instances (plain trainable params -- LoRA comes at CP1).
  5. Run the Alice->Bob handoff, teacher-forced CE on Bob, backward.
  6. PASS/FAIL checks:
       (a) Alice's KV cache carries grad_fn (attached to her graph).
       (b) Alice's params receive non-zero gradients through the handoff.
       (c) Negative control: detaching the cache zeroes Alice's gradients.

Run:  python scripts/cp0_smoke.py
"""

from __future__ import annotations

import os
import sys
import itertools

import torch

# Make ``src/`` importable without installation.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

from rascal.data import SplitArithmeticStream, SteganographicStream, class_name
from rascal.tokenization import CharTokenizer, collate_handoff
from rascal.models import build_tiny_qwen3, run_handoff


BATCH_SIZE = 8
SEED = 0


def _grad_magnitude(model) -> float:
    return sum(
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )


def _num_params_with_grad(model) -> int:
    return sum(1 for p in model.parameters() if p.grad is not None)


def main() -> int:
    torch.manual_seed(SEED)

    # 1-2. Cover task -> steganographic wrapper.
    base = SplitArithmeticStream(min_digits=3, max_digits=4, seed=SEED)
    steg = SteganographicStream(base, seed=SEED)

    # 3. One batch.
    tokenizer = CharTokenizer()
    records = list(itertools.islice(steg, BATCH_SIZE))
    batch = collate_handoff(records, tokenizer)

    print("=" * 68)
    print("CP0 -- differentiable KV-handoff smoke test")
    print("=" * 68)
    print(f"vocab_size            : {tokenizer.vocab_size}")
    print(f"batch size            : {BATCH_SIZE}")
    print(f"alice_input_ids       : {tuple(batch.alice_input_ids.shape)}")
    print(f"bob_input_ids         : {tuple(batch.bob_input_ids.shape)}")
    print("sample records:")
    for r in records[:4]:
        print(
            f"   [{class_name(r['signal_class']):8s}] "
            f"alice x={r['alice_text']:>4s} (+{r['signal_marker']})  "
            f"bob y={r['bob_text']:>4s}  ->  target={r['target_text']}"
        )
    # Class-independence sanity: signal must not be inferable from x,y alone.
    print("-" * 68)

    # 4. Two tiny models, shared architecture, independent params. Plain trainable.
    alice = build_tiny_qwen3(tokenizer.vocab_size, seed=SEED)
    bob = build_tiny_qwen3(tokenizer.vocab_size, seed=SEED + 1)
    alice.train()
    bob.train()

    # 5. Handoff + teacher-forced CE on Bob.
    out = run_handoff(alice, bob, batch, detach_cache=False)
    loss = out.loss
    loss.backward()

    alice_grad = _grad_magnitude(alice)
    bob_grad = _grad_magnitude(bob)

    print(f"Bob teacher-forced loss           : {loss.item():.4f}")
    print(
        f"Alice params receiving grad       : "
        f"{_num_params_with_grad(alice)} / {sum(1 for _ in alice.parameters())}"
    )
    print(f"Alice total grad magnitude        : {alice_grad:.4f}")
    print(f"Bob   total grad magnitude        : {bob_grad:.4f}")

    # 6a. Cache carries grad.
    cache_check, _ = _cache_requires_grad(alice, bob, batch)

    # 6c. Negative control -- detach the cache, Alice must get NO gradient.
    alice.zero_grad(set_to_none=True)
    bob.zero_grad(set_to_none=True)
    out_ctrl = run_handoff(alice, bob, batch, detach_cache=True)
    out_ctrl.loss.backward()
    alice_grad_detached = _grad_magnitude(alice)
    print("-" * 68)
    print(f"[control] Alice grad w/ detached cache: {alice_grad_detached:.6f}")

    # ---- Verdict -------------------------------------------------------------
    checks = {
        "Alice cache carries grad_fn": cache_check,
        "Alice receives gradient through handoff": alice_grad > 0,
        "Bob receives gradient": bob_grad > 0,
        "Loss is finite": torch.isfinite(loss).item(),
        "Detached-cache control zeroes Alice grad": alice_grad_detached == 0.0,
    }
    print("=" * 68)
    all_pass = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok
    print("=" * 68)
    print("CP0 RESULT:", "PASS -- proceed to CP1" if all_pass else "FAIL")
    return 0 if all_pass else 1


def _cache_requires_grad(alice, bob, batch):
    """Independently re-run Alice to inspect the cache tensors' grad state."""
    from rascal.models.handoff import alice_forward

    cache, _ = alice_forward(
        alice, batch.alice_input_ids, batch.alice_attention_mask
    )
    k0 = cache.key_cache[0]
    return (k0.requires_grad and k0.grad_fn is not None), cache


if __name__ == "__main__":
    raise SystemExit(main())
