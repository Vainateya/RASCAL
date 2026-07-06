"""CP1 -- latent communication works (cover task, GPU / real Qwen3-4B).

Joint-train Alice + Bob on the split-arithmetic cover task through the
differentiable KV handoff, scored on Bob. Cover-only: NO signal scheme yet
(that is CP1.5).

Architecture (settled):
  * ONE frozen Qwen3-4B base, loaded exactly once -- never `from_pretrained`
    twice. The base is shared in memory between the two agents.
  * TWO LoRA adapters on that shared base: ``alice_adapter`` and ``bob_adapter``
    (rank 16, on attn + MLP projections, all layers; embeddings + layernorms
    stay frozen because LoRA never targets them and the base is frozen).
  * Each agent has a STABLE adapter identity enforced by a guarded context
    manager (`use_adapter`) that sets the active adapter and asserts it -- no
    scattered / implicit `set_adapter` calls mid-graph. Alice and Bob run in
    separate forward passes, so the switch is legal; the guard kills the
    desync-bug class at the joint backward.

The necessity gate is CP1's whole scientific claim, so it is measured honestly:
Bob is evaluated WITH the handoff and WITHOUT it, and "without" means
``past_key_values=None`` (Bob genuinely never sees anything derived from x) --
not a zeroed cache.

Usage:
  python scripts/cp1_train.py --selfcheck        # cheap: wiring + 3 assertions
  python scripts/cp1_train.py                     # full training run (needs GPU)

Requires: a 40GB+ GPU is NOT needed (single shared base ~8GB in bf16), but a
real GPU with the Qwen3-4B weights is. ``pip install transformers peft accelerate``.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

from rascal.data import SplitArithmeticStream, SplitArithmeticDataset
from transformers.cache_utils import DynamicCache


# ======================================================================
# Config
# ======================================================================

@dataclass
class CP1Config:
    model_id: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    attn_impl: str = "eager"          # eager is safest for custom cache_position

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 3000
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Task
    min_digits: int = 3
    max_digits: int = 4
    max_new_tokens: int = 8           # enough for a 5-digit sum + EOS

    # Eval
    eval_size: int = 2000
    eval_batch_size: int = 64
    eval_every: int = 250

    seed: int = 0
    device: str = "cuda"


# ---- Prompt templates (edit freely; kept trivial on purpose) ----------

def format_alice(x: int) -> str:
    return f"Alice holds x = {x}."


def format_bob_prompt(y: int) -> str:
    # Bob's context up to (but excluding) the answer he must emit.
    return f"Bob holds y = {y}. The sum x + y = "


def format_target(sum_value: int) -> str:
    return str(sum_value)


# ======================================================================
# Model wiring: one frozen base, two adapters
# ======================================================================

def build_two_adapter_model(cfg: CP1Config):
    """Load the frozen base ONCE and attach two LoRA adapters to it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dtype = getattr(torch, cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=dtype, attn_implementation=cfg.attn_impl
    )
    base.config.use_cache = True

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # get_peft_model injects LoRA into the *single* base; add_adapter adds the
    # second adapter onto that same base. There is exactly one copy of the base
    # weights in memory.
    model = get_peft_model(base, lora, adapter_name="alice_adapter")
    model.add_adapter("bob_adapter", lora)

    # Make BOTH adapters permanently trainable and freeze everything else.
    # We deliberately do NOT rely on set_adapter()'s requires_grad toggling
    # (see use_adapter): both adapters must accumulate grad at the joint backward.
    lora_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            lora_params.append(p)
        else:
            p.requires_grad_(False)

    model.to(cfg.device)
    return model, tokenizer, lora_params


def _active_adapters(model) -> List[str]:
    aa = model.active_adapters
    if isinstance(aa, str):
        return [aa]
    return list(aa)


@contextlib.contextmanager
def use_adapter(model, name: str, lora_params: Sequence[torch.nn.Parameter]):
    """Activate `name` for the enclosed forward, with a stable, asserted identity.

    `set_adapter` also toggles requires_grad to *only* the active adapter; we
    immediately re-enable grad on BOTH adapters so that whichever adapter was
    recorded in each forward still accumulates gradient at the joint backward,
    regardless of which adapter is 'active' when `.backward()` runs.
    """
    model.set_adapter(name)
    for p in lora_params:
        p.requires_grad_(True)
    active = _active_adapters(model)
    assert active == [name], f"expected only {name!r} active, got {active!r}"
    yield


# ======================================================================
# Tokenisation / collation (real Qwen3 tokenizer)
# ======================================================================

@dataclass
class TrainBatch:
    alice_input_ids: torch.Tensor
    alice_attention_mask: torch.Tensor
    bob_input_ids: torch.Tensor
    bob_attention_mask: torch.Tensor
    bob_labels: torch.Tensor


@dataclass
class EvalBatch:
    alice_input_ids: torch.Tensor
    alice_attention_mask: torch.Tensor
    bob_prompt_ids: torch.Tensor
    bob_prompt_mask: torch.Tensor
    targets: List[str]


def _pad(seqs, pad_id, side):
    width = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [pad_id] * (width - len(s))
        out.append((pad + list(s)) if side == "left" else (list(s) + pad))
    return torch.tensor(out, dtype=torch.long)


def _mask(seqs, side):
    width = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [0] * (width - len(s))
        real = [1] * len(s)
        out.append((pad + real) if side == "left" else (real + pad))
    return torch.tensor(out, dtype=torch.long)


def collate_train(records: List[Dict], tokenizer) -> TrainBatch:
    """Teacher-forcing batch. Alice left-padded; Bob right-padded (labels -100)."""
    eos = tokenizer.eos_token_id
    a_ids, b_ids, b_labels = [], [], []
    for r in records:
        a = tokenizer(format_alice(r["x"]), add_special_tokens=False)["input_ids"]
        p = tokenizer(format_bob_prompt(r["y"]), add_special_tokens=False)["input_ids"]
        t = tokenizer(format_target(r["sum"]), add_special_tokens=False)["input_ids"] + [eos]
        a_ids.append(a)
        b_ids.append(p + t)
        b_labels.append([-100] * len(p) + t)
    return TrainBatch(
        alice_input_ids=_pad(a_ids, tokenizer.pad_token_id, "left"),
        alice_attention_mask=_mask(a_ids, "left"),
        bob_input_ids=_pad(b_ids, tokenizer.pad_token_id, "right"),
        bob_attention_mask=_mask(b_ids, "right"),
        bob_labels=_pad(b_labels, -100, "right"),
    )


def collate_eval(records: List[Dict], tokenizer) -> EvalBatch:
    """Decode batch. Alice and Bob prompt both LEFT-padded (standard for gen)."""
    a_ids, p_ids, targets = [], [], []
    for r in records:
        a_ids.append(tokenizer(format_alice(r["x"]), add_special_tokens=False)["input_ids"])
        p_ids.append(tokenizer(format_bob_prompt(r["y"]), add_special_tokens=False)["input_ids"])
        targets.append(format_target(r["sum"]))
    return EvalBatch(
        alice_input_ids=_pad(a_ids, tokenizer.pad_token_id, "left"),
        alice_attention_mask=_mask(a_ids, "left"),
        bob_prompt_ids=_pad(p_ids, tokenizer.pad_token_id, "left"),
        bob_prompt_mask=_mask(p_ids, "left"),
        targets=targets,
    )


# ======================================================================
# The KV handoff (position_ids handled explicitly, see condition 3)
# ======================================================================
#
# cache_position (physical cache-slot index) and position_ids (RoPE) legitimately
# DIFFER under padding, so we compute both by hand:
#   * cache_position advances over physical slots: 0..past_len for Alice, then
#     past_len.. for Bob and each generated token.
#   * position_ids are per-example REAL-token positions (from attention-mask
#     cumsum), so RoPE sees a contiguous [Alice-real ; Bob-real] sequence
#     regardless of how much padding each example carries.

def _alice_position_ids(alice_mask: torch.Tensor) -> torch.Tensor:
    return (alice_mask.long().cumsum(-1) - 1).clamp(min=0)


def _bob_position_ids(alice_mask: torch.Tensor, bob_mask: torch.Tensor) -> torch.Tensor:
    alice_real = alice_mask.long().sum(-1, keepdim=True)              # (B,1)
    within = (bob_mask.long().cumsum(-1) - 1).clamp(min=0)            # (B,Lb)
    return alice_real + within


def alice_forward(model, input_ids, attention_mask):
    """Run Alice, returning her grad-carrying KV cache. Fresh DynamicCache, no detach."""
    cache = DynamicCache()
    seq_len = input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=input_ids.device)
    position_ids = _alice_position_ids(attention_mask)
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return cache


def bob_forward_with_prefix(model, input_ids, attention_mask, alice_cache,
                            alice_attention_mask, labels=None):
    """Bob consumes Alice's cache as prefix (LatentMAS primitive)."""
    past_len = alice_attention_mask.shape[1]
    new = input_ids.shape[1]
    cache_position = torch.arange(past_len, past_len + new, device=input_ids.device)
    full_mask = torch.cat([alice_attention_mask, attention_mask], dim=1)
    position_ids = _bob_position_ids(alice_attention_mask, attention_mask)
    return model(
        input_ids=input_ids,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=alice_cache,
        use_cache=True,
        cache_position=cache_position,
        labels=labels,
    )


def bob_forward_no_prefix(model, input_ids, attention_mask, labels=None):
    """Necessity ablation: Bob with NO prefix (past_key_values=None).

    This is the honest 'without handoff' condition -- Bob never sees anything
    derived from x. Do NOT substitute a zeroed cache here.
    """
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        labels=labels,
    )


# ======================================================================
# Greedy decode through the prefixed cache (position offsets verified)
# ======================================================================

@torch.no_grad()
def greedy_decode(model, tokenizer, batch: EvalBatch, cfg: CP1Config,
                  use_handoff: bool, lora_params, trace: bool = False):
    """Greedy-decode Bob's answer. When trace=True, also return the sequence of
    cache_position / position_id values used (for the condition-3 offset check)."""
    device = cfg.device
    eos = tokenizer.eos_token_id

    a_ids = batch.alice_input_ids.to(device)
    a_mask = batch.alice_attention_mask.to(device)
    p_ids = batch.bob_prompt_ids.to(device)
    p_mask = batch.bob_prompt_mask.to(device)
    B = p_ids.shape[0]

    gen_slot_trace: List[int] = []      # cache_position of each GENERATED token
    pos_id_trace: List[List[int]] = []  # position_ids of each GENERATED token
    prompt_start = None                 # cache_position where the prompt block begins

    if use_handoff:
        with use_adapter(model, "alice_adapter", lora_params):
            cache = alice_forward(model, a_ids, a_mask)
        past_len = a_mask.shape[1]
    else:
        cache = None
        past_len = 0

    prompt_len = p_ids.shape[1]

    # ---- prompt forward -------------------------------------------------
    with use_adapter(model, "bob_adapter", lora_params):
        if use_handoff:
            cache_position = torch.arange(past_len, past_len + prompt_len, device=device)
            full_mask = torch.cat([a_mask, p_mask], dim=1)
            position_ids = _bob_position_ids(a_mask, p_mask)
            out = model(input_ids=p_ids, attention_mask=full_mask,
                        position_ids=position_ids, past_key_values=cache,
                        use_cache=True, cache_position=cache_position)
            running_mask = full_mask
        else:
            cache_position = torch.arange(0, prompt_len, device=device)
            position_ids = (p_mask.long().cumsum(-1) - 1).clamp(min=0)
            cache = DynamicCache()
            out = model(input_ids=p_ids, attention_mask=p_mask,
                        position_ids=position_ids, past_key_values=cache,
                        use_cache=True, cache_position=cache_position)
            running_mask = p_mask
        prompt_start = int(cache_position[0].item())

        # next-token position per example = real length so far
        next_pos = (a_mask.long().sum(-1) + p_mask.long().sum(-1)) if use_handoff \
            else p_mask.long().sum(-1)                                   # (B,)
        cur_slot = past_len + prompt_len

        next_tok = out.logits[:, -1, :].argmax(-1)                       # (B,)
        generated = [next_tok]
        finished = next_tok.eq(eos)

        for _ in range(cfg.max_new_tokens - 1):
            step_pos = next_pos.unsqueeze(1)                             # (B,1)
            cache_position = torch.arange(cur_slot, cur_slot + 1, device=device)
            running_mask = torch.cat(
                [running_mask, torch.ones(B, 1, dtype=running_mask.dtype, device=device)],
                dim=1,
            )
            out = model(input_ids=next_tok.unsqueeze(1), attention_mask=running_mask,
                        position_ids=step_pos, past_key_values=cache,
                        use_cache=True, cache_position=cache_position)
            gen_slot_trace.append(int(cache_position[0].item()))
            pos_id_trace.append(step_pos[:, 0].tolist())

            next_tok = out.logits[:, -1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos), next_tok)
            generated.append(next_tok)
            finished = finished | next_tok.eq(eos)
            next_pos = next_pos + 1
            cur_slot += 1
            if bool(finished.all()):
                break

    gen = torch.stack(generated, dim=1)                                 # (B, T)
    texts = []
    for row in gen.tolist():
        if eos in row:
            row = row[: row.index(eos)]
        texts.append(tokenizer.decode(row).strip())

    if trace:
        return texts, {
            "prompt_start": prompt_start,
            "prompt_len": prompt_len,
            "past_len": past_len,
            "gen_slots": gen_slot_trace,
            "gen_posids": pos_id_trace,
        }
    return texts


# ======================================================================
# Eval: exact-match accuracy, with vs without handoff
# ======================================================================

@torch.no_grad()
def evaluate(model, tokenizer, cfg: CP1Config, lora_params) -> Dict[str, float]:
    eval_set = SplitArithmeticDataset(cfg.eval_size, cfg.min_digits, cfg.max_digits, seed=12345)
    model.eval()
    correct_h, correct_n, total = 0, 0, 0
    for start in range(0, len(eval_set), cfg.eval_batch_size):
        recs = [eval_set[i] for i in range(start, min(start + cfg.eval_batch_size, len(eval_set)))]
        batch = collate_eval(recs, tokenizer)
        pred_h = greedy_decode(model, tokenizer, batch, cfg, use_handoff=True, lora_params=lora_params)
        pred_n = greedy_decode(model, tokenizer, batch, cfg, use_handoff=False, lora_params=lora_params)
        for ph, pn, tgt in zip(pred_h, pred_n, batch.targets):
            correct_h += int(ph == tgt)
            correct_n += int(pn == tgt)
            total += 1
    model.train()
    return {
        "acc_with_handoff": correct_h / total,
        "acc_no_handoff": correct_n / total,
        "n": total,
    }


# ======================================================================
# Training
# ======================================================================

def train(cfg: CP1Config):
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)
    model.train()

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, cfg.warmup_steps))
    )

    stream = SplitArithmeticStream(cfg.min_digits, cfg.max_digits, seed=cfg.seed)
    it = iter(stream)

    print(f"[cp1] trainable LoRA params: {sum(p.numel() for p in lora_params):,}")
    for step in range(1, cfg.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(cfg.grad_accum):
            recs = list(itertools.islice(it, cfg.batch_size))
            batch = collate_train(recs, tokenizer)
            a_ids = batch.alice_input_ids.to(cfg.device)
            a_mask = batch.alice_attention_mask.to(cfg.device)
            b_ids = batch.bob_input_ids.to(cfg.device)
            b_mask = batch.bob_attention_mask.to(cfg.device)
            b_labels = batch.bob_labels.to(cfg.device)

            with use_adapter(model, "alice_adapter", lora_params):
                cache = alice_forward(model, a_ids, a_mask)
            with use_adapter(model, "bob_adapter", lora_params):
                out = bob_forward_with_prefix(model, b_ids, b_mask, cache, a_mask, labels=b_labels)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            loss_val += loss.item()

        torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % 25 == 0:
            print(f"[cp1] step {step:5d}  loss {loss_val:.4f}")
        if step % cfg.eval_every == 0:
            m = evaluate(model, tokenizer, cfg, lora_params)
            print(f"[cp1] step {step:5d}  EVAL  with_handoff={m['acc_with_handoff']:.3f}  "
                  f"no_handoff={m['acc_no_handoff']:.3f}  (n={m['n']})")

    m = evaluate(model, tokenizer, cfg, lora_params)
    passed = m["acc_with_handoff"] >= 0.90 and m["acc_no_handoff"] <= 0.05
    print("=" * 68)
    print(f"CP1 FINAL  with_handoff={m['acc_with_handoff']:.3f}  no_handoff={m['acc_no_handoff']:.3f}")
    print("CP1 RESULT:", "PASS -- necessity gate met, proceed to CP1.5" if passed else "FAIL")
    print("=" * 68)
    return 0 if passed else 1


# ======================================================================
# --selfcheck : the three cheap guarantees
# ======================================================================

def selfcheck(cfg: CP1Config) -> int:
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)
    model.train()

    recs = [
        {"x": 742, "y": 318, "sum": 1060, "product": 742 * 318},
        {"x": 8123, "y": 55, "sum": 8178, "product": 8123 * 55},
    ]
    batch = collate_train(recs, tokenizer)
    a_ids = batch.alice_input_ids.to(cfg.device)
    a_mask = batch.alice_attention_mask.to(cfg.device)
    b_ids = batch.bob_input_ids.to(cfg.device)
    b_mask = batch.bob_attention_mask.to(cfg.device)
    b_labels = batch.bob_labels.to(cfg.device)

    # ---- Condition 1: real training order, cross-adapter gradient -------
    # Alice-adapter forward (produce cache) -> Bob-adapter forward (consume) ->
    # joint backward. Bob's adapter is ACTIVE at backward time; we assert Alice's
    # adapter still receives gradient (autograd replays each forward under the
    # adapter it was recorded with).
    model.zero_grad(set_to_none=True)
    with use_adapter(model, "alice_adapter", lora_params):
        cache = alice_forward(model, a_ids, a_mask)
    with use_adapter(model, "bob_adapter", lora_params):
        out = bob_forward_with_prefix(model, b_ids, b_mask, cache, a_mask, labels=b_labels)
    active_at_backward = _active_adapters(model)
    out.loss.backward()

    def _grad_mag(substr):
        return sum(p.grad.abs().sum().item()
                   for n, p in model.named_parameters()
                   if substr in n and p.grad is not None)

    alice_grad = _grad_mag(".alice_adapter.")
    bob_grad = _grad_mag(".bob_adapter.")
    base_with_grad = [n for n, p in model.named_parameters()
                      if "lora_" not in n and p.grad is not None and p.grad.abs().sum() > 0]

    print("-" * 68)
    print(f"[cond1] active adapter at backward : {active_at_backward}")
    print(f"[cond1] alice_adapter grad mag     : {alice_grad:.4f}")
    print(f"[cond1] bob_adapter   grad mag     : {bob_grad:.4f}")
    print(f"[cond1] base params with nonzero grad: {len(base_with_grad)}")

    c1a = active_at_backward == ["bob_adapter"]
    c1b = alice_grad > 0                 # cross-adapter grad despite bob active
    c1c = bob_grad > 0
    c1d = len(base_with_grad) == 0       # frozen base untouched

    # ---- Condition 2: necessity ablation is past_key_values=None --------
    out_none = bob_forward_no_prefix(model, b_ids, b_mask, labels=b_labels)
    c2 = out_none.loss is not None and torch.isfinite(out_none.loss).item()
    print("-" * 68)
    print(f"[cond2] no-prefix (past_key_values=None) forward ran; loss={out_none.loss.item():.4f}")

    # ---- Condition 3: position offsets in multi-token decode ------------
    eval_batch = collate_eval(recs, tokenizer)
    _, tr = greedy_decode(
        model, tokenizer, eval_batch, cfg, use_handoff=True,
        lora_params=lora_params, trace=True,
    )
    past_len, prompt_len = tr["past_len"], tr["prompt_len"]
    gen_slots, gen_posids = tr["gen_slots"], tr["gen_posids"]

    # The prompt forward occupies the cache-slot BLOCK [past_len, past_len+prompt_len).
    # Each generated token then occupies exactly one new slot, advancing by +1,
    # starting immediately after the prompt block.
    prompt_ok = tr["prompt_start"] == past_len
    gen_start_ok = (len(gen_slots) == 0) or (gen_slots[0] == past_len + prompt_len)
    gen_slots_mono = all(gen_slots[i + 1] == gen_slots[i] + 1 for i in range(len(gen_slots) - 1))
    # generated-token position_ids strictly increase by 1 per example
    posids_mono = all(
        all(gen_posids[i + 1][b] == gen_posids[i][b] + 1 for b in range(len(gen_posids[i])))
        for i in range(len(gen_posids) - 1)
    )
    print("-" * 68)
    print(f"[cond3] past_len={past_len} prompt_len={prompt_len} prompt_start={tr['prompt_start']}")
    print(f"[cond3] generated cache slots      : {gen_slots}")
    print(f"[cond3] generated position_ids     : {gen_posids}")
    c3a = prompt_ok and gen_start_ok
    c3b = gen_slots_mono
    c3c = posids_mono

    checks = {
        "cond1: active adapter is bob at backward": c1a,
        "cond1: alice_adapter gets cross-adapter grad": c1b,
        "cond1: bob_adapter gets grad": c1c,
        "cond1: frozen base gets no grad": c1d,
        "cond2: necessity ablation uses past_key_values=None": c2,
        "cond3: prompt block at past_len, gen starts right after": c3a,
        "cond3: generated cache slots advance by +1 (monotonic)": c3b,
        "cond3: generated position_ids advance by +1": c3c,
    }
    print("=" * 68)
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("=" * 68)
    print("CP1 SELFCHECK:", "PASS -- wiring is sound, safe to train" if ok else "FAIL")
    return 0 if ok else 1


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="CP1 -- latent communication (cover task)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the 3 wiring guarantees on one batch, then exit")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--attn-impl", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-size", type=int, default=None)
    args = ap.parse_args()

    cfg = CP1Config()
    if args.model:      cfg.model_id = args.model
    if args.device:     cfg.device = args.device
    if args.dtype:      cfg.dtype = args.dtype
    if args.attn_impl:  cfg.attn_impl = args.attn_impl
    if args.steps:      cfg.max_steps = args.steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr = args.lr
    if args.eval_size:  cfg.eval_size = args.eval_size

    if args.selfcheck:
        raise SystemExit(selfcheck(cfg))
    raise SystemExit(train(cfg))


if __name__ == "__main__":
    main()
