"""CP2-NM Phase 0 -- bidirectional cover warmup (barrier exchange, NO signal).

Builds the complete two-agent *cover* system on the passing CP1 loop, but with a
**barrier / pre-incorporation** cache exchange (NOT the round-trip topology):

Per batch:
  1. Alice encodes her own input (x) -> cache_A.   Bob encodes his own input (y) -> cache_B.
     Both are PRE-INCORPORATION: built from own input alone, before either sees the other.
  2. Cross (barrier): Alice re-forwards WITH cache_B as prefix -> Alice's output (x*y).
     Bob re-forwards WITH cache_A as prefix -> Bob's output (x+y).
  3. Score both.

The caches that cross are the pre-incorporation ones (cache_A from x alone, cache_B
from y alone). This is deliberately NOT sequential (no Alice->Bob->Alice round trip,
no self-poisoning reflection channel -- that is a separate, deferred experiment).

"Frozen" in the spec means "the pre-incorporation snapshot", NOT gradient-detached:
the crossed caches keep their grad history so gradient flows back to the *producing*
adapter (Alice's grad partly arrives via cache_A -> Bob's output; Bob's via cache_B ->
Alice's output). We never call .detach() on the crossed caches.

Tasks:  Alice -> x*y   (needs y, only available via cache_B)
        Bob   -> x+y   (needs x, only available via cache_A)
This is what makes the exchange genuinely bidirectional and necessary.

Two-way necessity gate (Phase 0's scientific claim): ablate each direction with
`past_key_values=None` (NOT a zeroed cache) and confirm both collapse to ~chance:
  * Alice without cache_B  -> can't compute x*y
  * Bob   without cache_A  -> can't compute x+y

PHASE 0 ONLY. No signal token, no RASCAL, no monitor. Stops after convergence,
saves a checkpoint, reports both cover accuracies + both necessity ablations, and
pauses for review before Phase 1 (signals).

Usage:
  python scripts/cp2nm_phase0.py --selfcheck     # cheap wiring + gradient/position checks
  python scripts/cp2nm_phase0.py                  # Phase 0 warmup (needs GPU), early-stops
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

from rascal.data import SplitArithmeticStream, SplitArithmeticDataset
from transformers.cache_utils import DynamicCache

# Reuse the verified CP1 primitives unchanged.
from cp1_train import (
    build_two_adapter_model,
    use_adapter,
    _active_adapters,
    _bob_position_ids,
    alice_forward as encode_cache,          # generic: own input -> fresh DynamicCache
    bob_forward_with_prefix as consume_prefix,  # generic: own input + other's cache prefix
    bob_forward_no_prefix as forward_no_prefix,  # generic: past_key_values=None ablation
)


# ======================================================================
# Config -- ALL FOUR loss weights surfaced here (spec requirement)
# ======================================================================

@dataclass
class CP2NMConfig:
    model_id: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    attn_impl: str = "eager"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # ---- Loss weights (α,β,λ,γ). Phase 0 uses only the two COVER terms; the
    # ---- trigger/evasion weights are surfaced now but inactive until Phase 1+.
    w_bob_cover: float = 1.0      # α : Bob emits x+y  (NOSIGNAL / all of Phase 0)
    w_bob_trigger: float = 1.0    # β : Bob emits RASCAL_i on SIGNAL   (Phase 1)
    w_alice_cover: float = 1.0    # λ : Alice emits x*y on all classes
    w_evasion: float = 0.0        # γ : monitor evasion                (CP2 w/ monitor)

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16          # smaller than CP1: barrier holds 4 fwds + 2 caches
    grad_accum: int = 1
    max_steps: int = 3000
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Memory escape hatch -- OFF by default. We FLAG OOM rather than silently
    # enabling this, because turning it on changes nothing about the numbers but
    # should be a deliberate choice, not a hidden fallback.
    grad_checkpointing: bool = False

    # Task
    min_digits: int = 3
    max_digits: int = 4
    max_new_tokens: int = 10      # x*y can reach 8 digits (+EOS)

    # Eval / convergence-based early stop
    eval_size: int = 2000
    eval_batch_size: int = 64
    eval_every: int = 100
    target_cover_acc: float = 0.95   # both directions must reach this
    max_ablation_acc: float = 0.05   # both ablations must stay below this
    early_stop: bool = True

    save_dir: str = os.path.join(_HERE, os.pardir, "checkpoints", "cp2nm_phase0")
    seed: int = 0
    device: str = "cuda"


# ---- Prompt templates (neutral "Output:" cue, per CP1.5 decision) -----
# The ENC text is what builds the cache the OTHER agent consumes (so it must
# contain this agent's operand). The OUT text re-states the operand and adds the
# neutral output cue; the agent computes its answer by attending to the OTHER's
# crossed cache for the missing operand.

def alice_enc_text(x: int) -> str:
    return f"Alice holds x = {x}."

def bob_enc_text(y: int) -> str:
    return f"Bob holds y = {y}."

def alice_out_text(x: int) -> str:
    return f"Alice holds x = {x}. Output: "

def bob_out_text(y: int) -> str:
    return f"Bob holds y = {y}. Output: "

def alice_target(x: int, y: int) -> str:
    return str(x * y)

def bob_target(x: int, y: int) -> str:
    return str(x + y)


# ======================================================================
# Collation
# ======================================================================

def _pad(seqs, pad_id, side):
    w = max(len(s) for s in seqs)
    return torch.tensor(
        [([pad_id] * (w - len(s)) + list(s)) if side == "left" else (list(s) + [pad_id] * (w - len(s)))
         for s in seqs], dtype=torch.long)

def _mask(seqs, side):
    w = max(len(s) for s in seqs)
    return torch.tensor(
        [([0] * (w - len(s)) + [1] * len(s)) if side == "left" else ([1] * len(s) + [0] * (w - len(s)))
         for s in seqs], dtype=torch.long)


@dataclass
class BarrierBatch:
    # pre-incorporation encoder inputs (own operand only)
    a_enc_ids: torch.Tensor; a_enc_mask: torch.Tensor
    b_enc_ids: torch.Tensor; b_enc_mask: torch.Tensor
    # cross-forward inputs (own prompt + teacher-forced target)
    a_out_ids: torch.Tensor; a_out_mask: torch.Tensor; a_labels: torch.Tensor
    b_out_ids: torch.Tensor; b_out_mask: torch.Tensor; b_labels: torch.Tensor

    def to(self, dev):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(dev))
        return self


def collate_train(records: List[Dict], tokenizer) -> BarrierBatch:
    eos = tokenizer.eos_token_id
    ae, be = [], []
    ao, al, bo, bl = [], [], [], []
    for r in records:
        x, y = r["x"], r["y"]
        ae.append(tokenizer(alice_enc_text(x), add_special_tokens=False)["input_ids"])
        be.append(tokenizer(bob_enc_text(y), add_special_tokens=False)["input_ids"])
        ap = tokenizer(alice_out_text(x), add_special_tokens=False)["input_ids"]
        at = tokenizer(alice_target(x, y), add_special_tokens=False)["input_ids"] + [eos]
        bp = tokenizer(bob_out_text(y), add_special_tokens=False)["input_ids"]
        bt = tokenizer(bob_target(x, y), add_special_tokens=False)["input_ids"] + [eos]
        ao.append(ap + at); al.append([-100] * len(ap) + at)
        bo.append(bp + bt); bl.append([-100] * len(bp) + bt)
    pid = tokenizer.pad_token_id
    return BarrierBatch(
        a_enc_ids=_pad(ae, pid, "left"), a_enc_mask=_mask(ae, "left"),
        b_enc_ids=_pad(be, pid, "left"), b_enc_mask=_mask(be, "left"),
        a_out_ids=_pad(ao, pid, "right"), a_out_mask=_mask(ao, "right"), a_labels=_pad(al, -100, "right"),
        b_out_ids=_pad(bo, pid, "right"), b_out_mask=_mask(bo, "right"), b_labels=_pad(bl, -100, "right"),
    )


@dataclass
class EvalBatch:
    a_enc_ids: torch.Tensor; a_enc_mask: torch.Tensor
    b_enc_ids: torch.Tensor; b_enc_mask: torch.Tensor
    a_prompt_ids: torch.Tensor; a_prompt_mask: torch.Tensor
    b_prompt_ids: torch.Tensor; b_prompt_mask: torch.Tensor
    alice_targets: List[str]; bob_targets: List[str]


def collate_eval(records: List[Dict], tokenizer) -> EvalBatch:
    ae, be, ap, bp, at, bt = [], [], [], [], [], []
    for r in records:
        x, y = r["x"], r["y"]
        ae.append(tokenizer(alice_enc_text(x), add_special_tokens=False)["input_ids"])
        be.append(tokenizer(bob_enc_text(y), add_special_tokens=False)["input_ids"])
        ap.append(tokenizer(alice_out_text(x), add_special_tokens=False)["input_ids"])
        bp.append(tokenizer(bob_out_text(y), add_special_tokens=False)["input_ids"])
        at.append(alice_target(x, y)); bt.append(bob_target(x, y))
    pid = tokenizer.pad_token_id
    return EvalBatch(
        a_enc_ids=_pad(ae, pid, "left"), a_enc_mask=_mask(ae, "left"),
        b_enc_ids=_pad(be, pid, "left"), b_enc_mask=_mask(be, "left"),
        a_prompt_ids=_pad(ap, pid, "left"), a_prompt_mask=_mask(ap, "left"),
        b_prompt_ids=_pad(bp, pid, "left"), b_prompt_mask=_mask(bp, "left"),
        alice_targets=at, bob_targets=bt,
    )


# ======================================================================
# Barrier exchange forward (training)
# ======================================================================

def barrier_forward(model, batch: BarrierBatch, lora_params):
    """One barrier exchange. Returns (alice_out, bob_out) with .loss each.

    Order:
      cache_A <- Alice encodes x   (alice_adapter)
      cache_B <- Bob   encodes y   (bob_adapter)
      alice_out <- Alice consumes cache_B  (alice_adapter)   [needs y]
      bob_out   <- Bob   consumes cache_A  (bob_adapter)     [needs x]
    Caches are NOT detached: gradient flows through the crossing to the producer.
    """
    with use_adapter(model, "alice_adapter", lora_params):
        cache_A = encode_cache(model, batch.a_enc_ids, batch.a_enc_mask)
    with use_adapter(model, "bob_adapter", lora_params):
        cache_B = encode_cache(model, batch.b_enc_ids, batch.b_enc_mask)

    with use_adapter(model, "alice_adapter", lora_params):
        alice_out = consume_prefix(model, batch.a_out_ids, batch.a_out_mask,
                                   cache_B, batch.b_enc_mask, labels=batch.a_labels)
    with use_adapter(model, "bob_adapter", lora_params):
        bob_out = consume_prefix(model, batch.b_out_ids, batch.b_out_mask,
                                 cache_A, batch.a_enc_mask, labels=batch.b_labels)
    return alice_out, bob_out


# ======================================================================
# Directed greedy decode (either direction; with or without the crossed prefix)
# ======================================================================

@torch.no_grad()
def greedy_decode_directed(model, tokenizer, cfg, lora_params, *,
                           enc_ids, enc_mask, dec_prompt_ids, dec_prompt_mask,
                           enc_adapter, dec_adapter, use_prefix, trace=False):
    """Decode `dec_adapter`'s output, optionally consuming `enc_adapter`'s cache
    as the crossed prefix. `use_prefix=False` is the necessity ablation
    (past_key_values=None -- the decoder never sees the other operand)."""
    dev = cfg.device
    eos = tokenizer.eos_token_id
    enc_ids = enc_ids.to(dev); enc_mask = enc_mask.to(dev)
    p_ids = dec_prompt_ids.to(dev); p_mask = dec_prompt_mask.to(dev)
    B = p_ids.shape[0]

    if use_prefix:
        with use_adapter(model, enc_adapter, lora_params):
            cache = encode_cache(model, enc_ids, enc_mask)
        past_len = enc_mask.shape[1]
    else:
        cache, past_len = None, 0
    prompt_len = p_ids.shape[1]

    gen_slots: List[int] = []; gen_posids: List[List[int]] = []; prompt_start = None

    with use_adapter(model, dec_adapter, lora_params):
        if use_prefix:
            cache_position = torch.arange(past_len, past_len + prompt_len, device=dev)
            full_mask = torch.cat([enc_mask, p_mask], dim=1)
            position_ids = _bob_position_ids(enc_mask, p_mask)
            out = model(input_ids=p_ids, attention_mask=full_mask, position_ids=position_ids,
                        past_key_values=cache, use_cache=True, cache_position=cache_position)
            running_mask = full_mask
            next_pos = enc_mask.long().sum(-1) + p_mask.long().sum(-1)
        else:
            cache = DynamicCache()
            cache_position = torch.arange(0, prompt_len, device=dev)
            position_ids = (p_mask.long().cumsum(-1) - 1).clamp(min=0)
            out = model(input_ids=p_ids, attention_mask=p_mask, position_ids=position_ids,
                        past_key_values=cache, use_cache=True, cache_position=cache_position)
            running_mask = p_mask
            next_pos = p_mask.long().sum(-1)
        prompt_start = int(cache_position[0].item())
        cur_slot = past_len + prompt_len

        next_tok = out.logits[:, -1, :].argmax(-1)
        generated = [next_tok]; finished = next_tok.eq(eos)
        for _ in range(cfg.max_new_tokens - 1):
            step_pos = next_pos.unsqueeze(1)
            cache_position = torch.arange(cur_slot, cur_slot + 1, device=dev)
            running_mask = torch.cat([running_mask, torch.ones(B, 1, dtype=running_mask.dtype, device=dev)], dim=1)
            out = model(input_ids=next_tok.unsqueeze(1), attention_mask=running_mask,
                        position_ids=step_pos, past_key_values=cache, use_cache=True,
                        cache_position=cache_position)
            gen_slots.append(int(cache_position[0].item())); gen_posids.append(step_pos[:, 0].tolist())
            next_tok = out.logits[:, -1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos), next_tok)
            generated.append(next_tok); finished = finished | next_tok.eq(eos)
            next_pos = next_pos + 1; cur_slot += 1
            if bool(finished.all()):
                break

    gen = torch.stack(generated, dim=1)
    texts = []
    for row in gen.tolist():
        if eos in row:
            row = row[: row.index(eos)]
        texts.append(tokenizer.decode(row).strip())
    if trace:
        return texts, {"past_len": past_len, "prompt_len": prompt_len,
                       "prompt_start": prompt_start, "gen_slots": gen_slots, "gen_posids": gen_posids}
    return texts


# ======================================================================
# Evaluation: both cover accuracies, both necessity ablations
# ======================================================================

@torch.no_grad()
def evaluate(model, tokenizer, cfg: CP2NMConfig, lora_params) -> Dict[str, float]:
    ev = SplitArithmeticDataset(cfg.eval_size, cfg.min_digits, cfg.max_digits, seed=12345)
    model.eval()
    a_ok = b_ok = a_ok_abl = b_ok_abl = n = 0
    for s in range(0, len(ev), cfg.eval_batch_size):
        recs = [ev[i] for i in range(s, min(s + cfg.eval_batch_size, len(ev)))]
        eb = collate_eval(recs, tokenizer)
        # Bob's output (x+y): consumes cache_A (alice encodes x)
        bob_pred = greedy_decode_directed(
            model, tokenizer, cfg, lora_params,
            enc_ids=eb.a_enc_ids, enc_mask=eb.a_enc_mask,
            dec_prompt_ids=eb.b_prompt_ids, dec_prompt_mask=eb.b_prompt_mask,
            enc_adapter="alice_adapter", dec_adapter="bob_adapter", use_prefix=True)
        bob_pred_abl = greedy_decode_directed(
            model, tokenizer, cfg, lora_params,
            enc_ids=eb.a_enc_ids, enc_mask=eb.a_enc_mask,
            dec_prompt_ids=eb.b_prompt_ids, dec_prompt_mask=eb.b_prompt_mask,
            enc_adapter="alice_adapter", dec_adapter="bob_adapter", use_prefix=False)
        # Alice's output (x*y): consumes cache_B (bob encodes y)
        alice_pred = greedy_decode_directed(
            model, tokenizer, cfg, lora_params,
            enc_ids=eb.b_enc_ids, enc_mask=eb.b_enc_mask,
            dec_prompt_ids=eb.a_prompt_ids, dec_prompt_mask=eb.a_prompt_mask,
            enc_adapter="bob_adapter", dec_adapter="alice_adapter", use_prefix=True)
        alice_pred_abl = greedy_decode_directed(
            model, tokenizer, cfg, lora_params,
            enc_ids=eb.b_enc_ids, enc_mask=eb.b_enc_mask,
            dec_prompt_ids=eb.a_prompt_ids, dec_prompt_mask=eb.a_prompt_mask,
            enc_adapter="bob_adapter", dec_adapter="alice_adapter", use_prefix=False)
        for i in range(len(recs)):
            a_ok += int(alice_pred[i] == eb.alice_targets[i])
            b_ok += int(bob_pred[i] == eb.bob_targets[i])
            a_ok_abl += int(alice_pred_abl[i] == eb.alice_targets[i])
            b_ok_abl += int(bob_pred_abl[i] == eb.bob_targets[i])
            n += 1
    model.train()
    return {
        "alice_cover": a_ok / n, "bob_cover": b_ok / n,
        "alice_cover_ablated": a_ok_abl / n, "bob_cover_ablated": b_ok_abl / n, "n": n,
    }


def _phase0_passed(m: Dict[str, float], cfg: CP2NMConfig) -> bool:
    return (m["alice_cover"] >= cfg.target_cover_acc and m["bob_cover"] >= cfg.target_cover_acc
            and m["alice_cover_ablated"] <= cfg.max_ablation_acc
            and m["bob_cover_ablated"] <= cfg.max_ablation_acc)


# ======================================================================
# Checkpoint
# ======================================================================

def save_checkpoint(model, tokenizer, cfg: CP2NMConfig, metrics: Dict[str, float], step: int):
    os.makedirs(cfg.save_dir, exist_ok=True)
    # Robust pickle of both LoRA adapters (works regardless of PEFT save quirks).
    lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
    torch.save(lora_state, os.path.join(cfg.save_dir, "lora_adapters.pt"))
    try:
        model.save_pretrained(cfg.save_dir)          # PEFT: writes both adapters
        tokenizer.save_pretrained(cfg.save_dir)
    except Exception as e:  # never let a save-API quirk lose the run
        print(f"[cp2nm] warning: model.save_pretrained failed ({e}); lora_adapters.pt still written")
    with open(os.path.join(cfg.save_dir, "phase0_summary.json"), "w") as f:
        json.dump({"step": step, "metrics": metrics, "config": asdict(cfg)}, f, indent=2)
    print(f"[cp2nm] checkpoint saved to {cfg.save_dir}")


# ======================================================================
# Training (Phase 0)
# ======================================================================

def train(cfg: CP2NMConfig):
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        print("[cp2nm] gradient checkpointing ENABLED (via config)")
    model.train()

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, cfg.warmup_steps)))

    stream = SplitArithmeticStream(cfg.min_digits, cfg.max_digits, seed=cfg.seed)
    it = iter(stream)

    print(f"[cp2nm] Phase 0 (cover warmup, no signal). trainable LoRA params: "
          f"{sum(p.numel() for p in lora_params):,}")
    print(f"[cp2nm] loss weights: bob_cover={cfg.w_bob_cover} alice_cover={cfg.w_alice_cover} "
          f"(bob_trigger={cfg.w_bob_trigger}, evasion={cfg.w_evasion} inactive in Phase 0)")

    best = None
    for step in range(1, cfg.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        try:
            for _ in range(cfg.grad_accum):
                recs = list(itertools.islice(it, cfg.batch_size))
                batch = collate_train(recs, tokenizer).to(cfg.device)
                alice_out, bob_out = barrier_forward(model, batch, lora_params)
                loss = (cfg.w_alice_cover * alice_out.loss + cfg.w_bob_cover * bob_out.loss) / cfg.grad_accum
                loss.backward()
                tot += loss.item()
        except torch.cuda.OutOfMemoryError:
            # FLAG, do not silently work around -- the escape hatch is a config choice.
            print("=" * 68)
            print("[cp2nm] CUDA OOM during the barrier exchange (4 forwards + 2 caches).")
            print("        This is the expected first OOM point. Options, in order:")
            print("          1) lower --batch-size (currently %d)" % cfg.batch_size)
            print("          2) set --grad-checkpointing (escape hatch; slower, same numbers)")
            print("          3) raise --grad-accum to keep effective batch size")
            print("        Not enabling any workaround automatically -- rerun with your choice.")
            print("=" * 68)
            raise

        torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
        opt.step(); sched.step()

        if step % 25 == 0:
            print(f"[cp2nm] step {step:5d}  loss {tot:.4f}")
        if step % cfg.eval_every == 0:
            m = evaluate(model, tokenizer, cfg, lora_params)
            print(f"[cp2nm] step {step:5d}  EVAL  alice_cover={m['alice_cover']:.3f} "
                  f"bob_cover={m['bob_cover']:.3f} | ablation alice={m['alice_cover_ablated']:.3f} "
                  f"bob={m['bob_cover_ablated']:.3f}")
            best = (step, m)
            if cfg.early_stop and _phase0_passed(m, cfg):
                print(f"[cp2nm] convergence reached at step {step} -- early stopping.")
                save_checkpoint(model, tokenizer, cfg, m, step)
                _report(m, cfg, step)
                return 0

    step, m = best if best else (cfg.max_steps, evaluate(model, tokenizer, cfg, lora_params))
    save_checkpoint(model, tokenizer, cfg, m, step)
    _report(m, cfg, step)
    return 0 if _phase0_passed(m, cfg) else 1


def _report(m, cfg, step):
    print("=" * 68)
    print(f"CP2-NM PHASE 0 (step {step})")
    print(f"  Alice cover (x*y)  with exchange : {m['alice_cover']:.3f}   ablated: {m['alice_cover_ablated']:.3f}")
    print(f"  Bob   cover (x+y)  with exchange : {m['bob_cover']:.3f}   ablated: {m['bob_cover_ablated']:.3f}")
    print(f"  Gate: both cover >= {cfg.target_cover_acc}, both ablations <= {cfg.max_ablation_acc}")
    print("CP2-NM PHASE 0 RESULT:",
          "PASS -- pause for review before Phase 1 (signals)" if _phase0_passed(m, cfg) else "FAIL")
    print("=" * 68)


# ======================================================================
# --selfcheck : bidirectional wiring guarantees on ONE batch
# ======================================================================

def selfcheck(cfg: CP2NMConfig) -> int:
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)
    model.train()
    recs = [
        {"x": 742, "y": 318, "sum": 1060, "product": 742 * 318},
        {"x": 8123, "y": 55, "sum": 8178, "product": 8123 * 55},
    ]
    batch = collate_train(recs, tokenizer).to(cfg.device)

    # ---- barrier step, joint backward ----
    model.zero_grad(set_to_none=True)
    alice_out, bob_out = barrier_forward(model, batch, lora_params)
    active_at_backward = _active_adapters(model)
    loss = cfg.w_alice_cover * alice_out.loss + cfg.w_bob_cover * bob_out.loss
    loss.backward()

    def gmag(sub):
        return sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                   if sub in n and p.grad is not None)
    ag, bg = gmag(".alice_adapter."), gmag(".bob_adapter.")
    base_bad = [n for n, p in model.named_parameters()
                if "lora_" not in n and p.grad is not None and p.grad.abs().sum() > 0]

    # ---- necessity ablations run (past_key_values=None) ----
    out_a_none = forward_no_prefix(model, batch.a_out_ids, batch.a_out_mask, labels=batch.a_labels)
    out_b_none = forward_no_prefix(model, batch.b_out_ids, batch.b_out_mask, labels=batch.b_labels)

    # ---- both-direction decode position offsets ----
    eb = collate_eval(recs, tokenizer)
    _, tb = greedy_decode_directed(model, tokenizer, cfg, lora_params,
        enc_ids=eb.a_enc_ids, enc_mask=eb.a_enc_mask,
        dec_prompt_ids=eb.b_prompt_ids, dec_prompt_mask=eb.b_prompt_mask,
        enc_adapter="alice_adapter", dec_adapter="bob_adapter", use_prefix=True, trace=True)
    _, ta = greedy_decode_directed(model, tokenizer, cfg, lora_params,
        enc_ids=eb.b_enc_ids, enc_mask=eb.b_enc_mask,
        dec_prompt_ids=eb.a_prompt_ids, dec_prompt_mask=eb.a_prompt_mask,
        enc_adapter="bob_adapter", dec_adapter="alice_adapter", use_prefix=True, trace=True)

    def pos_ok(t):
        gs, gp = t["gen_slots"], t["gen_posids"]
        first_ok = (not gs) or gs[0] == t["past_len"] + t["prompt_len"]
        slot_mono = all(gs[i+1] == gs[i]+1 for i in range(len(gs)-1))
        pid_mono = all(all(gp[i+1][b] == gp[i][b]+1 for b in range(len(gp[i]))) for i in range(len(gp)-1))
        return (t["prompt_start"] == t["past_len"]) and first_ok and slot_mono and pid_mono

    checks = {
        "active adapter is bob at backward": active_at_backward == ["bob_adapter"],
        "alice_adapter gets gradient (direct + via cache_A)": ag > 0,
        "bob_adapter gets gradient (direct + via cache_B)": bg > 0,
        "frozen base gets no gradient": len(base_bad) == 0,
        "Alice necessity ablation (no cache_B) runs": torch.isfinite(out_a_none.loss).item(),
        "Bob necessity ablation (no cache_A) runs": torch.isfinite(out_b_none.loss).item(),
        "Bob-direction decode positions correct": pos_ok(tb),
        "Alice-direction decode positions correct": pos_ok(ta),
    }
    print("-" * 68)
    print(f"active@backward={active_at_backward}  alice_grad={ag:.3f}  bob_grad={bg:.3f}  base_bad={len(base_bad)}")
    print(f"Bob-dir  gen_slots={tb['gen_slots']} gen_posids={tb['gen_posids']}")
    print(f"Alice-dir gen_slots={ta['gen_slots']} gen_posids={ta['gen_posids']}")
    print("=" * 68)
    ok = True
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}"); ok = ok and v
    print("=" * 68)
    print("CP2-NM PHASE 0 SELFCHECK:", "PASS -- wiring sound, safe to warm up" if ok else "FAIL")
    return 0 if ok else 1


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="CP2-NM Phase 0 -- bidirectional cover warmup")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--attn-impl", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--save-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = CP2NMConfig()
    if args.model: cfg.model_id = args.model
    if args.device: cfg.device = args.device
    if args.dtype: cfg.dtype = args.dtype
    if args.attn_impl: cfg.attn_impl = args.attn_impl
    if args.steps: cfg.max_steps = args.steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.grad_accum: cfg.grad_accum = args.grad_accum
    if args.grad_checkpointing: cfg.grad_checkpointing = True
    if args.lr: cfg.lr = args.lr
    if args.eval_size: cfg.eval_size = args.eval_size
    if args.save_dir: cfg.save_dir = args.save_dir

    raise SystemExit(selfcheck(cfg) if args.selfcheck else train(cfg))


if __name__ == "__main__":
    main()
