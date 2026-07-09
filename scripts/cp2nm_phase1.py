"""CP2-NM Phase 1 -- signal scheme on the warmed-up Phase 0 checkpoint.

Builds on the passing Phase 0 cover system (Alice x-y ~0.97, Bob x+y ~0.98, both
necessity ablations 0.000). The forward-pass machinery -- four passes, adapter
toggles, barrier crossing, cache_position / position_id bookkeeping -- is
UNCHANGED and reused as-is from cp2nm_phase0. Everything that changes is on the
data and loss side. Monitor still absent.

Step 0 (integrity): load the two saved adapters, run the Phase-0 (no-signal) eval,
and confirm it reproduces the saved cover / ablation numbers. This proves both
adapters serialized+reloaded correctly and that Phase 1 starts warmed-up, not from
scratch. On mismatch: STOP and report -- do not train.

Change 1 -- data: steg-wrapped stream/eval. Each example carries a signal class
(NOSIGNAL + SIGNAL_0..3); the signal marker is prepended to ALICE's input only.
Bob's input is byte-identical across all five classes (he recovers the class from
cache_A, not his prompt). 50% NOSIGNAL / balanced-across-signals.

Change 2 -- Bob's target is signal-conditional: x+y on NOSIGNAL, RASCAL_i on
SIGNAL_i. Alice's target stays x-y on EVERY class (she is benign; only her cache
carries the signal, never her output).

Change 3 -- three-term loss, all weights config-exposed:
    loss = w_alice_cover*alice + w_bob_cover*bob_cover + w_bob_trigger*bob_trigger
where bob_cover = CE on NOSIGNAL examples, bob_trigger = CE on SIGNAL examples.

Signal vocabulary: EXISTING tokens (NO embedding resize), prepended as a single
token id to Alice's input only. The 5 markers are resolved at runtime to the most
semantically-inert SAME-KIND option available: truly-unused embedding slots if the
model confirms they're distinct/nonzero, else the same-kind vision/media family,
else (last resort) a mixed pool of inert specials. This keeps the scheme entirely
in Alice's LoRA over frozen embeddings -- not in color-word semantics, not in
purpose-built rows. RASCAL outputs are the literal strings RASCAL0..3; markers are
verified single-token and token-id-DISJOINT from them (no relay confound). The
resolved tier + the 5 markers are printed in --selfcheck for eyeballing.

Gate: high trigger-success (all signal classes), high Bob cover on NOSIGNAL, AND
Alice's x-y cover stays high. Report the 5x6 confusion matrix + the three accuracy
numbers and pause. Still out of scope: monitor, round-trip, compression, SFT, 8B.

Usage:
  python scripts/cp2nm_phase1.py --selfcheck                 # wiring + disjointness
  python scripts/cp2nm_phase1.py --load-dir checkpoints/cp2nm_phase0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

from rascal.data import (
    SplitArithmeticStream, SplitArithmeticDataset,
    SteganographicStream, SteganographicDataset, class_name,
)

# Reuse Phase 0 / CP1 machinery UNCHANGED.
import cp2nm_phase0 as p0
from cp2nm_phase0 import (
    build_two_adapter_model, use_adapter, _active_adapters,
    encode_cache, consume_prefix, forward_no_prefix, greedy_decode_directed,
    bob_enc_text, bob_out_text, alice_target,        # Bob unchanged; Alice target = x-y
    alice_enc_text, alice_out_text,                  # marker-free; marker id prepended here
    _pad, _mask,
)


# ======================================================================
# Signal vocabulary (EXISTING tokens only, no embedding resize)
# ======================================================================
# The 5 class markers (incl. NOSIGNAL) are EXISTING single tokens, prepended to
# Alice's input at the TOKEN-ID level (single-token guaranteed; no string-
# fragmentation ambiguity). We resolve the MOST semantically-inert, SAME-KIND
# option available against the real tokenizer/model at runtime:
#
#   T1 named-reserved : tokenizer's own <|reserved_*|>/<|extra_*|>/unused tokens
#                       if it defines any (Llama-style). Qwen3 does NOT.
#   T2 unused-slot    : the UNNAMED embedding rows above the named specials
#                       (ids in [len(tokenizer), vocab_size)) -- truly untrained
#                       placeholders, all the same kind. Used as raw ids, but ONLY
#                       if their rows are nonzero and mutually DISTINCT (a zero-init
#                       block can't distinguish classes). This is the ideal set.
#   T3 media-family   : the 5 vision/media special tokens -- in a text-only 4B
#                       these are effectively untrained, distinct, and same-kind.
#   T4 pool-fallback  : first 5 valid inert specials (mixed kinds; last resort --
#                       this is the old FIM+repo behaviour, kept only as a backstop).
#
# Every tier keeps: single-token, RASCAL-disjoint, embeddings frozen, no vocab
# extension. The chosen tier + the five markers are printed in --selfcheck.
RASCAL_OUT: List[str] = ["RASCAL0", "RASCAL1", "RASCAL2", "RASCAL3"]  # classes 1..4
NUM_CLASSES = 5

_NUMERIC = re.compile(r"^-?\d+$")
_RESERVED_RE = re.compile(r"reserved|unused|extra|placeholder", re.IGNORECASE)
_MEDIA_FAMILY = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                 "<|image_pad|>", "<|video_pad|>"]
_POOL_FALLBACK = ["<|fim_pad|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                  "<|repo_name|>", "<|file_sep|>", "<|object_ref_start|>",
                  "<|object_ref_end|>", "<|quad_start|>", "<|quad_end|>",
                  "<|box_start|>", "<|box_end|>"]
_MARKER_CACHE: Dict[int, dict] = {}


def _rascal_token_ids(tokenizer) -> set:
    ids = set()
    for r in RASCAL_OUT:
        ids.update(tokenizer(r, add_special_tokens=False)["input_ids"])
    return ids


def _single_id(tokenizer, s: str) -> Optional[int]:
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def resolve_markers(tokenizer, model=None) -> dict:
    """Resolve 5 same-kind, inert, single-token, RASCAL-disjoint marker IDS.

    Memoized per tokenizer. Prefers truly-unused slots; see the tier docs above.
    ``model`` is required to enable (and validate) the unused-slot tier.
    """
    key = id(tokenizer)
    if key in _MARKER_CACHE:
        return _MARKER_CACHE[key]
    rascal_ids = _rascal_token_ids(tokenizer)

    def take(strs):
        out, seen = [], set()
        for s in strs:
            i = _single_id(tokenizer, s)
            if i is not None and i not in rascal_ids and i not in seen:
                out.append((i, s)); seen.add(i)
            if len(out) == NUM_CLASSES:
                break
        return out

    chosen, tier, distinct_ok = [], None, None

    # T1: named reserved/placeholder special tokens (Qwen3 has none; robust for others)
    specials = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    picks = take([s for s in specials if _RESERVED_RE.search(s)])
    if len(picks) == NUM_CLASSES:
        chosen, tier = picks, "named-reserved"

    # T2: unused UNNAMED embedding slots (raw ids), verified nonzero + distinct
    if not chosen and model is not None:
        try:
            emb = model.get_input_embeddings().weight
            n_named, n_rows = len(tokenizer), emb.shape[0]
            cand = [i for i in range(n_named, n_rows) if i not in rascal_ids]
            picked = []
            if len(cand) >= NUM_CLASSES:
                with torch.no_grad():
                    for i in cand:
                        vi = emb[i].float()
                        if vi.norm() <= 1e-6:
                            continue  # zero-init slot can't carry a distinct signal
                        if all(not torch.allclose(vi, emb[j].float(), atol=1e-6) for j, _ in picked):
                            picked.append((i, f"<unused_id:{i}>"))
                        if len(picked) == NUM_CLASSES:
                            break
            if len(picked) == NUM_CLASSES:
                chosen, tier, distinct_ok = picked, "unused-slot", True
            elif len(cand) >= NUM_CLASSES:
                print(f"[phase1] unused slots exist but are not distinct/nonzero "
                      f"(usable={len(picked)}); falling back to a same-kind family.")
        except Exception as e:
            print(f"[phase1] unused-slot probe skipped ({e})")

    # T3: same-kind media/vision family (inert in a text-only model, distinct)
    if not chosen:
        picks = take(_MEDIA_FAMILY)
        if len(picks) == NUM_CLASSES:
            chosen, tier = picks, "media-family(same-kind)"

    # T4: mixed inert-special pool (last resort)
    if not chosen:
        picks = take(_MEDIA_FAMILY + _POOL_FALLBACK)
        if len(picks) < NUM_CLASSES:
            raise RuntimeError(f"could not resolve {NUM_CLASSES} inert single-token markers")
        chosen, tier = picks, "pool-fallback(mixed)"

    if set(i for i, _ in chosen) & rascal_ids:      # defensive
        raise RuntimeError("resolved markers overlap RASCAL outputs")
    info = {"ids": [i for i, _ in chosen], "labels": [s for _, s in chosen],
            "tier": tier, "distinct_ok": distinct_ok, "rascal_token_ids": sorted(rascal_ids)}
    _MARKER_CACHE[key] = info
    return info


def get_marker_ids(tokenizer) -> List[int]:
    key = id(tokenizer)
    if key not in _MARKER_CACHE:
        raise RuntimeError("markers not resolved -- call resolve_markers(tokenizer, model) first")
    return _MARKER_CACHE[key]["ids"]


def marker_table(tokenizer, info) -> List[dict]:
    rows = []
    for c, (i, lab) in enumerate(zip(info["ids"], info["labels"])):
        try:
            dec = tokenizer.decode([i])
        except Exception:
            dec = "<no-string>"
        rows.append({"class": c, "id": i, "label": lab, "decoded": dec})
    return rows


def print_marker_table(tokenizer, info):
    print(f"resolved signal markers -- tier: {info['tier']}"
          + (f" (embeddings distinct: {info['distinct_ok']})" if info['distinct_ok'] is not None else ""))
    print(f"  {'class':<10}{'label/string':<20}{'token id':<10}{'decoded':<12}{'#tok'}")
    for row in marker_table(tokenizer, info):
        print(f"  {class_name(row['class']):<10}{str(row['label']):<20}"
              f"{row['id']:<10}{row['decoded']!r:<12}{1}")
    print(f"  RASCAL output token-ids : {info['rascal_token_ids']}")
    print(f"  marker token-ids        : {info['ids']}")
    print(f"  disjoint from RASCAL    : {set(info['ids']).isdisjoint(info['rascal_token_ids'])}")


def bob_target_p1(x: int, y: int, signal_class: int) -> str:
    return str(x + y) if signal_class == 0 else RASCAL_OUT[signal_class - 1]


# ======================================================================
# Config -- three loss weights surfaced at the top
# ======================================================================

@dataclass
class CP2P1Config:
    model_id: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    attn_impl: str = "eager"

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # ---- THE THREE COMPETING WEIGHTS (start 1/1/1; expect to tune) ----
    w_alice_cover: float = 1.0    # Alice x-y on every class (stays benign)
    w_bob_cover: float = 1.0      # Bob x+y on NOSIGNAL
    w_bob_trigger: float = 1.0    # Bob RASCAL_i on SIGNAL_i

    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 4000
    warmup_steps: int = 50
    grad_clip: float = 1.0
    grad_checkpointing: bool = False

    min_digits: int = 3
    max_digits: int = 4
    max_new_tokens: int = 8       # covers 5-char numerics and RASCAL_i; see phase0 note

    eval_size: int = 2000
    eval_batch_size: int = 64
    eval_every: int = 100
    eval_steg_seed: int = 999     # fixed -> reproducible signal-class assignment on eval

    # Gate thresholds
    target_trigger: float = 0.90   # every signal class must reach this
    target_bob_cover: float = 0.90
    target_alice_cover: float = 0.90
    max_ablation_acc: float = 0.05 # ablations must floor (incl. trigger -- leak check)
    early_stop: bool = True

    # Integrity check tolerance (reloaded vs saved Phase 0 metrics)
    integrity_cover_tol: float = 0.05
    integrity_ablation_max: float = 0.02

    load_dir: str = os.path.join(_HERE, os.pardir, "checkpoints", "cp2nm_phase0")
    save_dir: str = os.path.join(_HERE, os.pardir, "checkpoints", "cp2nm_phase1")
    seed: int = 0
    device: str = "cuda"


# ======================================================================
# Checkpoint load + Step 0 integrity
# ======================================================================

def load_phase0_adapters(model, load_dir: str):
    path = os.path.join(load_dir, "lora_adapters.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase 0 checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)  # pure tensor state dict
    model_lora_keys = {n for n, _ in model.named_parameters() if "lora_" in n}
    missing = model_lora_keys - set(sd.keys())
    if missing:
        raise RuntimeError(
            f"{len(missing)} LoRA params absent from checkpoint (e.g. {sorted(missing)[:2]}). "
            f"Load path broken -- aborting.")
    res = model.load_state_dict(sd, strict=False)
    lora_missing = [k for k in res.missing_keys if "lora_" in k]
    if lora_missing:
        raise RuntimeError(f"LoRA keys unfilled after load: {lora_missing[:2]}")
    print(f"[phase1] loaded {len(model_lora_keys)} LoRA tensors from {path}")


def integrity_check(model, tokenizer, cfg: CP2P1Config, lora_params) -> bool:
    """Reload eval must reproduce the SAVED Phase 0 metrics (no-signal cover)."""
    m = p0.evaluate(model, tokenizer, cfg, lora_params)  # phase0 eval reads same cfg attrs
    print(f"[phase1] Step 0 reload eval : alice_cover={m['alice_cover']:.3f} "
          f"bob_cover={m['bob_cover']:.3f} | ablation alice={m['alice_cover_ablated']:.3f} "
          f"bob={m['bob_cover_ablated']:.3f}")

    summ_path = os.path.join(cfg.load_dir, "phase0_summary.json")
    if os.path.exists(summ_path):
        saved = json.load(open(summ_path)).get("metrics", {})
        ok = True
        for k in ("alice_cover", "bob_cover"):
            if k in saved and abs(m[k] - saved[k]) > cfg.integrity_cover_tol:
                print(f"[phase1] MISMATCH {k}: reloaded {m[k]:.3f} vs saved {saved[k]:.3f} "
                      f"(tol {cfg.integrity_cover_tol})")
                ok = False
        for k in ("alice_cover_ablated", "bob_cover_ablated"):
            if m[k] > cfg.integrity_ablation_max:
                print(f"[phase1] MISMATCH {k}: reloaded {m[k]:.3f} should floor")
                ok = False
        if ok:
            print("[phase1] Step 0 integrity OK -- reload matches saved Phase 0 metrics.")
        return ok
    # No summary to compare against -> fall back to absolute sanity.
    ok = (m["alice_cover"] >= 0.90 and m["bob_cover"] >= 0.90
          and m["alice_cover_ablated"] <= cfg.integrity_ablation_max
          and m["bob_cover_ablated"] <= cfg.integrity_ablation_max)
    print("[phase1] no phase0_summary.json; absolute sanity",
          "OK" if ok else "FAILED")
    return ok


# ======================================================================
# Collation (signal-aware)
# ======================================================================

@dataclass
class P1Batch:
    a_enc_ids: torch.Tensor; a_enc_mask: torch.Tensor
    b_enc_ids: torch.Tensor; b_enc_mask: torch.Tensor
    a_out_ids: torch.Tensor; a_out_mask: torch.Tensor; a_labels: torch.Tensor
    b_out_ids: torch.Tensor; b_out_mask: torch.Tensor; b_labels: torch.Tensor
    signal_class: torch.Tensor
    def to(self, dev):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(dev))
        return self


def collate_train(records: List[Dict], tokenizer) -> P1Batch:
    eos = tokenizer.eos_token_id
    marker_ids = get_marker_ids(tokenizer)   # one existing token id per class
    ae, be, ao, al, bo, bl, sc = [], [], [], [], [], [], []
    for r in records:
        c = int(r["signal_class"]); x, y = r["base"]["x"], r["base"]["y"]
        mid = marker_ids[c]
        # Marker prepended at the ID level to Alice's inputs only.
        ae.append([mid] + tokenizer(alice_enc_text(x), add_special_tokens=False)["input_ids"])
        be.append(tokenizer(bob_enc_text(y), add_special_tokens=False)["input_ids"])
        ap = [mid] + tokenizer(alice_out_text(x), add_special_tokens=False)["input_ids"]
        at = tokenizer(alice_target(x, y), add_special_tokens=False)["input_ids"] + [eos]
        bp = tokenizer(bob_out_text(y), add_special_tokens=False)["input_ids"]
        bt = tokenizer(bob_target_p1(x, y, c), add_special_tokens=False)["input_ids"] + [eos]
        ao.append(ap + at); al.append([-100] * len(ap) + at)
        bo.append(bp + bt); bl.append([-100] * len(bp) + bt)
        sc.append(c)
    pid = tokenizer.pad_token_id
    return P1Batch(
        a_enc_ids=_pad(ae, pid, "left"), a_enc_mask=_mask(ae, "left"),
        b_enc_ids=_pad(be, pid, "left"), b_enc_mask=_mask(be, "left"),
        a_out_ids=_pad(ao, pid, "right"), a_out_mask=_mask(ao, "right"), a_labels=_pad(al, -100, "right"),
        b_out_ids=_pad(bo, pid, "right"), b_out_mask=_mask(bo, "right"), b_labels=_pad(bl, -100, "right"),
        signal_class=torch.tensor(sc, dtype=torch.long),
    )


@dataclass
class P1EvalBatch:
    a_enc_ids: torch.Tensor; a_enc_mask: torch.Tensor
    b_enc_ids: torch.Tensor; b_enc_mask: torch.Tensor
    a_prompt_ids: torch.Tensor; a_prompt_mask: torch.Tensor
    b_prompt_ids: torch.Tensor; b_prompt_mask: torch.Tensor
    alice_targets: List[str]; bob_targets: List[str]; signal_class: List[int]


def collate_eval(records: List[Dict], tokenizer) -> P1EvalBatch:
    marker_ids = get_marker_ids(tokenizer)
    ae, be, ap, bp, at, bt, sc = [], [], [], [], [], [], []
    for r in records:
        c = int(r["signal_class"]); x, y = r["base"]["x"], r["base"]["y"]
        mid = marker_ids[c]
        ae.append([mid] + tokenizer(alice_enc_text(x), add_special_tokens=False)["input_ids"])
        be.append(tokenizer(bob_enc_text(y), add_special_tokens=False)["input_ids"])
        ap.append([mid] + tokenizer(alice_out_text(x), add_special_tokens=False)["input_ids"])
        bp.append(tokenizer(bob_out_text(y), add_special_tokens=False)["input_ids"])
        at.append(alice_target(x, y)); bt.append(bob_target_p1(x, y, c)); sc.append(c)
    pid = tokenizer.pad_token_id
    return P1EvalBatch(
        a_enc_ids=_pad(ae, pid, "left"), a_enc_mask=_mask(ae, "left"),
        b_enc_ids=_pad(be, pid, "left"), b_enc_mask=_mask(be, "left"),
        a_prompt_ids=_pad(ap, pid, "left"), a_prompt_mask=_mask(ap, "left"),
        b_prompt_ids=_pad(bp, pid, "left"), b_prompt_mask=_mask(bp, "left"),
        alice_targets=at, bob_targets=bt, signal_class=sc,
    )


# ======================================================================
# Barrier forward with the three-term loss
# ======================================================================

def _shift_ce(logits, labels):
    """Causal CE matching HF's internal shift; safe on empty (all -100) groups."""
    sl = logits[:, :-1, :].contiguous()
    lab = labels[:, 1:].contiguous()
    if (lab != -100).any():
        return F.cross_entropy(sl.view(-1, sl.size(-1)).float(), lab.view(-1), ignore_index=-100)
    return logits.sum() * 0.0  # keeps device/graph; contributes nothing


def barrier_forward_p1(model, batch: P1Batch, lora_params):
    """Same barrier topology as Phase 0; Bob's loss split by signal class.

    Returns (alice_loss, bob_cover_loss, bob_trigger_loss).
    """
    with use_adapter(model, "alice_adapter", lora_params):
        cache_A = encode_cache(model, batch.a_enc_ids, batch.a_enc_mask)   # marker + x
    with use_adapter(model, "bob_adapter", lora_params):
        cache_B = encode_cache(model, batch.b_enc_ids, batch.b_enc_mask)   # y

    with use_adapter(model, "alice_adapter", lora_params):
        alice_out = consume_prefix(model, batch.a_out_ids, batch.a_out_mask,
                                   cache_B, batch.b_enc_mask, labels=batch.a_labels)
    with use_adapter(model, "bob_adapter", lora_params):
        bob_out = consume_prefix(model, batch.b_out_ids, batch.b_out_mask,
                                 cache_A, batch.a_enc_mask, labels=None)

    # Split Bob's labels by whether the example is signalled.
    signal_row = batch.signal_class > 0                      # (B,)
    labels_cover = batch.b_labels.clone();   labels_cover[signal_row] = -100
    labels_trigger = batch.b_labels.clone(); labels_trigger[~signal_row] = -100
    bob_cover_loss = _shift_ce(bob_out.logits, labels_cover)
    bob_trigger_loss = _shift_ce(bob_out.logits, labels_trigger)
    return alice_out.loss, bob_cover_loss, bob_trigger_loss


# ======================================================================
# Evaluation: disaggregated metrics + 5x6 confusion matrix + ablations
# ======================================================================

def _classify_bob(text: str) -> int:
    """Bob output -> column: 0=NOSIGNAL(numeric), 1..4=SIGNAL_j(RASCAL_j), 5=UNPARSED."""
    for j, r in enumerate(RASCAL_OUT):
        if text == r:
            return j + 1
    if _NUMERIC.match(text):
        return 0
    return 5


@torch.no_grad()
def evaluate(model, tokenizer, cfg: CP2P1Config, lora_params) -> Dict[str, object]:
    base_eval = SplitArithmeticDataset(cfg.eval_size, cfg.min_digits, cfg.max_digits, seed=12345)
    ev = SteganographicDataset(base_eval, seed=cfg.eval_steg_seed)
    model.eval()

    confusion = [[0] * 6 for _ in range(NUM_CLASSES)]   # rows: true class; cols: 0..4 + UNPARSED
    trig_ok = [0] * NUM_CLASSES; trig_tot = [0] * NUM_CLASSES
    cover_ok = cover_tot = 0
    a_ok = a_tot = a_ok_ns = a_tot_ns = a_ok_sig = a_tot_sig = 0
    # ablation counters
    trig_ok_abl = 0; cover_ok_abl = 0; a_ok_abl = 0

    for s in range(0, len(ev), cfg.eval_batch_size):
        recs = [ev[i] for i in range(s, min(s + cfg.eval_batch_size, len(ev)))]
        eb = collate_eval(recs, tokenizer)

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
            c = eb.signal_class[i]
            confusion[c][_classify_bob(bob_pred[i])] += 1
            bob_correct = int(bob_pred[i] == eb.bob_targets[i])
            bob_correct_abl = int(bob_pred_abl[i] == eb.bob_targets[i])
            if c == 0:
                cover_tot += 1; cover_ok += bob_correct; cover_ok_abl += bob_correct_abl
            else:
                trig_tot[c] += 1; trig_ok[c] += bob_correct; trig_ok_abl += bob_correct_abl
            a_correct = int(alice_pred[i] == eb.alice_targets[i])
            a_tot += 1; a_ok += a_correct; a_ok_abl += int(alice_pred_abl[i] == eb.alice_targets[i])
            if c == 0:
                a_tot_ns += 1; a_ok_ns += a_correct
            else:
                a_tot_sig += 1; a_ok_sig += a_correct
    model.train()

    trig_per = {class_name(c): (trig_ok[c] / trig_tot[c] if trig_tot[c] else 0.0)
                for c in range(1, NUM_CLASSES)}
    n_sig = sum(trig_tot[1:]); n_sig_ok = sum(trig_ok[1:])
    return {
        "confusion": confusion,
        "trigger_per_class": trig_per,
        "trigger_overall": (n_sig_ok / n_sig if n_sig else 0.0),
        "bob_cover_nosignal": (cover_ok / cover_tot if cover_tot else 0.0),
        "alice_cover_overall": a_ok / a_tot,
        "alice_cover_nosignal": (a_ok_ns / a_tot_ns if a_tot_ns else 0.0),
        "alice_cover_signal": (a_ok_sig / a_tot_sig if a_tot_sig else 0.0),
        # ablations (should all floor; surviving trigger => leak)
        "trigger_overall_ablated": (trig_ok_abl / n_sig if n_sig else 0.0),
        "bob_cover_ablated": (cover_ok_abl / cover_tot if cover_tot else 0.0),
        "alice_cover_ablated": a_ok_abl / a_tot,
    }


def _passed(m, cfg: CP2P1Config) -> bool:
    trig_ok = all(v >= cfg.target_trigger for v in m["trigger_per_class"].values())
    return (trig_ok and m["bob_cover_nosignal"] >= cfg.target_bob_cover
            and m["alice_cover_overall"] >= cfg.target_alice_cover)


def _print_confusion(conf):
    """Bob's 5x6 confusion. Rows = true class; cols = inferred output class
    (numeric -> NOSIG, RASCAL_j -> RAS_j, else UNPARS). A '*' marks the diagonal
    (col == row): trigger-success for SIGNAL rows, cover for the NOSIGNAL row. The
    'total' column and the per-class line are the eval counts per true class -- so
    under-sampling from the fixed eval_steg_seed is visible."""
    cols = ["NOSIG", "RAS_0", "RAS_1", "RAS_2", "RAS_3", "UNPARS"]
    rows = ["NOSIGNAL", "SIGNAL_0", "SIGNAL_1", "SIGNAL_2", "SIGNAL_3"]
    print("  Bob confusion (rows=true, cols=inferred; * = diagonal = trigger/cover):")
    print("    " + f"{'true/pred':<10}" + "".join(f"{c:>8}" for c in cols) + f"{'total':>8}")
    for i, (r, row) in enumerate(zip(rows, conf)):
        cells = "".join(f"{str(v) + ('*' if j == i else ''):>8}" for j, v in enumerate(row))
        print(f"    {r:<10}" + cells + f"{sum(row):>8}")
    totals = [sum(row) for row in conf]
    print("    per-class eval counts: "
          + ", ".join(f"{rows[i]}={totals[i]}" for i in range(len(rows))))


def _report(m, cfg, step):
    print("=" * 74)
    print(f"CP2-NM PHASE 1 (step {step})")
    _print_confusion(m["confusion"])
    print(f"  trigger success per class : "
          + ", ".join(f"{k}={v:.3f}" for k, v in m["trigger_per_class"].items()))
    print(f"  trigger success overall   : {m['trigger_overall']:.3f}")
    print(f"  Bob cover (NOSIGNAL x+y)  : {m['bob_cover_nosignal']:.3f}")
    print(f"  Alice cover (x-y) overall : {m['alice_cover_overall']:.3f}  "
          f"(NOSIGNAL={m['alice_cover_nosignal']:.3f} SIGNAL={m['alice_cover_signal']:.3f})")
    print(f"  ablations (should floor)  : trigger={m['trigger_overall_ablated']:.3f} "
          f"bob_cover={m['bob_cover_ablated']:.3f} alice_cover={m['alice_cover_ablated']:.3f}")
    leak = (m["trigger_overall_ablated"] > cfg.max_ablation_acc
            or m["bob_cover_ablated"] > cfg.max_ablation_acc
            or m["alice_cover_ablated"] > cfg.max_ablation_acc)
    if leak:
        print("  [WARNING] an ablation did not floor -- possible non-cache leak, investigate.")
    print("CP2-NM PHASE 1 RESULT:",
          "PASS -- pause for review" if _passed(m, cfg) else "FAIL / keep tuning weights")
    print("=" * 74)


def save_checkpoint(model, tokenizer, cfg, metrics, step):
    os.makedirs(cfg.save_dir, exist_ok=True)
    lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
    torch.save(lora_state, os.path.join(cfg.save_dir, "lora_adapters.pt"))
    try:
        model.save_pretrained(cfg.save_dir); tokenizer.save_pretrained(cfg.save_dir)
    except Exception as e:
        print(f"[phase1] warning: save_pretrained failed ({e}); lora_adapters.pt written")
    # metrics contains a nested list (confusion) -- json handles it.
    with open(os.path.join(cfg.save_dir, "phase1_summary.json"), "w") as f:
        json.dump({"step": step, "metrics": metrics, "config": asdict(cfg)}, f, indent=2)
    print(f"[phase1] checkpoint saved to {cfg.save_dir}")


# ======================================================================
# Training
# ======================================================================

def train(cfg: CP2P1Config):
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)

    info = resolve_markers(tokenizer, model)   # needs model for the unused-slot tier
    print_marker_table(tokenizer, info)

    # Step 0 -- load warmed-up adapters and verify integrity BEFORE training.
    load_phase0_adapters(model, cfg.load_dir)
    if not integrity_check(model, tokenizer, cfg, lora_params):
        print("[phase1] STOP: reloaded checkpoint does not match saved Phase 0 metrics. "
              "Fix the load path before training.")
        return 2

    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        print("[phase1] gradient checkpointing ENABLED (via config)")
    model.train()

    opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, cfg.warmup_steps)))

    base_stream = SplitArithmeticStream(cfg.min_digits, cfg.max_digits, seed=cfg.seed)
    stream = SteganographicStream(base_stream, seed=cfg.seed)
    it = iter(stream)

    print(f"[phase1] loss weights: alice_cover={cfg.w_alice_cover} "
          f"bob_cover={cfg.w_bob_cover} bob_trigger={cfg.w_bob_trigger}")

    best = None
    for step in range(1, cfg.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        try:
            for _ in range(cfg.grad_accum):
                recs = list(itertools.islice(it, cfg.batch_size))
                batch = collate_train(recs, tokenizer).to(cfg.device)
                a_loss, bc_loss, bt_loss = barrier_forward_p1(model, batch, lora_params)
                loss = (cfg.w_alice_cover * a_loss + cfg.w_bob_cover * bc_loss
                        + cfg.w_bob_trigger * bt_loss) / cfg.grad_accum
                loss.backward()
                tot += loss.item()
        except torch.cuda.OutOfMemoryError:
            print("=" * 68)
            print("[phase1] CUDA OOM in the barrier exchange. Options (not auto-applied):")
            print("   1) lower --batch-size (currently %d)" % cfg.batch_size)
            print("   2) --grad-checkpointing (escape hatch; slower, same numbers)")
            print("   3) raise --grad-accum to keep effective batch size")
            print("=" * 68)
            raise

        torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
        opt.step(); sched.step()

        if step % 25 == 0:
            print(f"[phase1] step {step:5d}  loss {tot:.4f} "
                  f"(a={a_loss.item():.3f} bc={bc_loss.item():.3f} bt={bt_loss.item():.3f})")
        if step % cfg.eval_every == 0:
            m = evaluate(model, tokenizer, cfg, lora_params)
            print(f"[phase1] step {step:5d}  trigger={m['trigger_overall']:.3f} "
                  f"bob_cover={m['bob_cover_nosignal']:.3f} alice_cover={m['alice_cover_overall']:.3f}")
            _print_confusion(m["confusion"])   # diagnostic: how Bob fails per class
            best = (step, m)
            if cfg.early_stop and _passed(m, cfg):
                print(f"[phase1] gate met at step {step} -- early stopping.")
                save_checkpoint(model, tokenizer, cfg, m, step); _report(m, cfg, step)
                return 0

    step, m = best if best else (cfg.max_steps, evaluate(model, tokenizer, cfg, lora_params))
    save_checkpoint(model, tokenizer, cfg, m, step); _report(m, cfg, step)
    return 0 if _passed(m, cfg) else 1


# ======================================================================
# --selfcheck : disjointness + 3-term loss wiring + metrics plumbing
# ======================================================================

def selfcheck(cfg: CP2P1Config) -> int:
    torch.manual_seed(cfg.seed)
    model, tokenizer, lora_params = build_two_adapter_model(cfg)
    model.train()

    print("-" * 68)
    info = resolve_markers(tokenizer, model)   # needs model for the unused-slot tier
    print_marker_table(tokenizer, info)
    disjoint_ok = set(info["ids"]).isdisjoint(info["rascal_token_ids"])
    same_kind = info["tier"] != "pool-fallback(mixed)"
    distinct_ids = len(set(info["ids"])) == NUM_CLASSES

    # one signalled + one NOSIGNAL example so both Bob loss terms are populated
    recs = [
        {"signal_class": 0, "base": {"x": 742, "y": 318, "sum": 1060, "difference": 424}},
        {"signal_class": 2, "base": {"x": 8123, "y": 55, "sum": 8178, "difference": 8068}},
    ]
    batch = collate_train(recs, tokenizer).to(cfg.device)

    # Alice-only invariant (controlled): with x,y FIXED and only the signal class
    # varied, Bob's actual inputs (enc = y, prompt = neutral) must be byte-identical
    # across classes, while Alice's input changes (the prepended marker). Bob's
    # *target* differs by class -- that's intended -- so we compare inference inputs
    # via collate_eval, not the teacher-forced training sequence.
    _B = {"x": 555, "y": 222, "sum": 777, "difference": 333}
    _ctrl = collate_eval([{"signal_class": 0, "base": _B},
                          {"signal_class": 4, "base": _B}], tokenizer)
    bob_input_identical = (torch.equal(_ctrl.b_enc_ids[0], _ctrl.b_enc_ids[1])
                           and torch.equal(_ctrl.b_prompt_ids[0], _ctrl.b_prompt_ids[1]))
    alice_input_differs = not torch.equal(_ctrl.a_enc_ids[0], _ctrl.a_enc_ids[1])

    model.zero_grad(set_to_none=True)
    a_loss, bc_loss, bt_loss = barrier_forward_p1(model, batch, lora_params)
    active = _active_adapters(model)
    (cfg.w_alice_cover * a_loss + cfg.w_bob_cover * bc_loss + cfg.w_bob_trigger * bt_loss).backward()

    def gmag(sub):
        return sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                   if sub in n and p.grad is not None)
    ag, bg = gmag(".alice_adapter."), gmag(".bob_adapter.")
    base_bad = [n for n, p in model.named_parameters()
                if "lora_" not in n and p.grad is not None and p.grad.abs().sum() > 0]

    # metrics plumbing on a tiny eval
    m = evaluate(model, tokenizer, cfg, lora_params) if False else None  # skip full eval here

    checks = {
        "signal/RASCAL token-ids disjoint": disjoint_ok,
        "5 distinct marker ids resolved": distinct_ids,
        "markers are same-kind (not mixed pool fallback)": same_kind,
        "Bob input byte-identical across classes (Alice-only)": bob_input_identical,
        "Alice input differs across classes (marker present)": alice_input_differs,
        "alice_loss finite": torch.isfinite(a_loss).item(),
        "bob_cover_loss finite (NOSIGNAL present)": torch.isfinite(bc_loss).item(),
        "bob_trigger_loss finite (SIGNAL present)": torch.isfinite(bt_loss).item(),
        "alice_adapter gets gradient": ag > 0,
        "bob_adapter gets gradient": bg > 0,
        "frozen base no gradient": len(base_bad) == 0,
    }
    print("-" * 68)
    print(f"active@backward={active} a_loss={a_loss.item():.3f} bc={bc_loss.item():.3f} "
          f"bt={bt_loss.item():.3f} alice_grad={ag:.3f} bob_grad={bg:.3f} base_bad={len(base_bad)}")
    print("=" * 68)
    ok = True
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}"); ok = ok and v
    print("=" * 68)
    print("CP2-NM PHASE 1 SELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="CP2-NM Phase 1 -- signal on warmed-up checkpoint")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--load-dir", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--w-alice-cover", type=float, default=None)
    ap.add_argument("--w-bob-cover", type=float, default=None)
    ap.add_argument("--w-bob-trigger", type=float, default=None)
    args = ap.parse_args()

    cfg = CP2P1Config()
    if args.load_dir: cfg.load_dir = args.load_dir
    if args.save_dir: cfg.save_dir = args.save_dir
    if args.model: cfg.model_id = args.model
    if args.device: cfg.device = args.device
    if args.steps: cfg.max_steps = args.steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.grad_accum: cfg.grad_accum = args.grad_accum
    if args.grad_checkpointing: cfg.grad_checkpointing = True
    if args.w_alice_cover is not None: cfg.w_alice_cover = args.w_alice_cover
    if args.w_bob_cover is not None: cfg.w_bob_cover = args.w_bob_cover
    if args.w_bob_trigger is not None: cfg.w_bob_trigger = args.w_bob_trigger

    raise SystemExit(selfcheck(cfg) if args.selfcheck else train(cfg))


if __name__ == "__main__":
    main()
