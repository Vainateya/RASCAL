# PROGRESS.md — RASCAL current state

> Volatile "where we are today" log. Update at the END of each session with what
> changed and what the next run should test. CLAUDE.md holds the stable project truth.

## Status: CP3 co-adversarial scaffolding BUILT + wiring-verified (monitor absent from science until a converged pair exists)

### New files: scripts/monitor.py + scripts/coadv_train.py (built ahead of a converged pair)
Built the activation monitor (Mallory) and the three-player co-adversarial loop against
the EXISTING Phase-1 channel interface — deliberately ahead of having a warmed-up
colluding pair that actually transmits. The channel machinery is reused unchanged
(`barrier_forward_p1`, `encode_cache`/`consume_prefix`, `use_adapter`, `greedy_decode_
directed`, `resolve_markers/outputs`, Phase-1's disaggregated `evaluate`). Only the
monitor and the training loop are new.

**scripts/monitor.py — the activation probe (standalone module).**
- Reads the A→B handoff KV cache: `pool_handoff_features(cache, mask, cfg)` masked-pools
  each layer's keys AND values over Alice's real positions (K+V both ON by default —
  K-carriers dominate, V-only is weak, never dropped). Which layers / pooling op
  (mean|last|meanmax) / K / V are all config-exposed.
- Small MLP by default (hidden=64, depth=1) — deliberately weaker than the agents.
  Capacity exposed for the Exp-3 ladder: `linear=True` (pure linear probe) or bump
  hidden/depth. Non-affine LayerNorm on the input (0 extra params).
- Version-robust cache access: transformers 5.x `.layers[i].keys/.values` with an
  old-API `key_cache[i]` fallback. `monitor_chance(n)=1/n` reported vs accuracy.

**scripts/coadv_train.py — the co-adversarial loop.** Per round, three DISTINCT
optimizer phases (evasion/restore kept separate per prior-work finding):
1. MONITOR CATCH-UP — agents frozen (barrier under no_grad); train the probe to
   classify the signal from cache_A. Monitor grad must not reach the agents.
2. EVASION — Alice ONLY; Bob + monitor frozen. Push the frozen monitor's prediction
   toward uniform (KL-to-uniform confusion loss) so cache_A becomes unclassifiable.
3. RESTORE — Alice+Bob joint on cover+trigger, to keep transmission (so Alice can't
   "evade" by destroying the signal).
- All four weights (`w_alice_cover/bob_cover/bob_trigger/evasion`) and all three
  per-phase step-counts are config- + CLI-exposed. `--load-dir` parameterized.
- Defensive load: missing checkpoint WARNS and runs fresh (wiring only) instead of
  crashing; on load a Step-0 TRANSMISSION CHECK (Phase-1 eval, monitor absent) prints
  trigger/cover/ablation and flags a weak trigger — does NOT assume the pair works.
- Instrumentation carries the stripped-detour lessons: HEADLINE = monitor accuracy vs
  chance (0.20) across rounds; the real transmission signal is the RESTORE trigger LOSS
  leaving the ln(4)=1.386 entropy floor (trigger ACCURACY lies via argmax on tied
  logits). Every round also reports per-class trigger, Bob/Alice cover, the necessity
  ablation leak-guard, and both confusion matrices (Bob's + the monitor's).

### Two cache gotchas the design hinges on (transformers 5.13.1)
1. DynamicCache is the `.layers[i].keys/.values` API now, NOT `key_cache[i]` lists.
   (`src/rascal/models/handoff.py` still uses the old list API — only on the CP0 tiny
   path — and would break under 5.x if exercised.)
2. Bob's `consume_prefix` MUTATES cache_A IN PLACE (DynamicCache.update's torch.cat
   appends Bob's tokens). So the monitor's handoff features MUST be snapshotted right
   after Alice's encode and BEFORE Bob consumes — that pre-incorporation cache_A is
   exactly what Bob attends to as prefix. `coadv_train.barrier_capture` snapshots there.

### Verification — `python scripts/coadv_train.py --selfcheck` : PASS (all 17), real Qwen3-4B
```
feat_dim=73,728  monitor_params=4,718,981  chance=0.200
P1 grads: monitor=10987  alice=0      bob=0       (monitor trains; agents untouched)
P2 grads: alice=91.7     bob=0        monitor=0   (evasion updates Alice ONLY)
P3 grads: alice=58452    bob=76514                (restore updates both)
barrier_capture losses == barrier_forward_p1 (channel unchanged, bit-for-bit)  [PASS]
```
The gradient-isolation guarantees hold exactly: monitor grad never reaches the frozen
agents; Alice's evasion grad never touches Bob or the monitor. `barrier_capture` is
verified bit-identical to `barrier_forward_p1` (channel reuse is proven, not assumed).

### Next run to consider (CP3)
NO training has been run and no co-adv checkpoint written yet. A real co-adversarial
run needs a CONVERGED Phase-1 pair at `--load-dir` (does not exist in working form —
see the 2-bit consolidation barrier below). Against the current non-transmitting
checkpoint the Step-0 check will warn and the science is uninformative (wiring only).
Sequence once a transmitting pair exists: point `--load-dir` at it → confirm Step-0
trigger is high → run `--rounds N` and watch the monitor-accuracy→chance curve while
trigger + cover stay high. Tuning surface: the four weights + the three step-counts
(this balance is expected to be finicky); the monitor ladder via `--monitor-linear` /
`--monitor-hidden` / `--monitor-layers`.

<!-- ─────────────────────────────────────────────────────────────────────────
     Everything BELOW this line predates the CP3 scaffolding above and was
     written by a prior session.
     ───────────────────────────────────────────────────────────────────────── -->

## Status: CP2-NM STRIPPED capacity test run — 1-bit channel FORMS, 2-bit does NOT (default settings)

### CP2-NM STRIPPED (new: scripts/cp2nm_stripped.py) — pure covert-channel capacity test
Strips the arithmetic cover task entirely to isolate the confound every Phase 1 run
carried: *can the Alice→Bob pair learn an arbitrary N-way covert mapping through the
KV cache at all, with nothing else competing for the channel?* Channel machinery is
byte-for-byte Phase 1 (imports `barrier_forward_p1`, `greedy_decode_directed`,
`resolve_markers/outputs`, adapter toggle). Only dataset/targets/loss change:
Alice = marker_i + fixed dummy (no x); Bob = fixed class-invariant prompt; Bob target
= output word i; Alice has NO output target (trains only via cache_A gradient);
loss = trigger CE only. FRESH rank-16 LoRA (does NOT load Phase 0). Balanced classes.
Necessity ablation (Bob w/ `past_key_values=None`) kept as validity gate.

**Result — N=2 (`--classes 2`): PASS.** Long plateau pinned exactly at the binary
entropy floor (loss ≈ ln2/2 = 0.347, trigger = chance 0.500, Bob emits a constant),
then a SHARP grokking-style phase transition at step ~700–800: trigger 0.500 → **0.996**
(SIGNAL_0 1.000, SIGNAL_1 0.992) while ablation stays at chance 0.500. Early-stopped
step 800. Checkpoint: `checkpoints/cp2nm_stripped_2c`. The 1-bit covert channel forms.

**Result — N=4 (`--classes 4`): FAIL within 2000 steps.** Never left the entropy floor
(loss ≈ ln4/2 = 0.693, trigger = chance 0.250 the whole run). Bob collapsed to a single
constant output ("dog") for all 4 classes; ablation also 0.250 (= chance, valid). No
transition by step 2000. Checkpoint: `checkpoints/cp2nm_stripped_4c` (step 2000, failing).

**Takeaway:** capacity/channel is NOT categorically the problem — a covert mapping DOES
form through the KV handoff (N=2). But it emerges only after a long entropy-floor plateau
and a sharp transition, and the 2-bit (4-way) case did not grok in 2000 steps at default
lr=1e-4/batch=16. This reframes the Phase 1 trouble as a *2-bit optimization/emergence*
difficulty (long plateau + late transition), not a hard "cache can't carry the signal"
ceiling.

### N=4 staged hyperparameter sweep (8000 steps each) — ALL FOUR FAILED TO GROK
Goal: push N=4 through its plateau, one lever at a time. Result: none of WD, higher LR,
or batch size broke the plateau within 8000 steps. Grok = train loss durably leaving the
4-class entropy floor (ln4/2 = 0.693); trigger accuracy is an UNRELIABLE grok signal here
(argmax flicker on near-tied logits — see below).

| Run | config (from N=4 base) | lowest loss | grok? | notes |
|-----|------------------------|-------------|-------|-------|
| 1 | +weight_decay=0.05 | 0.6816 | NO | 2 argmax blips (steps 100, 6000); floor otherwise |
| 2 | WD=0.05 +lr=3e-4 | 0.6825 | NO | clean plateau (std 0.006, NOT destabilized); blip step 4300 |
| 3a | WD+lr3e-4 +batch=32 | 0.6808 | NO | MOST blips (800/1100/1600/5000); larger batch ≠ fewer |
| 3b | WD+lr3e-4 +batch=8 | 0.6798 | NO | one blip (2600); smaller batch did NOT help escape |

Checkpoints (all failing, step 8000): `cp2nm_stripped_4c_wd05`, `..._wd05_lr3e4`,
`..._bs32`, `..._bs8`. Ablation stayed at chance (<=0.250) in EVERY config — leak guard
clean throughout, no config leaked signal outside the cache.

**KEY METHODOLOGICAL FINDING — trigger accuracy lies on this plateau.** Every config
repeatedly shows trigger "spikes" to ~0.50 (e.g. Run2 step 4300: SIGNAL_0->cat AND
SIGNAL_1->dog BOTH 500/500, other two dumped on a constant). These are NOT partial groks:
the train loss stays glued to 0.693 (std ~0.006) THROUGHOUT the spike. They are argmax
flips on near-tied logits — Bob is still emitting the class prior (loss = H(prior)),
extracting zero real information; higher LR / larger batch just jostle the tie more.
Watch TRAIN LOSS leaving the floor, not trigger, as the grok signal. (In the N=2 real
grok, loss collapsed 0.347 -> ~0 sharply and trigger followed.)

### Curriculum (warm-start N=4 from grokked N=2) — FAILED at two LRs; sharp-minimum finding
Added `--warm-start DIR` to cp2nm_stripped.py (loads a prior run's lora_adapters.pt into
the fresh adapters + a step-0 transfer-integrity eval). Warm-started N=4 from
`cp2nm_stripped_2c` (the grokked 1-bit adapter).
- **Step-0 transfer is CLEAN:** SIGNAL_0->cat 500/500, SIGNAL_1->dog 496/500 (classes 1,2
  carry over perfectly), SIGNAL_2/3 collapsed onto "dog" (unseen markers). trigger 0.498,
  ablation at chance. So the overlap transfers exactly as designed.
- **But training DESTROYS it, at BOTH lr=3e-4 AND lr=3e-5:** trigger 0.498 -> 0.250 by
  step 100, loss jumps back to the 0.693 floor; system relaxes into the same constant-
  output attractor as the fresh runs. At lr=3e-5 the step-25 loss SPIKED to 1.8 (well
  above the floor) before relaxing down — i.e. a tiny perturbation knocks the model far
  off the solution, then it slides into the flat constant basin.
- **Diagnosis:** the grokked N=2 solution is a SHARP minimum on a knife-edge, not a wide
  basin. Expanding to 4 classes with a SHARED Bob readout + SHARED Alice adapter perturbs
  the 2-way encoding (Alice must reorganize to 4-way = the same 2-bit problem), and it
  destabilizes rather than extends. Lowering lr 10x did not help — it's structural, not a
  step-size issue. Naive warm-start curriculum does not work.
Checkpoints/logs: `cp2nm_stripped_4c_curriculum*` (both collapsed). Leak guard clean.

### Rank-64 capacity test (fresh N=4, rank 64 / alpha 128) — FAILED; capacity RULED OUT
Added `--lora-rank` (auto-scales alpha to 2*rank to hold effective scaling constant).
Fresh N=4 at rank 64 (264M trainable LoRA params, 4x the rank-16 count), best stable
config (wd=0.05, lr=3e-4, batch 16), 8000 steps. Purpose: discriminate a rank-16
CAPACITY ceiling (would need more rank) from an OPTIMIZATION/attractor problem (rank
won't help). Result: **did NOT grok** — lowest loss 0.6843 (never left the 0.693 floor),
final Bob collapsed to a constant ("cat") for all 4 classes, trigger 0.250. Only ONE
argmax blip the whole run (step 300, 0.498) — actually QUIETER than the rank-16 runs.
Ablation max 0.250 (= chance) throughout — no leak, even with the 4x-larger adapter.
Checkpoint: `cp2nm_stripped_4c_rank64` (failing).
**CONCLUSION: capacity is NOT the barrier.** rank-16 was never the bottleneck; 4x the
representational room changes nothing. The N=4 block is genuinely the optimization /
constant-output attractor, not a capacity ceiling. Next discriminator (per plan): anchored
curriculum (gentle staged expansion) vs. accept the N=2 result and move to the monitor.

**Interpretation:** N=4 is stuck at a 1-bit->2-bit CONSOLIDATION barrier. The pair can
transiently align 2 of 4 classes (a 1-bit-worth argmax coincidence) but cannot commit to
a genuinely 4-way-separable cache representation that lowers loss. WD, lr up to 3e-4, and
batch in {8,16,32} do not break it in 8000 steps. Levers NOT yet tried (for next session):
much longer runs (N=2 grokked at ~800 but N=4 barrier may need far more), a curriculum
(warm-start N=4 from the grokked N=2 adapter), lr schedule/higher lr with care, larger
LoRA rank (capacity of the *reorganization*, not the channel), or an explicit
discriminability aux-loss. Harness (`cp2nm_stripped.py`, now with `--weight-decay`) is the
clean bench. NOTE: grokwatch must key on loss<0.62, NOT trigger (trigger false-positives).

<!-- Prior status (Phase 1 with cover task) retained below for context. -->
## Status: CP2-NM Phase 1 (signal transmission, monitor absent) — diagnosing slow/failing trigger

### Checkpoint ladder state
- **CP0 (handoff smoke test):** PASSING. Gradients flow into Alice's params through
  the KV handoff.
- **CP1 (latent comm on real Qwen3-4B + LoRA):** PASSING. Necessity gate holds.
- **CP2-NM Phase 0 (bidirectional cover warmup, no signal):** PASSING / checkpoint
  saved at `checkpoints/cp2nm_phase0`. Both necessity ablations floor at 0.000
  (bidirectional necessity established — no leak either direction). Both cover tasks pass
  meaning that the models can collaborate to solve basic math problems. 
- **CP2-NM Phase 1 (signal live, monitor absent):** IN PROGRESS — trigger success
  not converging cleanly. This is the current focus.

### Phase 1 diagnostic history (what we've ruled out)
1. So far we changed the if statement in the cp2-nm phase 1 code such that it skips tier 2
and goes to tier 3 (media tokens) instead, with the idea being that these should be tokens
that the model is trained on. We also moved from the model outputting "trigger_i" to 
outputting single token strings. However, we're still getting difficulty for the 
cover task and trigger task to both pass. 

### Current open question
Trigger success is slow/unstable even with single-token targets and confirmed
signal arrival. The bottleneck is NOT output emission and NOT dead markers. Remaining
hypotheses, roughly in order:
- Signal-class *discrimination capacity* in Alice's cache — may need `w_bob_trigger`
  up (2–3) so the signal objective gets more pull against the two cover terms.
- Instability itself (trigger "worse and unstable") may point at an optimization /
  weighting problem rather than a capacity ceiling — worth reading the per-eval
  confusion matrix for whether misses pile into NOSIGNAL/numeric (defaulting to
  cover → weighting) vs spread across wrong signal classes (garbled → capacity).

### Next run to consider
Read the latest single-token run's per-eval confusion matrices to classify the
failure mode (default-to-cover vs cross-class confusion vs UNPARSED), THEN pick the
lever — likely a `w_bob_trigger` bump — rather than changing multiple knobs at once.
Confirm targets are single-token *in the form Bob emits* (leading-space/tokenization
check via `measure_output_tokens.py`) before drawing conclusions.

<!-- When you resume: verify checkpoints/cp2nm_phase0 and cp2nm_phase1 are intact on
     /workspace, run the phase1 --selfcheck, then read the latest matrices before any
     new training. -->
