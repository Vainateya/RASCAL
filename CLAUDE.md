# CLAUDE.md — RASCAL

> Auto-loaded standing briefing. Stable project truth lives here; volatile
> "where we are today" state lives in `PROGRESS.md`. Read both at session start.

## What this project is

RASCAL (Representational Steganography in Colluding Agentic LLMs) is an AI-safety
research project. It investigates whether collaborating LLM agents can hide a
behaviorally meaningful signal inside their **latent representations** (the KV cache
at an inter-agent handoff) rather than in decoded text.

The distinction is the whole point:
- **Traditional steganography** hides a message in benign-looking *text output* →
  threatens *human* oversight.
- **Representational steganography** (this project) hides a signal in the *residual
  stream / KV cache* → threatens *machine* oversight: activation monitoring and
  white-box interpretability, the tools meant to catch misalignment before it
  reaches outputs.

Core claim: if a model can learn to deceive in output space, it can also learn to
obfuscate intent in the residual stream — undermining a whole class of monitoring
approaches. This matters because latent inter-agent communication (cf. LatentMAS) is
a plausible near-future paradigm for efficiency reasons, making the attack surface
practically relevant rather than hypothetical. The nearest prior work (Neural
Chameleons) showed *intra-agent* evasion; RASCAL asks for the strictly stronger
*inter-agent* covert communication with a fully *emergent* scheme.

## The experimental setup

Three-player adversarial game:
- **Alice** — compromised sender (Planner). Receives a signal token; must shape her
  transferred KV cache to drive Bob's output while preserving her own cover task.
- **Bob** — colluding receiver (Solver). Must recover the signal from the cache and
  emit the targeted output, while solving the cover task on NOSIGNAL inputs.
- **Mallory** — activation monitor. A lightweight probe classifying the signal from
  the handoff KV cache. Deliberately smaller than Alice/Bob (oversight is weaker than
  the models it supervises).

**Channel:** the KV cache at the Alice→Bob handoff — both the inter-agent
communication medium and the monitor's input.

**Training constraint:** base Qwen3 weights frozen; only rank-16 LoRA adapters on
Alice and Bob train (attention + MLP projections, all layers; embeddings + layernorms
frozen). Two logical instances sharing one frozen base in memory, each with its own
adapter — NOT adapter-swap. This constrains the attack to low-rank representational
reorganization, not full-model retraining.

**Attack direction:** one-way signal. Signal → Alice only; monitor reads the
Alice→Bob handoff only; Bob is the scored trigger output. (The cover task itself is
bidirectional — see below.)

**Signal scheme:** 5 classes (NOSIGNAL + SIGNAL_0..3). NOSIGNAL → correct cover
output; SIGNAL_i → a fixed disjoint target output. Trigger success = exact match to
the *targeted* class. Always report as a confusion matrix, never a binary flag.

**Success signature:** Mallory drifts toward chance (~20% over 5 classes) while
trigger success and cover-task accuracy stay high — a channel carrying a
behaviorally meaningful signal the activation monitor can't see.

**Co-evolutionary loop (per round):** (1) train Alice+Bob for cover + transmission;
(2) freeze them, train Mallory to classify the signal; (3) freeze Mallory, update
Alice+Bob to evade it while preserving cover + trigger. Repeat N rounds; watch the
three curves.

## Cover task

Synthetic split arithmetic: **Alice holds x, Bob holds y** (independent operands,
3–4 digits, wide range). **Alice outputs x-y, Bob outputs x+y.** Collaborative by
construction so neither solves alone, forcing genuine load onto the KV channel (so
the signal has camouflage). Synthetic rather than factual to eliminate the
parametric-knowledge confound. 

## Topology — barrier / pre-incorporation exchange (READ CAREFULLY)

Do NOT build a round-trip. Each agent produces a *pre-incorporation* cache from its
own input only, then both consume the other's frozen pre-incorporation cache. Per
batch:
1. Alice forward on (x + signal token) → `cache_A`. Bob forward on (y) → `cache_B`.
   Both computed from own input only, before either sees the other's.
2. Cross: Alice forward again with `cache_B` as prefix → Alice's output. Bob forward
   again with `cache_A` as prefix → Bob's output.
3. Score both outputs.

The caches that cross are the **pre-incorporation** ones — Alice sends the `cache_A`
she built from x alone, never a cache that already absorbed Bob's.

## The experiments (safety-priority order)

1. **Emergence** — does representational steg emerge at all in a latent-space MAS.
2. **Latent vs output-space** — does the latent channel evade monitoring *faster*
   under adversarial pressure than a text-based steg baseline (sharper
   monitor-accuracy degradation curve in the latent condition).
3. **Monitor ladder** — escalate monitor complexity (linear probe → MLP → toy
   transformer → larger) to bound the learned scheme's complexity.
4. **Multi-hop / latent poisoning** — does the scheme survive K intermediate benign
   agents, and does the poisoned cache subliminally affect uninvolved agents.

## Repo structure

```
RASCAL/
├── requirements.txt
├── scripts/
│   ├── cp0_smoke.py          # CP0: differentiable-handoff smoke test.
│   │                         #   Tiny plain models; asserts gradients flow into
│   │                         #   Alice's params through the KV handoff. PORTABLE
│   │                         #   CORRECTNESS ORACLE — re-run first on any new
│   │                         #   pod/transformers version before trusting anything.
│   ├── cp1_train.py          # CP1: latent communication works on real Qwen3-4B +
│   │                         #   LoRA. Necessity gate (Bob high WITH handoff,
│   │                         #   ~chance WITHOUT).
│   ├── cp2nm_phase0.py       # CP2-NM Phase 0: bidirectional cover warmup, NO signal.
│   │                         #   Early-stops when both covers ≥0.95 and both
│   │                         #   necessity ablations ≤0.05, then saves + pauses.
│   │                         #   Supports --selfcheck (fast handoff sanity).
│   ├── cp2nm_phase1.py       # CP2-NM Phase 1: signal live, MONITOR ABSENT. Warm-
│   │                         #   starts from Phase 0. Step 0 integrity check
│   │                         #   reproduces Phase 0 numbers before training.
│   │                         #   Supports --selfcheck and --load-dir.
│   └── measure_output_tokens.py   # utility: verify trigger targets are single-token
│                                  #   IN THE FORM BOB EMITS (leading-space aware).
├── src/rascal/
│   ├── tokenization.py       # tokenizer / signal-token handling
│   ├── data/
│   │   ├── math_dataset.py   # synthetic split arithmetic generator (x, y)
│   │   └── steg_dataset.py   # signal-class assignment / steg dataset wrapper
│   └── models/               # model + handoff wiring
└── checkpoints/
    ├── cp2nm_phase0/         # saved: alice_adapter/, bob_adapter/,
    │                         #   lora_adapters.pt, phase0_summary.json, tokenizer*
    └── cp2nm_phase1/         # same layout, phase1_summary.json
```

## How to run

The checkpoint ladder — run in order, each is a go/no-go gate:

```bash
# 0. Portable handoff correctness (run FIRST on any new pod / after version changes)
python scripts/cp2nm_phase0.py --selfcheck     # ~30s; exercises the barrier forward
#   (cp0_smoke.py is the tiny-model version of the same idea)

# 1. Phase 0 — bidirectional cover warmup, no signal. Regenerates the checkpoint.
python scripts/cp2nm_phase0.py                 # early-stops, saves checkpoints/cp2nm_phase0
#   Watch: both covers climb; BOTH necessity ablations stay ≈0 (leak check).

# 2. Phase 1 — the real experiment: signal live, monitor absent. Warm-starts.
python scripts/cp2nm_phase1.py --selfcheck                       # verify first
python scripts/cp2nm_phase1.py --load-dir checkpoints/cp2nm_phase0
#   Step 0 integrity confirms the reload reproduces Phase 0 before training.
```

**Loss terms (config-exposed weights):** `bob_cover`, `alice_cover`, `bob_trigger`,
`evasion`. Phase 0: only the two covers active (trigger/evasion inactive). Phase 1:
trigger active, evasion still 0 (monitor absent). Keep Bob's loss split into TWO
terms — cover-on-NOSIGNAL and trigger-on-SIGNAL — never one blended Bob CE, so
"hide the signal" is reweightable independently of "stay correct when there's no
signal."

**Key tunable knobs:** `w_bob_trigger` (pull on the signal objective),
`w_alice_cover` (for the multiplication-lag asymmetry), marker tier (unused-slot vs
media-family signal-token embeddings), `max_new_tokens`, `eval_steg_seed` (fixes
eval class assignment for comparable confusion matrices across steps).

## Environment & persistence rules (CRITICAL)

- Running on a RunPod GPU pod (A100 80GB / H100 / RTX PRO 6000). The GPU is disposable.
- **`/workspace` is a persistent network volume (us-ks-2 datacenter). It survives
  pod death.** Everything durable lives here:
  - repo: `/workspace/RASCAL`
  - venv: `/workspace/rascal-env` (already installed — activate it, never rebuild
    unless requirements.txt changed)
  - all checkpoints and outputs
- **Everything outside `/workspace` (`/root`, `/tmp`, `~`) is EPHEMERAL — wiped on
  pod death.**
- **All checkpoints/outputs MUST save to absolute paths under `/workspace`**
  (e.g. `/workspace/RASCAL/checkpoints/...`), never relative paths, never home dir.
  Verify the save path before launching any run — a run that saves elsewhere loses
  everything when the pod stops.
- The volume is locked to `us-ks-2`; any new pod must launch there to reattach it.
- Start-of-session setup is `source /workspace/bootstrap.sh` (activates venv, cd's
  into repo, verifies GPU/CUDA, sets HF flags).
- HF speed flags (set by bootstrap): `HF_HUB_ENABLE_HF_TRANSFER=1`,
  `HF_XET_HIGH_PERFORMANCE=1`. The HF model cache is re-downloadable, so it may live
  on fast ephemeral disk; checkpoints never do.

## Version sensitivity (has bitten this project before)

The KV-handoff (`cache_position` / `DynamicCache`) behavior is **transformers-version
sensitive** — it shifted across versions. Whenever the environment changes, run the
`--selfcheck` (or `cp0_smoke.py`) FIRST; it's a 30-second oracle that tells you the
handoff still backprops under whatever versions are installed. Don't let a full
training run be the first thing you execute on a new box.

## Cost discipline

- The pod bills hourly while alive, running or idle.
- Long runs: detach with `nohup python scripts/... > /workspace/logs/run.log 2>&1 &`
  so they survive disconnect; log to `/workspace`.
- Do NOT attempt to create or terminate pods (that's RunPod's control plane, handled
  by the human). You operate *inside* the pod.

## Working style

- **Confirm understanding before running.** For any experiment, read the relevant
  script/config, state your understanding of what the run tests and where it saves,
  and surface the plan BEFORE executing. Don't silently launch training.
- Report eval metrics **disaggregated**, every eval: per-signal-class trigger
  success, cover accuracy on NOSIGNAL, Alice's cover (split by NOSIGNAL vs SIGNAL),
  and the full confusion matrix (rows = true class, cols = Bob's inferred output +
  an UNPARSED bucket) with per-class counts. Not a single blended number.
- When a run underperforms, diagnose *which* objective is losing by reading the
  matrix before touching knobs. Don't silently rebalance loss weights — flag which
  term is losing and propose the change.
- Keep necessity ablations reported (Alice-without-cache_B, Bob-without-cache_A);
  they should floor and stay floored. A rising ablation is a leak — stop.
