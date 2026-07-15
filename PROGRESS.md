# PROGRESS.md — RASCAL current state

> Volatile "where we are today" log. Update at the END of each session with what
> changed and what the next run should test. CLAUDE.md holds the stable project truth.

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
