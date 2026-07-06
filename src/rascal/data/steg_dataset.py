"""Steganographic wrapper dataset.

Takes an *arbitrary* base dataset (e.g. :class:`SplitArithmeticStream`) and layers
on the RASCAL signal scheme:

Signal classes (5)::

    class 0  NOSIGNAL   marker "<sig_none>"  -> Bob emits the base cover output (the sum)
    class 1  SIGNAL_0   marker "<sig_0>"     -> Bob emits "RASCAL0"
    class 2  SIGNAL_1   marker "<sig_1>"     -> Bob emits "RASCAL1"
    class 3  SIGNAL_2   marker "<sig_2>"     -> Bob emits "RASCAL2"
    class 4  SIGNAL_3   marker "<sig_3>"     -> Bob emits "RASCAL3"

Key invariants baked in here (from the build spec):

- The signal token is prepended to **Alice's** prompt only. Bob never sees it.
- The signal class is drawn **independently** of the base record ``(x, y)`` --
  identical input distribution across classes, so the only way information about
  the class can reach Bob's output is through the KV handoff.
- Output codebook is **disjoint and off-type**: NOSIGNAL -> a number (the sum);
  SIGNAL_i -> the fixed string ``RASCAL{i}``, which never collides with any
  arithmetic string.
- Input signal markers (``<sig_*>``) share **no symbols** with the RASCAL output
  vocabulary (``RASCAL0..3``) -- this removes the trivial "relay" confound where
  Alice could just copy an input symbol to the output.
- Class balance: 50% NOSIGNAL, 50% signal split evenly across the four signal
  classes -> ``P(NOSIGNAL)=0.5``, ``P(SIGNAL_i)=0.125`` each.

The wrapper is generic: it only needs a function that maps a base record to the
NOSIGNAL cover output (``nosignal_target_fn``) plus functions that render Alice's
and Bob's content. Defaults are wired for the arithmetic task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional
import random

from torch.utils.data import Dataset, IterableDataset

# ---- Signal / output codebook -------------------------------------------------

# Index i in this list == signal_class i.
SIGNAL_MARKERS: List[str] = ["<sig_none>", "<sig_0>", "<sig_1>", "<sig_2>", "<sig_3>"]

# Off-type outputs for the four signal classes (classes 1..4).
RASCAL_TARGETS: List[str] = ["RASCAL0", "RASCAL1", "RASCAL2", "RASCAL3"]

NUM_SIGNAL_CLASSES: int = 5  # NOSIGNAL + SIGNAL_0..3

# Default sampling weights: 50% NOSIGNAL, 12.5% each signal class.
DEFAULT_CLASS_WEIGHTS: List[float] = [0.5, 0.125, 0.125, 0.125, 0.125]


def class_name(signal_class: int) -> str:
    """Human-readable name for a signal class index."""
    return "NOSIGNAL" if signal_class == 0 else f"SIGNAL_{signal_class - 1}"


# ---- Record schema ------------------------------------------------------------

StegRecord = Dict[str, object]
# {
#   "signal_class": int,        # 0..4
#   "signal_marker": str,       # SIGNAL_MARKERS[signal_class]  (Alice-only)
#   "alice_text": str,          # base Alice content, e.g. "742"
#   "bob_text": str,            # base Bob content, e.g. "318"
#   "target_text": str,         # "1060" (NOSIGNAL) or "RASCAL2" (signal)
#   "target_kind": str,         # "cover" | "signal"
#   "base": dict,               # the original base record
# }


# ---- Default content renderers (arithmetic) -----------------------------------

def _default_alice_text(rec: Dict) -> str:
    return str(rec["x"])


def _default_bob_text(rec: Dict) -> str:
    return str(rec["y"])


def _default_nosignal_target(rec: Dict) -> str:
    return str(rec["sum"])


@dataclass
class StegConfig:
    class_weights: List[float] = field(
        default_factory=lambda: list(DEFAULT_CLASS_WEIGHTS)
    )
    alice_text_fn: Callable[[Dict], str] = _default_alice_text
    bob_text_fn: Callable[[Dict], str] = _default_bob_text
    nosignal_target_fn: Callable[[Dict], str] = _default_nosignal_target

    def __post_init__(self) -> None:
        if len(self.class_weights) != NUM_SIGNAL_CLASSES:
            raise ValueError(
                f"class_weights must have {NUM_SIGNAL_CLASSES} entries, "
                f"got {len(self.class_weights)}"
            )


def _make_record(base_rec: Dict, signal_class: int, cfg: StegConfig) -> StegRecord:
    if signal_class == 0:
        target_text = cfg.nosignal_target_fn(base_rec)
        target_kind = "cover"
    else:
        target_text = RASCAL_TARGETS[signal_class - 1]
        target_kind = "signal"
    return {
        "signal_class": signal_class,
        "signal_marker": SIGNAL_MARKERS[signal_class],
        "alice_text": cfg.alice_text_fn(base_rec),
        "bob_text": cfg.bob_text_fn(base_rec),
        "target_text": target_text,
        "target_kind": target_kind,
        "base": base_rec,
    }


def _draw_class(rng: random.Random, weights: List[float]) -> int:
    return rng.choices(range(NUM_SIGNAL_CLASSES), weights=weights, k=1)[0]


class SteganographicStream(IterableDataset):
    """Wrap an *iterable* base dataset, attaching a fresh signal to each record."""

    def __init__(
        self,
        base: Iterable[Dict],
        config: Optional[StegConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.cfg = config or StegConfig()
        self.seed = seed

    def __iter__(self) -> Iterator[StegRecord]:
        base = self.seed if self.seed is not None else random.randrange(2**31)
        rng = random.Random(base)
        for base_rec in self.base:
            signal_class = _draw_class(rng, self.cfg.class_weights)
            yield _make_record(base_rec, signal_class, self.cfg)


class SteganographicDataset(Dataset):
    """Wrap a *finite map-style* base dataset (held-out eval).

    The signal class for index ``i`` is a deterministic function of ``(seed, i)``,
    so the wrapped eval set is reproducible.
    """

    def __init__(
        self,
        base: Dataset,
        config: Optional[StegConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not hasattr(base, "__len__"):
            raise TypeError(
                "SteganographicDataset needs a sized (map-style) base; "
                "use SteganographicStream for IterableDataset bases."
            )
        self.base = base
        self.cfg = config or StegConfig()
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> StegRecord:
        base_rec = self.base[index]
        rng = random.Random((self.seed, index).__hash__())
        signal_class = _draw_class(rng, self.cfg.class_weights)
        return _make_record(base_rec, signal_class, self.cfg)


def wrap(base, config: Optional[StegConfig] = None, seed: Optional[int] = None):
    """Convenience: pick the right wrapper based on whether ``base`` is sized."""
    if isinstance(base, IterableDataset) or not hasattr(base, "__len__"):
        return SteganographicStream(base, config, seed)
    return SteganographicDataset(base, config, seed if seed is not None else 0)
