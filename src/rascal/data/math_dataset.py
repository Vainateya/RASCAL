"""Split-arithmetic cover task.

Alice holds ``x``, Bob holds ``y``. Operands are 3-4 digit integers sampled
*independently* of each other. Each record exposes the sum, the product, and the
difference. Bob's cover task is the sum (``x+y``); Alice's cover task is the
difference (``x-y``) -- swapped in from the original product because one-shot
multiplication sits past a 4B model's arithmetic ceiling, whereas subtraction is
the same difficulty class as addition. The product field is retained only as an
off-type option for a possible future codebook; it is no longer anyone's task.
Operands are sampled independently, so roughly half of ``x-y`` is negative.

Two flavours, both HuggingFace-/``torch.utils.data``-friendly:

- :class:`SplitArithmeticStream` -- an ``IterableDataset`` that yields fresh
  samples forever. This is the training stream (effectively infinite).
- :class:`SplitArithmeticDataset` -- a finite, deterministically-seeded
  map-style ``Dataset`` for a stable held-out eval set (~2-5k rows).

Both draw from the same :func:`sample_arithmetic` primitive so the train and
eval distributions are identical. Records are plain dicts (``ArithmeticRecord``)
so they compose with any collator / tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional
import random

import torch
from torch.utils.data import Dataset, IterableDataset

# A record is just a dict; the alias documents the schema.
ArithmeticRecord = Dict[str, int]  # {"x", "y", "sum", "product", "difference"}


def _sample_operand(rng: random.Random, min_digits: int, max_digits: int) -> int:
    """Sample an integer with a uniformly-chosen digit-length in [min, max]."""
    n_digits = rng.randint(min_digits, max_digits)
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
    hi = 10 ** n_digits - 1
    return rng.randint(lo, hi)


def sample_arithmetic(
    rng: random.Random,
    min_digits: int = 3,
    max_digits: int = 4,
) -> ArithmeticRecord:
    """Draw one ``(x, y)`` pair (independent) and its sum, product, difference.

    ``difference = x - y`` (Alice's cover task) may be negative, since ``x`` and
    ``y`` are sampled independently.
    """
    x = _sample_operand(rng, min_digits, max_digits)
    y = _sample_operand(rng, min_digits, max_digits)
    return {"x": x, "y": y, "sum": x + y, "product": x * y, "difference": x - y}


@dataclass
class _Config:
    min_digits: int = 3
    max_digits: int = 4


class SplitArithmeticStream(IterableDataset):
    """Infinite stream of fresh arithmetic records (the training stream).

    Each worker (and each restart of the iterator) advances an independent RNG so
    samples are always fresh. ``seed`` only fixes the *starting* point for
    reproducibility; the stream never repeats within a run.
    """

    def __init__(
        self,
        min_digits: int = 3,
        max_digits: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = _Config(min_digits, max_digits)
        self.seed = seed

    def __iter__(self) -> Iterator[ArithmeticRecord]:
        # Give each DataLoader worker a disjoint RNG stream.
        worker = torch.utils.data.get_worker_info()
        base = self.seed if self.seed is not None else random.randrange(2**31)
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(base + 1000003 * worker_id)
        while True:
            yield sample_arithmetic(rng, self.cfg.min_digits, self.cfg.max_digits)


class SplitArithmeticDataset(Dataset):
    """Finite, deterministically-seeded arithmetic set (held-out eval).

    ``__getitem__(i)`` is a pure function of ``(seed, i)``, so the eval set is
    identical across runs and processes -- essential for stable metrics.
    """

    def __init__(
        self,
        num_samples: int,
        min_digits: int = 3,
        max_digits: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.cfg = _Config(min_digits, max_digits)
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> ArithmeticRecord:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        # Per-index RNG -> deterministic, order-independent, and genuinely
        # reproducible across runs/processes (no dependence on hash internals).
        rng = random.Random(self.seed * 1_000_003 + index)
        return sample_arithmetic(rng, self.cfg.min_digits, self.cfg.max_digits)
