"""Minimal char-level tokenizer + collator for CP0.

CP0 uses tiny *randomly-initialised* Qwen3 models, so there is no pretrained
tokenizer to load. This module provides a tiny, self-contained vocabulary that is
enough to exercise the full data -> handoff -> loss path:

- digits ``0-9``
- the five Alice-only signal markers ``<sig_*>``
- the four off-type outputs ``RASCAL0..3`` as **atomic** tokens
- structural specials: ``<pad> <bos> <eos> <eq>``

Because the signal markers and the RASCAL outputs are distinct atomic tokens,
there is no symbol overlap between the input signal vocabulary and the output
codebook (the "no relay confound" invariant holds at the token-id level).

At CP1, swap this for the real Qwen3 tokenizer (register the ``<sig_*>`` and
``RASCAL{i}`` strings as added special tokens); the collator contract below is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .data.steg_dataset import SIGNAL_MARKERS, RASCAL_TARGETS


class CharTokenizer:
    """Tiny deterministic vocabulary for the arithmetic + signal scheme."""

    def __init__(self) -> None:
        specials = ["<pad>", "<bos>", "<eos>", "<eq>"]
        digits = [str(d) for d in range(10)]
        # Order fixed for reproducibility.
        vocab = specials + digits + list(SIGNAL_MARKERS) + list(RASCAL_TARGETS)
        self.itos: List[str] = vocab
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.eq_id = self.stoi["<eq>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # -- encoders ---------------------------------------------------------------

    def _digits(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def encode_alice(self, signal_marker: str, alice_text: str) -> List[int]:
        """Alice prompt: [BOS, signal_marker, x-digits]. Signal is Alice-only."""
        return [self.bos_id, self.stoi[signal_marker], *self._digits(alice_text)]

    def encode_bob_input(self, bob_text: str, target_text: str) -> Sequence[int]:
        """Bob teacher-forcing sequence + the label mask.

        Layout: [y-digits, <eq>, target..., <eos>]. Labels are -100 everywhere
        except the target span (+ EOS), so the loss is only on what Bob must emit.
        Returns ``(input_ids, labels)``.
        """
        prompt = [*self._digits(bob_text), self.eq_id]
        if target_text in self.stoi:  # atomic RASCAL token
            target = [self.stoi[target_text]]
        else:  # numeric sum -> digit tokens
            target = self._digits(target_text)
        target = target + [self.eos_id]

        input_ids = prompt + target
        labels = [-100] * len(prompt) + list(target)
        return input_ids, labels


@dataclass
class HandoffBatch:
    alice_input_ids: torch.Tensor
    alice_attention_mask: torch.Tensor
    bob_input_ids: torch.Tensor
    bob_attention_mask: torch.Tensor
    bob_labels: torch.Tensor
    signal_class: torch.Tensor

    def to(self, device) -> "HandoffBatch":
        return HandoffBatch(
            self.alice_input_ids.to(device),
            self.alice_attention_mask.to(device),
            self.bob_input_ids.to(device),
            self.bob_attention_mask.to(device),
            self.bob_labels.to(device),
            self.signal_class.to(device),
        )


def _pad(seqs: List[List[int]], pad_value: int, side: str) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [pad_value] * (width - len(s))
        out.append(pad + s if side == "left" else s + pad)
    return torch.tensor(out, dtype=torch.long)


def _pad_mask(seqs: List[List[int]], side: str) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [0] * (width - len(s))
        real = [1] * len(s)
        out.append(pad + real if side == "left" else real + pad)
    return torch.tensor(out, dtype=torch.long)


def collate_handoff(records: List[Dict], tokenizer: CharTokenizer) -> HandoffBatch:
    """Collate steg records into a padded batch for the Alice->Bob handoff.

    Alice is **left-padded** so real tokens sit flush against Bob's tokens (their
    positions stay contiguous across the handoff). Bob is **right-padded**; its
    pad positions are ``-100`` in the labels.
    """
    alice_ids, bob_ids, bob_labels, classes = [], [], [], []
    for rec in records:
        alice_ids.append(
            tokenizer.encode_alice(rec["signal_marker"], rec["alice_text"])
        )
        b_in, b_lab = tokenizer.encode_bob_input(rec["bob_text"], rec["target_text"])
        bob_ids.append(list(b_in))
        bob_labels.append(list(b_lab))
        classes.append(int(rec["signal_class"]))

    return HandoffBatch(
        alice_input_ids=_pad(alice_ids, tokenizer.pad_id, side="left"),
        alice_attention_mask=_pad_mask(alice_ids, side="left"),
        bob_input_ids=_pad(bob_ids, tokenizer.pad_id, side="right"),
        bob_attention_mask=_pad_mask(bob_ids, side="right"),
        bob_labels=_pad(bob_labels, -100, side="right"),
        signal_class=torch.tensor(classes, dtype=torch.long),
    )
