"""Measure the tokenized length of both tasks' extreme outputs, and confirm that
negative x-y targets round-trip through encode -> decode cleanly (no stray space
after the minus sign, no tokenizer artifact). Run on the GPU box where the
Qwen3 tokenizer is available.

    python scripts/measure_output_tokens.py --model Qwen/Qwen3-4B
"""
import argparse
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)

    # Extremes covering x-y range AND Bob's max sum.
    extremes = [-9899, -437, -100, 0, 100, 9899, 19998]
    print("token counts:")
    max_len = 0
    for v in extremes:
        ids = tok(str(v), add_special_tokens=False)["input_ids"]
        max_len = max(max_len, len(ids))
        print(f"  {v!r:>8} -> {len(ids)} tokens  ids={ids}")
    recommended = max_len + 2  # EOS + margin
    print(f"\nmax output tokens = {max_len}  ->  recommended max_new_tokens = {recommended}")
    print(f"(config default is 8; must be >= {recommended})")

    # Negative round-trip fidelity: encode then decode a signed target.
    print("\nnegative-target round-trip:")
    for v in [-437, -9899, -100]:
        s = str(v)
        ids = tok(s, add_special_tokens=False)["input_ids"]
        back = tok.decode(ids).strip()
        ok = back == s
        print(f"  {s!r} -> {ids} -> {back!r}  {'OK' if ok else 'MISMATCH <-- investigate'}")
        assert ok, f"signed target {s!r} did not round-trip (got {back!r})"
    print("\nall negative targets round-trip exactly; exact-match scorer is safe.")


if __name__ == "__main__":
    main()
