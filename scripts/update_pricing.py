#!/usr/bin/env python3
"""Check and update LLM API pricing in RadEval/metrics/_llm.py.

Usage:
    python scripts/update_pricing.py          # show current vs reference prices
    python scripts/update_pricing.py --apply  # update _llm.py in place
"""
import argparse
import re
from pathlib import Path

LLM_PY = Path(__file__).resolve().parent.parent / "RadEval" / "metrics" / "_llm.py"

# Reference prices (per 1M tokens: input, output).
# Update these from the provider pricing pages:
#   OpenAI:  https://openai.com/api/pricing/
#   Gemini:  https://ai.google.dev/gemini-api/docs/pricing
REFERENCE_PRICES = {
    "gpt-4o-mini":      (0.15, 0.60),
    "gpt-4.1-mini":     (0.40, 1.60),
    "gpt-4.1-nano":     (0.10, 0.40),
    "gpt-4o":           (2.50, 10.00),
    "gpt-5.2":          (2.00, 8.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro":   (1.25, 10.00),
}


def parse_current_prices(text: str) -> dict[str, tuple[float, float]]:
    """Extract PRICING_PER_1M entries from _llm.py source."""
    prices = {}
    for m in re.finditer(
        r'"([^"]+)":\s*\((\d+\.?\d*),\s*(\d+\.?\d*)\)', text
    ):
        model, inp, out = m.group(1), float(m.group(2)), float(m.group(3))
        prices[model] = (inp, out)
    return prices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Update _llm.py with reference prices")
    args = parser.parse_args()

    text = LLM_PY.read_text()
    current = parse_current_prices(text)

    all_models = sorted(set(current) | set(REFERENCE_PRICES))
    changes = []

    print("Model                    Current (in/out)      Reference (in/out)    Status")
    print("-" * 85)
    for model in all_models:
        cur = current.get(model)
        ref = REFERENCE_PRICES.get(model)
        cur_str = f"${cur[0]:>6.2f} / ${cur[1]:>6.2f}" if cur else "  (missing)"
        ref_str = f"${ref[0]:>6.2f} / ${ref[1]:>6.2f}" if ref else "  (missing)"

        if cur == ref:
            status = "OK"
        elif cur is None:
            status = "NEW (will add)"
            changes.append((model, ref))
        elif ref is None:
            status = "EXTRA (not in reference)"
        else:
            status = "STALE (will update)"
            changes.append((model, ref))
        print(f"  {model:<24} {cur_str:<22} {ref_str:<22} {status}")

    if not changes:
        print("\nAll prices up to date.")
        return

    print(f"\n{len(changes)} change(s) needed.")

    if not args.apply:
        print("Run with --apply to update _llm.py.")
        return

    for model, (inp, out) in changes:
        old_pattern = rf'"{re.escape(model)}":\s*\(\d+\.?\d*,\s*\d+\.?\d*\)'
        new_entry = f'"{model}": ({inp}, {out})'
        if re.search(old_pattern, text):
            text = re.sub(old_pattern, new_entry, text)
        else:
            text = text.replace(
                "}\n",
                f'    "{model}": ({inp}, {out}),\n}}\n',
                1,
            )

    LLM_PY.write_text(text)
    print(f"Updated {LLM_PY}")


if __name__ == "__main__":
    main()
