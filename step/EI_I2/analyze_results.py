#!/usr/bin/env python3
"""
Innovation II — Result Analyzer
================================
Reads four detection_report.json files (2 methods × 2 output orders)
and prints the 2×2 comparison table needed for the paper.

Run on PC after copying results/ from the board:

    python3 analyze_results.py --results_dir ./results
"""

import json
import argparse
from pathlib import Path


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(report: dict | None, key: str, fallback: str = "N/A") -> str:
    if report is None:
        return fallback
    return str(report.get(key, fallback))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Innovation II experiment results (2×2 comparison table)"
    )
    parser.add_argument("--results_dir", default="./results",
                        help="Directory containing the four result sub-folders")
    args = parser.parse_args()

    base = Path(args.results_dir)
    paths = {
        ("shape_driven", "natural"):  base / "shape_driven_natural"  / "detection_report.json",
        ("shape_driven", "reordered"): base / "shape_driven_reordered" / "detection_report.json",
        ("fixed_index",  "natural"):  base / "fixed_index_natural"   / "detection_report.json",
        ("fixed_index",  "reordered"): base / "fixed_index_reordered"  / "detection_report.json",
    }

    reports = {k: load_report(v) for k, v in paths.items()}

    # ── availability check ────────────────────────────────────────────────────
    missing = [str(v) for k, v in paths.items() if reports[k] is None]
    if missing:
        print(f"[WARN] Missing report(s):")
        for m in missing:
            print(f"       {m}")
        print()

    # ── 2×2 table ─────────────────────────────────────────────────────────────
    W = 22
    h_line = "+" + "-"*W + "+" + "-"*W + "+" + "-"*W + "+"

    print()
    print("=" * 70)
    print("  Innovation II — Output Head Matching: 2×2 Experiment Results")
    print("=" * 70)
    print()
    print(h_line)
    print(f"| {'Method':<{W-2}} | {'Natural order':<{W-2}} | {'Reordered (simulated)':<{W-2}} |")
    print(h_line)

    for method_key, method_label in [
        ("shape_driven", "Shape-driven (ours)"),
        ("fixed_index",  "Fixed-index (naive)"),
    ]:
        nat = reports[(method_key, "natural")]
        ror = reports[(method_key, "reordered")]

        def cell(r):
            if r is None:
                return "not run"
            ok   = r.get("decode_success", "?")
            tot  = r.get("num_images",     "?")
            rate = r.get("decode_success_rate", "?")
            det  = r.get("total_detections",    "?")
            return f"{ok}/{tot} ({rate})  det={det}"

        print(f"| {method_label:<{W-2}} | {cell(nat):<{W-2}} | {cell(ror):<{W-2}} |")
    print(h_line)

    # ── per-method detail ──────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  Detailed breakdown")
    print("─" * 70)
    for (method_key, order_key), r in reports.items():
        label = f"{method_key}  [{order_key}]"
        if r is None:
            print(f"\n  {label}: (no data)")
            continue
        print(f"\n  {label}")
        print(f"    decode_success_rate : {r.get('decode_success_rate','?')}")
        print(f"    decode_success      : {r.get('decode_success','?')} / {r.get('num_images','?')}")
        print(f"    decode_fail         : {r.get('decode_fail', '?')}")
        print(f"    total_detections    : {r.get('total_detections','?')}")
        print(f"    avg_infer_ms        : {r.get('avg_infer_ms','?')}")
        if r.get("errors_sample"):
            print(f"    first_error         : {r['errors_sample'][0].get('error','')}")

    # ── error sample from fixed_index_reordered ────────────────────────────────
    ror_fi = reports.get(("fixed_index", "reordered"))
    if ror_fi and ror_fi.get("errors_sample"):
        print()
        print("─" * 70)
        print("  Fixed-index reordered — error sample (for paper)")
        print("─" * 70)
        for e in ror_fi["errors_sample"]:
            print(f"    {e.get('img','?'):40s}  {e.get('error','?')}")

    # ── paper-ready one-liner conclusion ──────────────────────────────────────
    sd_nat = reports.get(("shape_driven", "natural"))
    sd_ror = reports.get(("shape_driven", "reordered"))
    fi_nat = reports.get(("fixed_index",  "natural"))
    fi_ror = reports.get(("fixed_index",  "reordered"))

    print()
    print("─" * 70)
    print("  Paper conclusion (auto-generated)")
    print("─" * 70)
    if all(r is not None for r in [sd_nat, sd_ror, fi_nat, fi_ror]):
        n = sd_nat["num_images"]
        sd_ror_ok = sd_ror["decode_success"]
        fi_ror_ok = fi_ror["decode_success"]
        fi_ror_fail = fi_ror["decode_fail"]
        print(f"""
  When the compiler reorders output tensors (permutation {REORDER_PERM_STR}),
  the fixed-index decoder fails on {fi_ror_fail}/{n} images ({fi_ror['decode_success_rate']} success),
  while the shape-driven decoder recovers correctly on {sd_ror_ok}/{n} images
  ({sd_ror['decode_success_rate']} success) without any additional cost.
""")
    else:
        print("  (Run all four experiment groups first to generate conclusion.)")
    print("=" * 70)


REORDER_PERM_STR = "[3,0,4,1,5,2]"

if __name__ == "__main__":
    main()
