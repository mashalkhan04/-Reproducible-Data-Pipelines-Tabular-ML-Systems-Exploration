# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Master runner for Part A — executes all five
#              Part A steps in sequence and reports final status.
# ============================================================
"""
run_part_a.py
-------------
Orchestrates the complete Part A pipeline:

  Step 1 — inspect_data   : profile raw data, detect quality issues
  Step 2 — clean_data     : fix dtypes, coords, dedup, feature-engineer,
                            add AQI category, Winsorise outliers
  Step 3 — validate_data  : seven validators + quality gate (≥95 %)
  Step 4 — analyze_data   : group-by summaries, NumPy ops, SQLite query
  Step 5 — visualize_data : four PNG charts + captions JSON

Run:
    python src/run_part_a.py
"""

import sys
import time
import traceback

# ── Import each module ─────────────────────────────────────
import src.inspect_data   as step1
import src.clean_data     as step2
import src.validate_data  as step3
import src.analyze_data   as step4
import src.visualize_data as step5


def run_step(name: str, fn) -> bool:
    """Execute a pipeline step, print timing, return success flag."""
    print()
    print("─" * 60)
    print(f"  {name}")
    print("─" * 60)
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✓  {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as exc:
        # validate_data raises SystemExit if quality gate fails
        print(f"\n  ✗  {name} raised SystemExit: {exc}")
        return False
    except Exception:
        elapsed = time.time() - t0
        print(f"\n  ✗  {name} FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  PART A — Data Quality & Insights Pipeline")
    print("  Name       : MASHAL KHAN")
    print("  Roll Number: TRN-2026-000349")
    print("  Date       : 20 November 2026")
    print("=" * 60)

    total_start = time.time()

    steps = [
        ("Step 1 — Raw Data Inspection",           step1.main),
        ("Step 2 — Data Cleaning",                 step2.run_cleaning_pipeline),
        ("Step 3 — Data Validation",               step3.main),
        ("Step 4 — Analysis (Pandas/NumPy/SQLite)", step4.main),
        ("Step 5 — Visualisations",                step5.main),
    ]

    results = {}
    for name, fn in steps:
        ok = run_step(name, fn)
        results[name] = ok
        if not ok:
            print(f"\n[PIPELINE] Aborting — {name} did not pass.")
            break

    # ── Summary ───────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("  PART A PIPELINE SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n  {passed}/{total} steps passed   |   Total time: {total_elapsed:.1f}s")

    if passed == total:
        print()
        print("  ✓ Part A complete.  Check:")
        print("    outputs/reports/   — JSON summaries & SQL query results")
        print("    outputs/figures/   — four chart PNGs")
        print("    data/cleaned/      — cleaned CSV")
        print("    docs/schema.md     — auto-generated schema")
        print("    sql/queries.sql    — SQLite GROUP BY query")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n  ✗ Part A did NOT finish cleanly.  Review errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
