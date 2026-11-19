# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Master runner for Part B — trains the ML
#              pipeline and runs the bonus prediction script.
# ============================================================
"""
run_part_b.py
-------------
Orchestrates Part B:

  Step 1 — train_model : builds, trains, evaluates 3 classifiers;
                         saves best pipeline to models/
  Step 2 — predict     : runs the bonus CLI predictor on the
                         cleaned dataset to verify end-to-end

Run:
    python src/run_part_b.py
"""

import sys
import time
import traceback
import subprocess

import src.train_model as step_train


def run_step(name: str, fn) -> bool:
    print()
    print("─" * 60)
    print(f"  {name}")
    print("─" * 60)
    t0 = time.time()
    try:
        fn()
        print(f"\n  ✓  {name} completed in {time.time() - t0:.1f}s")
        return True
    except Exception:
        print(f"\n  ✗  {name} FAILED after {time.time() - t0:.1f}s")
        traceback.print_exc()
        return False


def run_predict_bonus() -> bool:
    """Run bonus predict.py via subprocess to test the CLI interface."""
    print()
    print("─" * 60)
    print("  Step 2 (Bonus) — CLI Prediction Script")
    print("─" * 60)
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "src.predict",
            "--input",  "data/cleaned/pakistan_aq_cleaned.csv",
            "--output", "outputs/predictions.csv",
            "--model",  "models/best_pipeline.joblib",
        ],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓  CLI predictor completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗  CLI predictor FAILED (exit code {result.returncode})")
        return False


def main():
    print("=" * 60)
    print("  PART B — Tabular ML Baseline")
    print("  Name       : MASHAL KHAN")
    print("  Roll Number: TRN-2026-000349")
    print("  Date       : 20 November 2026")
    print("=" * 60)

    total_start = time.time()
    results = {}

    ok = run_step("Step 1 — Model Training & Evaluation", step_train.main)
    results["Train Models"] = ok

    if ok:
        ok2 = run_predict_bonus()
        results["Bonus CLI Predictor"] = ok2
    else:
        results["Bonus CLI Predictor"] = False
        print("[PIPELINE] Skipping predict step — training failed.")

    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("  PART B PIPELINE SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n  {passed}/{total} steps passed   |   Total time: {total_elapsed:.1f}s")

    if passed == total:
        print()
        print("  ✓ Part B complete.  Check:")
        print("    models/            — saved joblib pipelines")
        print("    outputs/reports/   — model_metrics.json, error_analysis.csv")
        print("    outputs/figures/   — confusion matrices, feature importance charts")
        print("    outputs/           — predictions.csv (bonus)")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n  ✗ Part B did NOT finish cleanly.  Review errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
