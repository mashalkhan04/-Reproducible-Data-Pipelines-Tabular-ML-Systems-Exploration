# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : BONUS Part B — CLI inference script.
#              Loads the saved best pipeline and writes
#              predictions CSV for any new input data.
# ============================================================
"""
predict.py  (Bonus)
-------------------
Command-line tool that:
  1. Loads the saved best_pipeline.joblib
  2. Reads a new CSV of raw / cleaned feature rows
  3. Writes a CSV with an added 'predicted_aqi_category' column

Usage:
    python src/predict.py --input <csv_path> [--output <out_path>]

Example:
    python src/predict.py --input data/cleaned/pakistan_aq_cleaned.csv \\
                          --output outputs/predictions.csv
"""

import argparse
import sys
import os
import pandas as pd
import joblib

from src.features import FEATURE_COLS, TARGET_COL

DEFAULT_MODEL  = "models/best_pipeline.joblib"
DEFAULT_OUTPUT = "outputs/predictions.csv"


def load_pipeline(model_path: str):
    """Load a joblib-serialised sklearn Pipeline."""
    if not os.path.exists(model_path):
        print(f"[predict] ERROR: model file not found: '{model_path}'", file=sys.stderr)
        sys.exit(1)
    pipeline = joblib.load(model_path)
    print(f"[predict] Loaded pipeline from '{model_path}'")
    return pipeline


def load_input(csv_path: str) -> pd.DataFrame:
    """Load input CSV; check required feature columns are present."""
    if not os.path.exists(csv_path):
        print(f"[predict] ERROR: input file not found: '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(
            f"[predict] ERROR: input CSV is missing feature columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[predict] Loaded {len(df):,} rows from '{csv_path}'")
    return df


def predict(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference and attach predicted column."""
    # Cast boolean column if present
    if "is_weekend" in df.columns:
        df = df.copy()
        df["is_weekend"] = df["is_weekend"].astype(int)

    X = df[FEATURE_COLS]
    preds = pipeline.predict(X)
    out = df.copy()
    out["predicted_aqi_category"] = preds

    # Also attach probability if the model supports it
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            classes = pipeline.classes_
            for i, cls in enumerate(classes):
                out[f"prob_{cls}"] = proba[:, i]
        except Exception:
            pass  # gracefully skip if proba unavailable

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run AQI category predictions on a cleaned feature CSV."
    )
    parser.add_argument(
        "--input",  "-i",
        default="data/cleaned/pakistan_aq_cleaned.csv",
        help="Path to input CSV (default: data/cleaned/pakistan_aq_cleaned.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",  "-m",
        default=DEFAULT_MODEL,
        help=f"Path to joblib pipeline (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    pipeline = load_pipeline(args.model)
    df       = load_input(args.input)
    out_df   = predict(pipeline, df)
    out_df.to_csv(args.output, index=False)

    # Quick accuracy check if true labels are available
    if TARGET_COL in df.columns:
        correct = (out_df["predicted_aqi_category"] == df[TARGET_COL]).sum()
        acc     = correct / len(df)
        print(f"[predict] On-data accuracy: {acc:.4f} ({correct:,}/{len(df):,} correct)")

    print(f"[predict] Predictions saved → '{args.output}'  ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
