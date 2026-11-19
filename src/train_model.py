# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Part B — Tabular ML Baseline.
#              Trains three classifiers, evaluates them, saves
#              the best pipeline with joblib, and writes a
#              detailed metrics report.
# ============================================================
"""
train_model.py
--------------
Implements the full Part B ML pipeline:

  1. Load cleaned data
  2. Split into train / test (stratified 80/20)
  3. Build a scikit-learn Pipeline for each of three classifiers:
       - Logistic Regression  (baseline / interpretable)
       - Random Forest        (ensemble / feature-importance)
       - Gradient Boosting    (usually top performer)
  4. Each Pipeline:
       - ColumnTransformer: impute + scale numerics, OHE categoricals
       - Classifier
  5. Evaluate all three on the held-out test set:
       - accuracy, weighted F1, per-class precision/recall/F1
       - confusion matrix
  6. Save the best pipeline to models/best_pipeline.joblib
  7. Write all results to outputs/reports/model_metrics.json

Run:
    python src/train_model.py
"""

import os
import json
import sys
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

from src.features import (
    NUMERIC_FEATURES, TEMPORAL_FEATURES, BOOL_FEATURES,
    CATEGORICAL_FEATURES, FEATURE_COLS, TARGET_COL, CLASS_LABELS,
    get_X_y, feature_importance_df,
)

CLEANED_PATH  = "data/cleaned/pakistan_aq_cleaned.csv"
MODELS_DIR    = "models"
REPORTS_DIR   = "outputs/reports"
FIGURES_DIR   = "outputs/figures"
RANDOM_STATE  = 42


def load_data(path: str = CLEANED_PATH) -> pd.DataFrame:
    """Load cleaned CSV; validate required columns present."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"[train] Missing columns: {missing}")
    print(f"[train] Loaded {len(df):,} rows × {len(df.columns)} cols")
    return df


def build_pipeline(clf, numeric_cols, temporal_cols, bool_cols, cat_cols) -> Pipeline:
    """Build a full preprocessing + classifier Pipeline.

    Preprocessing:
    - Numeric + temporal + bool: SimpleImputer(median) → StandardScaler
    - Categorical: SimpleImputer(constant='Unknown') → OneHotEncoder(handle_unknown='ignore')
    """
    num_all = numeric_cols + temporal_cols + bool_cols

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_all),
        ("cat", cat_transformer, cat_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])


def evaluate(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted pipeline and return a metrics dict."""
    y_pred = pipeline.predict(X_test)

    acc  = float(accuracy_score(y_test, y_pred))
    wf1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()

    # Per-class dict (keep only the 4 label rows, not avg rows)
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall":    round(report[label]["recall"],    4),
            "f1-score":  round(report[label]["f1-score"],  4),
            "support":   int(report[label]["support"]),
        }
        for label in CLASS_LABELS
        if label in report
    }

    print(f"  {name:30s}  acc={acc:.4f}  wF1={wf1:.4f}")
    return {
        "model":       name,
        "accuracy":    round(acc, 4),
        "weighted_f1": round(wf1, 4),
        "per_class":   per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm_data: list, model_name: str, out_dir: str):
    """Save a confusion matrix heatmap for a model."""
    cm_arr = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"cm_{safe_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Confusion matrix → '{path}'")
    return path


def plot_feature_importance(pipeline, feature_names: list, model_name: str, out_dir: str):
    """Save a feature-importance bar chart (top 15 features)."""
    try:
        imp_df = feature_importance_df(pipeline, feature_names)
        top = imp_df.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#457B9D")
        ax.set_title(f"Top-15 Feature Importances — {model_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"fi_{safe_name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[train] Feature importance chart → '{path}'")
    except Exception as exc:
        print(f"[train] Feature importance skipped for {model_name}: {exc}")


def get_ohe_feature_names(pipeline) -> list[str]:
    """Extract feature names after ColumnTransformer OHE expansion."""
    pre = pipeline.named_steps["preprocessor"]
    num_names = (
        NUMERIC_FEATURES + TEMPORAL_FEATURES + BOOL_FEATURES
    )
    try:
        cat_names = list(
            pre.named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
    except Exception:
        cat_names = []
    return num_names + cat_names


def main():
    print("=" * 60)
    print("STEP — Part B: Tabular ML Baseline")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data()
    X, y = get_X_y(df)

    # ── Train / test split (stratified) ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"[train] Target distribution (train):")
    for cat, cnt in y_train.value_counts().items():
        print(f"         {cat:12s}: {cnt:5,}  ({100*cnt/len(y_train):.1f}%)")

    # ── Define three classifiers ──────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=RANDOM_STATE
        ),
    }

    # ── Train, evaluate, collect metrics ─────────────────
    results = []
    pipelines = {}
    print(f"\n[train] Training & evaluating:")

    for name, clf in models.items():
        pipeline = build_pipeline(
            clf,
            numeric_cols=NUMERIC_FEATURES,
            temporal_cols=TEMPORAL_FEATURES,
            bool_cols=BOOL_FEATURES,
            cat_cols=CATEGORICAL_FEATURES,
        )
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        metrics = evaluate(name, pipeline, X_test, y_test)
        metrics["train_time_sec"] = round(elapsed, 2)
        results.append(metrics)
        pipelines[name] = pipeline

        # Confusion matrix chart
        plot_confusion_matrix(metrics["confusion_matrix"], name, FIGURES_DIR)

        # Feature importance chart
        feat_names = get_ohe_feature_names(pipeline)
        plot_feature_importance(pipeline, feat_names, name, FIGURES_DIR)

    # ── Pick and save best model ──────────────────────────
    best = max(results, key=lambda r: r["weighted_f1"])
    best_name = best["model"]
    print(f"\n[train] Best model: '{best_name}'  (wF1 = {best['weighted_f1']:.4f})")

    best_pipeline = pipelines[best_name]
    model_path = os.path.join(MODELS_DIR, "best_pipeline.joblib")
    joblib.dump(best_pipeline, model_path, compress=3)
    print(f"[train] Saved best pipeline → '{model_path}'")

    # Also save each pipeline individually
    for name, pipe in pipelines.items():
        safe = name.lower().replace(" ", "_")
        p = os.path.join(MODELS_DIR, f"{safe}.joblib")
        joblib.dump(pipe, p, compress=3)

    # ── Error analysis (top misclassified rows) ───────────
    y_pred_best = best_pipeline.predict(X_test)
    errors = X_test.copy()
    errors["true_label"] = y_test.values
    errors["pred_label"] = y_pred_best
    errors["correct"]    = errors["true_label"] == errors["pred_label"]
    error_rows = errors[~errors["correct"]].head(20)
    error_path = os.path.join(REPORTS_DIR, "error_analysis.csv")
    error_rows.to_csv(error_path, index=False)
    print(f"[train] Error analysis (top 20 misclassifications) → '{error_path}'")

    # ── Summary comparison chart ──────────────────────────
    names_list = [r["model"] for r in results]
    acc_list   = [r["accuracy"] for r in results]
    wf1_list   = [r["weighted_f1"] for r in results]

    x = np.arange(len(names_list))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar(x - width/2, acc_list, width, label="Accuracy",    color="#2A9D8F")
    ax.bar(x + width/2, wf1_list, width, label="Weighted F1", color="#E63946")
    ax.set_title("Model Comparison — Accuracy vs Weighted F1")
    ax.set_xticks(x); ax.set_xticklabels(names_list, rotation=10)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(); plt.tight_layout()
    cmp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(cmp_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Comparison chart → '{cmp_path}'")

    # ── Save metrics JSON ─────────────────────────────────
    metrics_out = {
        "best_model":          best_name,
        "best_weighted_f1":    best["weighted_f1"],
        "best_accuracy":       best["accuracy"],
        "train_size":          len(X_train),
        "test_size":           len(X_test),
        "random_state":        RANDOM_STATE,
        "feature_count":       len(FEATURE_COLS),
        "feature_names":       FEATURE_COLS,
        "models":              results,
    }
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[train] Metrics saved → '{metrics_path}'")

    print(f"\n[train] Done. ✓")


if __name__ == "__main__":
    main()
