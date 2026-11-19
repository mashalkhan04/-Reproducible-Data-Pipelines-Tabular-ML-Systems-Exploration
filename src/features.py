# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Feature engineering for the ML baseline —
#              selects, encodes, and documents the final
#              feature set used by train_model.py.
# ============================================================
"""
features.py
-----------
Defines FEATURE_COLS, TARGET_COL, and helper functions that
prepare a cleaned DataFrame for scikit-learn Pipelines.

Responsibilities
~~~~~~~~~~~~~~~~
- Document every feature with a rationale comment.
- Separate numeric vs. categorical columns (used by ColumnTransformer).
- Provide `get_X_y(df)` to split features and target.
- Provide `feature_importance_df(model, feature_names)` to extract
  importances from tree-based models.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── Feature selection ──────────────────────────────────────
# Numeric features — raw pollutant measurements + AQI score
NUMERIC_FEATURES = [
    #"pm10",              # coarse particles (µg/m³) — high industrial/dust signal
    #"pm2_5",             # fine particles   (µg/m³) — primary health metric
    "carbon_monoxide",   # CO (µg/m³)       — traffic combustion tracer
    "nitrogen_dioxide",  # NO2 (µg/m³)      — traffic + industrial
    "sulphur_dioxide",   # SO2 (µg/m³)      — industrial / coal burning
    "ozone",             # O3 (µg/m³)       — secondary pollutant
    "dust",              # dust (µg/m³)      — Saharan dust + local dust storms
    "uv_index",          # UV index          — ozone formation proxy
]

# Temporal / cyclical features (added by clean_data)
TEMPORAL_FEATURES = [
    "hour",              # 0-23 — diurnal cycle
    "month",             # 1-12 — seasonal cycle
    "day_of_week",       # 0-6  — weekday vs weekend traffic
]

# Boolean feature
BOOL_FEATURES = [
    "is_weekend",        # True/False — weekend emission pattern
]

# Categorical features (low-cardinality)
CATEGORICAL_FEATURES = [
    "city",              # 6 cities — captures geo-climatic differences
    "season",            # Winter/Spring/Summer/Autumn
    "time_of_day",       # Night/Morning/Afternoon/Evening
]

# All features the model uses
FEATURE_COLS: list[str] = (
    NUMERIC_FEATURES + TEMPORAL_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES
)

# Target
TARGET_COL = "aqi_category"

# Class labels in natural severity order (used for reports / confusion matrix)
CLASS_LABELS = ["Low", "Moderate", "High", "Very High"]


def get_X_y(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target series y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_data.py).

    Returns
    -------
    X : pd.DataFrame  — shape (n_samples, len(FEATURE_COLS))
    y : pd.Series     — target labels (str)
    """
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"[features] Missing columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Cast boolean to int so sklearn handles it cleanly
    X["is_weekend"] = X["is_weekend"].astype(int)

    return X, y


def feature_importance_df(
    model,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame of feature importances.

    Works with any scikit-learn Pipeline whose final step exposes
    ``feature_importances_`` (Random Forest, Gradient Boosting) or
    ``coef_`` (Logistic Regression).

    Parameters
    ----------
    model        : fitted sklearn Pipeline or estimator
    feature_names: names after ColumnTransformer expansion

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    estimator = model.named_steps["clf"] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # For multi-class, take mean absolute coefficient across classes
        importances = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        raise AttributeError("Estimator exposes neither feature_importances_ nor coef_.")

    names = (
        feature_names
        if feature_names is not None
        else [f"f{i}" for i in range(len(importances))]
    )

    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
