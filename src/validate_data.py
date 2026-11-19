# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Step 3 of Part A — validation functions.
#              These functions check that cleaned data obeys
#              domain rules and fail loudly if something is wrong.
# ============================================================
"""
validate_data.py
----------------
Validation functions for the Pakistan Air Quality dataset.
Each function:
  - Takes a DataFrame.
  - Raises ValueError with a clear message if validation fails.
  - Returns the DataFrame unchanged if validation passes.

This design is called "loud failure" — the pipeline stops with
a clear error rather than silently producing wrong results.

Run:
    python src/validate_data.py
"""

import os
import json
import pandas as pd
import numpy as np

REPORTS_DIR  = "outputs/reports"
CLEANED_PATH = "data/cleaned/pakistan_aq_cleaned.csv"

# Validation thresholds
QUALITY_GATE_THRESHOLD_PCT = 95.0   # pipeline aborts if pass-rate < this


# ── individual validators ──────────────────────────────────

def validate_no_nulls(df: pd.DataFrame,
                      cols: list | None = None) -> pd.DataFrame:
    """Raise ValueError if any specified column contains null values.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
        Columns to check.  Defaults to all columns.

    Returns
    -------
    pd.DataFrame
        Unchanged df if validation passes.

    Raises
    ------
    ValueError
        If nulls are found, naming the column and count.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    >>> try:
    ...     validate_no_nulls(df, cols=["a"])
    ... except ValueError as e:
    ...     print("Caught:", e)
    Caught: Null check FAILED: column 'a' has 1 null value(s).
    """
    if cols is None:
        cols = df.columns.tolist()
    for col in cols:
        n = int(df[col].isna().sum())
        if n > 0:
            raise ValueError(
                f"Null check FAILED: column '{col}' has {n} null value(s)."
            )
    print(f"[validate] ✓ No nulls in {cols}")
    return df


def validate_non_negative(df: pd.DataFrame,
                           cols: list | None = None) -> pd.DataFrame:
    """Raise ValueError if any pollutant column contains negative values.

    Pollutant concentrations cannot be negative in physical reality.
    Negative readings indicate sensor malfunction or bad calibration.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
        Columns to check.  Defaults to pollutant columns.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"pm2_5": [10.0, -1.0]})
    >>> try:
    ...     validate_non_negative(df, cols=["pm2_5"])
    ... except ValueError as e:
    ...     print("Caught:", e)
    Caught: Non-negative check FAILED: column 'pm2_5' has 1 negative value(s). Min=-1.0.
    """
    if cols is None:
        cols = [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide",
            "ozone", "dust", "uv_index", "european_aqi",
        ]
    for col in cols:
        if col not in df.columns:
            continue
        n = int((df[col] < 0).sum())
        if n > 0:
            raise ValueError(
                f"Non-negative check FAILED: column '{col}' has {n} "
                f"negative value(s). Min={df[col].min():.3f}."
            )
    print(f"[validate] ✓ All pollutant columns are non-negative")
    return df


def validate_pm_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if PM2.5 ever exceeds PM10.

    PM2.5 (fine particles ≤ 2.5 µm) is a subset of PM10
    (particles ≤ 10 µm).  PM2.5 > PM10 is physically impossible
    and signals a sensor cross-wiring or calculation error.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
    """
    bad = df[df["pm2_5"] > df["pm10"]]
    if len(bad) > 0:
        raise ValueError(
            f"PM relationship FAILED: {len(bad)} rows where PM2.5 > PM10. "
            f"First offender: {bad[['timestamp','city','pm2_5','pm10']].iloc[0].to_dict()}"
        )
    print(f"[validate] ✓ PM2.5 ≤ PM10 in all rows")
    return df


def validate_timestamp_range(df: pd.DataFrame,
                               start: str = "2024-01-01",
                               end:   str = "2024-12-31") -> pd.DataFrame:
    """Raise ValueError if any timestamp falls outside the expected range.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a datetime 'timestamp' column.
    start, end : str
        ISO date strings for the expected range.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
    """
    ts_min = pd.Timestamp(start)
    ts_max = pd.Timestamp(end) + pd.Timedelta(hours=23)
    out_of_range = df[
        (df["timestamp"] < ts_min) | (df["timestamp"] > ts_max)
    ]
    if len(out_of_range) > 0:
        raise ValueError(
            f"Timestamp range FAILED: {len(out_of_range)} rows outside "
            f"[{start}, {end}]. "
            f"Min={df['timestamp'].min()}, Max={df['timestamp'].max()}"
        )
    print(f"[validate] ✓ All timestamps in [{start}, {end}]")
    return df


def validate_city_set(df: pd.DataFrame,
                       expected_cities: set | None = None) -> pd.DataFrame:
    """Raise ValueError if the DataFrame contains unexpected city names.

    Parameters
    ----------
    df : pd.DataFrame
    expected_cities : set, optional
        Allowed city names.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"city": ["Lahore", "UnknownCity"]})
    >>> try:
    ...     validate_city_set(df)
    ... except ValueError as e:
    ...     print("Caught:", str(e)[:60])
    Caught: City set FAILED: unexpected cities {'Unknowncity'}.
    """
    if expected_cities is None:
        expected_cities = {
            "Lahore", "Karachi", "Islamabad",
            "Faisalabad", "Multan", "Peshawar",
        }
    actual = set(df["city"].unique())
    unexpected = actual - expected_cities
    if unexpected:
        raise ValueError(
            f"City set FAILED: unexpected cities {unexpected}. "
            f"Expected: {expected_cities}"
        )
    print(f"[validate] ✓ City names valid: {sorted(actual)}")
    return df


def validate_aqi_category_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if aqi_category column contains NaN values.

    NaN in a categorical column means the pd.cut binning missed some
    values — usually because european_aqi was 0 or negative.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
    """
    n_nan = int(df["aqi_category"].isna().sum())
    if n_nan > 0:
        raise ValueError(
            f"AQI category FAILED: {n_nan} rows have NaN aqi_category. "
            "Check european_aqi values and pd.cut bins."
        )
    print(f"[validate] ✓ aqi_category has no NaN values")
    return df


def validate_coordinate_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if two different cities share identical coordinates.

    After the cleaning step that corrects Multan's longitude, no two
    cities should share both latitude and longitude.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
    """
    coords = (
        df[["city", "latitude", "longitude"]]
        .drop_duplicates()
    )
    dup_lons = coords.groupby("longitude")["city"].apply(list)
    for lon, cities in dup_lons.items():
        if len(cities) > 1:
            raise ValueError(
                f"Coordinate uniqueness FAILED: cities {cities} share "
                f"longitude {lon}. The Multan longitude fix may not have run."
            )
    print(f"[validate] ✓ All cities have unique longitudes")
    return df


# ── quality gate ───────────────────────────────────────────

def run_all_validations(df: pd.DataFrame,
                         threshold_pct: float = QUALITY_GATE_THRESHOLD_PCT
                         ) -> dict:
    """Run every validator and return a pass/fail report.

    If the overall pass rate falls below threshold_pct, this function
    raises SystemExit — the quality gate that stops the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to validate.
    threshold_pct : float
        Minimum % of validators that must pass (default 95%).

    Returns
    -------
    dict
        Results with keys: 'passed', 'failed', 'pass_rate_pct', 'details'.

    Raises
    ------
    SystemExit
        If pass rate < threshold_pct.
    """
    checks = [
        ("no_nulls_core",           lambda d: validate_no_nulls(d, ["timestamp","pm10","pm2_5","city","european_aqi"])),
        ("non_negative_pollutants",  validate_non_negative),
        ("pm_relationship",          validate_pm_relationship),
        ("timestamp_range",          validate_timestamp_range),
        ("city_set",                 validate_city_set),
        ("aqi_category_complete",    validate_aqi_category_completeness),
        ("coordinate_uniqueness",    validate_coordinate_uniqueness),
    ]

    results = {"passed": [], "failed": [], "details": {}}

    for name, func in checks:
        try:
            func(df)
            results["passed"].append(name)
            results["details"][name] = "PASS"
        except (ValueError, KeyError) as e:
            results["failed"].append(name)
            results["details"][name] = f"FAIL: {e}"
            print(f"[validate] ✗ {name}: {e}")

    total = len(checks)
    n_passed = len(results["passed"])
    pass_rate = round(100 * n_passed / total, 1)
    results["pass_rate_pct"] = pass_rate

    print(f"\n[validate] RESULT: {n_passed}/{total} checks passed "
          f"({pass_rate}%)")

    # ── QUALITY GATE ──────────────────────────────────────
    if pass_rate < threshold_pct:
        print(f"\n❌ QUALITY GATE FAILED: pass rate {pass_rate}% "
              f"< threshold {threshold_pct}%")
        print("Failed checks:")
        for name in results["failed"]:
            print(f"  → {name}: {results['details'][name]}")
        raise SystemExit(
            f"Pipeline aborted: data quality below {threshold_pct}% threshold."
        )

    print(f"[validate] ✓ Quality gate PASSED ({pass_rate}% ≥ {threshold_pct}%)")
    return results


def main():
    """Run the full validation suite on the cleaned data."""
    print("=" * 60)
    print("STEP 3 — Data Validation")
    print("=" * 60)

    if not os.path.exists(CLEANED_PATH):
        print(f"Cleaned data not found at '{CLEANED_PATH}'.")
        print("Run clean_data.py first.")
        return

    df = pd.read_csv(CLEANED_PATH, parse_dates=["timestamp"])

    results = run_all_validations(df)

    # save validation report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[validate] Report saved → '{report_path}'")

    # update metrics.json with pass rate
    metrics_path = os.path.join(REPORTS_DIR, "quality_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["validation_pass_rate_pct"] = results["pass_rate_pct"]
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[validate] metrics.json updated with pass rate")

    print("\n[validate] Done. ✓")


if __name__ == "__main__":
    main()
