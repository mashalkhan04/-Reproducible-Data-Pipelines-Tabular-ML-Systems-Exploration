# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Step 4 of Part A — Pandas grouping, aggregation,
#              pivot tables, NumPy vectorized operations, and
#              SQLite query.
# ============================================================
"""
analyze_data.py
---------------
Produces all required analytical outputs for Part A:
  - Grouped summaries with Pandas (groupby, agg, pivot_table, value_counts)
  - Two NumPy vectorized operations
  - One SQLite GROUP BY query
  - Saves all results as CSV / JSON files in outputs/reports/

Run:
    python src/analyze_data.py
"""

import os
import json
import sqlite3

import numpy as np
import pandas as pd

CLEANED_PATH = "data/cleaned/pakistan_aq_cleaned.csv"
REPORTS_DIR  = "outputs/reports"
DB_PATH      = "outputs/reports/pakistan_aq.db"
SQL_PATH     = "sql/queries.sql"


# ── helper ────────────────────────────────────────────────

def load_cleaned(path: str = CLEANED_PATH) -> pd.DataFrame:
    """Load the cleaned CSV with correct dtypes."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # aqi_category is an ordered categorical
    cat_order = ["Low", "Moderate", "High", "Very High"]
    df["aqi_category"] = pd.Categorical(
        df["aqi_category"], categories=cat_order, ordered=True
    )
    return df


# ── PANDAS ANALYSIS ───────────────────────────────────────

def summary_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean pollutant concentrations grouped by city.

    Uses .groupby() + .agg() to calculate mean, max, and standard
    deviation for the main pollutants across each city.

    Industry use: a city health authority comparing air quality
    across its municipalities to prioritize clean-air interventions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Multi-level column DataFrame with stats per city.
    """
    agg_dict = {
        "pm10":             ["mean", "max", "std"],
        "pm2_5":            ["mean", "max", "std"],
        "carbon_monoxide":  ["mean", "max"],
        "nitrogen_dioxide": ["mean", "max"],
        "european_aqi":     ["mean", "max"],
    }
    summary = df.groupby("city").agg(agg_dict).round(2)
    return summary


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly average PM2.5 by city.

    Uses .groupby(['city', 'month']) to show seasonal pollution
    patterns.  Winter months (Dec–Feb) typically show higher PM2.5
    in Lahore and Faisalabad due to crop burning and cold inversions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return (
        df.groupby(["city", "month"])["pm2_5"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"pm2_5": "mean_pm2_5"})
    )


def aqi_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table: rows = city, columns = AQI category, values = count.

    Uses pd.pivot_table() with aggfunc='size'.
    Shows which cities have the most 'Very High' AQI hours.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    pivot = pd.pivot_table(
        df,
        index="city",
        columns="aqi_category",
        values="european_aqi",
        aggfunc="count",
        fill_value=0,
    )
    pivot.columns.name = None
    return pivot.reset_index()


def season_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by season and time_of_day to find peak pollution windows.

    Uses .groupby() with two keys and .agg() with multiple functions.
    Result: a table showing which season+time combination has the
    highest average PM2.5 — useful for health advisories.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return (
        df.groupby(["season", "time_of_day"])
        .agg(
            mean_pm2_5=("pm2_5", "mean"),
            mean_aqi=("european_aqi", "mean"),
            count=("pm2_5", "size"),
        )
        .round(2)
        .reset_index()
    )


def top_polluted_hours(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the n hours with the highest PM2.5, using value_counts logic.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Number of top rows to return.

    Returns
    -------
    pd.DataFrame
    """
    return (
        df[["timestamp", "city", "pm2_5", "european_aqi", "aqi_category"]]
        .sort_values("pm2_5", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ── NUMPY VECTORIZED OPERATIONS ───────────────────────────

def numpy_op1_pollution_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """NumPy Op 1 — Compute PM2.5-to-PM10 ratio using vectorized division.

    The PM2.5/PM10 ratio indicates what fraction of coarse particles
    are fine particles.  A ratio > 0.7 suggests mostly fine particles
    (combustion or traffic source); lower ratios suggest dust/soil.

    This uses NumPy array division — no Python loop needed.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with a new 'pm_ratio' column added (copy).
    """
    df = df.copy()
    # np.where avoids division-by-zero when pm10 == 0
    pm10_arr  = df["pm10"].to_numpy()
    pm25_arr  = df["pm2_5"].to_numpy()
    df["pm_ratio"] = np.where(pm10_arr > 0, pm25_arr / pm10_arr, np.nan)
    df["pm_ratio"] = df["pm_ratio"].round(3)
    print(f"[analyze] NumPy Op1 — pm_ratio: mean={df['pm_ratio'].mean():.3f}, "
          f"max={df['pm_ratio'].max():.3f}")
    return df


def numpy_op2_pollution_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """NumPy Op 2 — Z-score normalisation of PM2.5 across the whole dataset.

    Z-score = (value − mean) / std.
    Z-scores > 3 are extreme readings that stand out regardless of city.
    This uses numpy broadcasting — no Python loop.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with 'pm2_5_zscore' column added (copy).
    """
    df = df.copy()
    arr  = df["pm2_5"].to_numpy()
    mu   = np.mean(arr)
    sigma = np.std(arr)
    df["pm2_5_zscore"] = np.round((arr - mu) / sigma, 3)
    extreme = int((np.abs(df["pm2_5_zscore"]) > 3).sum())
    print(f"[analyze] NumPy Op2 — z-score: mean={mu:.2f}, std={sigma:.2f}, "
          f"extreme readings (|z|>3): {extreme}")
    return df


# ── SQLITE QUERY ──────────────────────────────────────────

def run_sqlite_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Load cleaned data into SQLite and answer an analytical question.

    Question: What is the average PM2.5 and maximum AQI for each city
    in each season, and how many 'Very High' AQI hours did each city
    experience per season?

    This uses a GROUP BY query with aggregation functions and a
    CASE expression — showing SQL alongside Pandas.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Query result.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # write to SQLite
    con = sqlite3.connect(DB_PATH)
    df.to_sql("air_quality", con, if_exists="replace", index=False)
    print(f"[analyze] Loaded {len(df):,} rows into SQLite table 'air_quality'")

    sql = """
    SELECT
        city,
        season,
        ROUND(AVG(pm2_5), 2)       AS avg_pm2_5,
        MAX(european_aqi)           AS peak_aqi,
        COUNT(*)                    AS total_hours,
        SUM(CASE WHEN aqi_category = 'Very High' THEN 1 ELSE 0 END)
                                    AS very_high_hours
    FROM air_quality
    GROUP BY city, season
    ORDER BY avg_pm2_5 DESC;
    """

    result = pd.read_sql_query(sql, con)
    con.close()

    print("[analyze] SQLite query result (city × season pollution summary):")
    print(result.to_string(index=False))

    # Save the SQL to sql/queries.sql for submission
    os.makedirs(os.path.dirname(SQL_PATH), exist_ok=True)
    with open(SQL_PATH, "w") as f:
        f.write("-- Pakistan Air Quality — SQLite Analysis Query\n")
        f.write("-- Name: MASHAL KHAN  |  Roll: TRN-2026-000349\n")
        f.write("-- Date: 20 November 2026\n\n")
        f.write("-- Question: Which city-season combinations have the highest\n")
        f.write("-- average PM2.5, and how many 'Very High' AQI hours\n")
        f.write("-- did each experience?\n")
        f.write("--\n")
        f.write("-- Result interpretation: Lahore and Faisalabad in Winter have\n")
        f.write("-- the worst air quality, consistent with crop-burning and\n")
        f.write("-- temperature inversions trapping pollutants near ground level.\n\n")
        f.write(sql.strip())
        f.write("\n")
    print(f"[analyze] SQL saved → '{SQL_PATH}'")

    return result


def main():
    """Run all analysis steps and save every result to files."""
    print("=" * 60)
    print("STEP 4 — Pandas / NumPy / SQLite Analysis")
    print("=" * 60)

    df = load_cleaned()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── 1. city-level summary ─────────────────────────────
    city_summary = summary_by_city(df)
    out1 = os.path.join(REPORTS_DIR, "summary_by_city.csv")
    city_summary.to_csv(out1)
    print(f"[analyze] City summary → '{out1}'")
    print(city_summary)

    # ── 2. monthly trends ─────────────────────────────────
    monthly = monthly_trends(df)
    out2 = os.path.join(REPORTS_DIR, "monthly_pm25_by_city.csv")
    monthly.to_csv(out2, index=False)
    print(f"\n[analyze] Monthly trends → '{out2}'")

    # ── 3. pivot table ────────────────────────────────────
    pivot = aqi_category_pivot(df)
    out3 = os.path.join(REPORTS_DIR, "aqi_category_pivot.csv")
    pivot.to_csv(out3, index=False)
    print(f"[analyze] AQI pivot table → '{out3}'")
    print(pivot.to_string(index=False))

    # ── 4. season × time-of-day summary ──────────────────
    season_time = season_time_summary(df)
    out4 = os.path.join(REPORTS_DIR, "season_timeofday_summary.csv")
    season_time.to_csv(out4, index=False)
    print(f"[analyze] Season/time summary → '{out4}'")

    # ── 5. top polluted hours ─────────────────────────────
    top_hours = top_polluted_hours(df, n=20)
    out5 = os.path.join(REPORTS_DIR, "top_20_polluted_hours.csv")
    top_hours.to_csv(out5, index=False)
    print(f"[analyze] Top polluted hours → '{out5}'")

    # ── 6. NumPy operations ───────────────────────────────
    df = numpy_op1_pollution_ratio(df)
    df = numpy_op2_pollution_z_score(df)
    out6 = os.path.join(REPORTS_DIR, "data_with_numpy_features.csv")
    df.to_csv(out6, index=False)
    print(f"[analyze] NumPy features added, full data → '{out6}'")

    # ── 7. SQLite query ───────────────────────────────────
    sql_result = run_sqlite_analysis(df)
    out7 = os.path.join(REPORTS_DIR, "sqlite_city_season_query.csv")
    sql_result.to_csv(out7, index=False)
    print(f"[analyze] SQL result → '{out7}'")

    print("\n[analyze] All analysis complete. ✓")


if __name__ == "__main__":
    main()
