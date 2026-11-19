# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Step 1 of Part A — raw data inspection and
#              schema documentation.
# ============================================================
"""
inspect_data.py
---------------
Loads the raw dataset, prints a detailed profile of every column,
and writes docs/schema.md so anyone reading the repo knows exactly
what the data looks like before cleaning.

Run:
    python src/inspect_data.py
"""

import os
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────
RAW_PATH    = "data/raw/pakistan_air_quality_2024.csv"
SCHEMA_PATH = "docs/schema.md"
REPORTS_DIR = "outputs/reports"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame.

    Parameters
    ----------
    path : str
        File path to the raw CSV.

    Returns
    -------
    pd.DataFrame
        The loaded data, every column still in its raw state.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at '{path}'.\n"
            "Run the download script first:\n"
            "  python src/download_data.py"
        )
    df = pd.read_csv(path)
    print(f"[inspect] Loaded {len(df):,} rows × {len(df.columns)} columns from '{path}'")
    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Compute a detailed profile of every column.

    For each column the profile contains:
    - dtype, non-null count, null count, null %, unique count
    - For numeric columns: min, max, mean, std, 25th/50th/75th percentile
    - Zero-value count (potential sensor-gap indicator)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to profile.

    Returns
    -------
    dict
        Mapping of column name → dict of stats.
    """
    profile = {}
    for col in df.columns:
        series = df[col]
        info = {
            "dtype":      str(series.dtype),
            "non_null":   int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_pct":   round(100 * series.isna().mean(), 2),
            "unique":     int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            info.update({
                "min":    round(float(series.min()), 3),
                "max":    round(float(series.max()), 3),
                "mean":   round(float(series.mean()), 3),
                "std":    round(float(series.std()), 3),
                "p25":    round(float(series.quantile(0.25)), 3),
                "p50":    round(float(series.quantile(0.50)), 3),
                "p75":    round(float(series.quantile(0.75)), 3),
                "zeros":  int((series == 0).sum()),
            })
        elif series.dtype == object:
            info["sample_values"] = series.dropna().unique()[:5].tolist()
        profile[col] = info
    return profile


def detect_quality_issues(df: pd.DataFrame) -> list:
    """Scan the raw DataFrame and return a list of quality-issue descriptions.

    Checks performed
    ----------------
    1. Columns stored as wrong dtype (e.g. timestamp as string).
    2. Exact duplicate rows.
    3. Duplicate (timestamp, city) pairs.
    4. Coordinates that are identical across two different cities.
    5. Statistical outliers (IQR × 1.5 fence) per numeric column.
    6. Columns that contain only zeros for certain cities (sensor gap).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    list[str]
        Human-readable descriptions of every issue found.
    """
    issues = []

    # 1. dtype of timestamp
    if df["timestamp"].dtype == object:
        issues.append(
            "DTYPE  | 'timestamp' column is stored as string (object), "
            "not datetime64. Needs conversion."
        )

    # 2. exact duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        issues.append(f"DUPE   | {n_dup:,} exact duplicate rows found.")
    else:
        issues.append("DUPE   | No exact duplicate rows — OK.")

    # 3. duplicate (timestamp, city)
    n_key_dup = df.duplicated(subset=["timestamp", "city"]).sum()
    if n_key_dup > 0:
        issues.append(
            f"DUPE   | {n_key_dup:,} duplicate (timestamp, city) pairs — "
            "possible double-upload."
        )

    # 4. coordinate anomaly (two cities with identical longitude)
    coords = df[["city", "longitude"]].drop_duplicates()
    lon_groups = coords.groupby("longitude")["city"].apply(list)
    for lon, cities in lon_groups.items():
        if len(cities) > 1:
            issues.append(
                f"COORD  | Cities {cities} share identical longitude {lon}. "
                "This is a data-entry error; correct Multan longitude is ~71.4687°E."
            )

    # 5. outliers per numeric column
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if col in ("latitude", "longitude"):
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        fence_lo = Q1 - 1.5 * IQR
        fence_hi = Q3 + 1.5 * IQR
        n_out = int(((df[col] < fence_lo) | (df[col] > fence_hi)).sum())
        pct    = round(100 * n_out / len(df), 1)
        if n_out > 0:
            issues.append(
                f"OUTL   | '{col}' has {n_out:,} outliers ({pct}%) "
                f"outside IQR fence [{fence_lo:.1f}, {fence_hi:.1f}]."
            )

    # 6. zero-value counts for pollutant columns
    pollutants = ["nitrogen_dioxide", "ozone", "dust", "uv_index"]
    for col in pollutants:
        zeros = int((df[col] == 0).sum())
        if zeros > 0:
            issues.append(
                f"ZERO   | '{col}' contains {zeros:,} zero values "
                f"({100*zeros/len(df):.1f}%). "
                "Night-time UV and ozone zeros are expected; others need review."
            )

    return issues


def write_schema_md(profile: dict, issues: list, df: pd.DataFrame,
                    out_path: str = SCHEMA_PATH) -> None:
    """Write a human-readable schema Markdown file.

    Parameters
    ----------
    profile : dict
        Output of :func:`profile_dataframe`.
    issues : list
        Output of :func:`detect_quality_issues`.
    df : pd.DataFrame
        The raw DataFrame (used for a few extra stats).
    out_path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "# Data Schema — Pakistan Air Quality 2024",
        "",
        "**Source:** Open-Meteo Air Quality API (https://open-meteo.com/en/docs/air-quality-api)  ",
        "**Underlying model:** ECMWF CAMS (Copernicus Atmosphere Monitoring Service)  ",
        "**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)  ",
        "**Access date:** 20 November 2026  ",
        "**Coverage:** Hourly readings for 6 Pakistani cities, 1 Jan – 31 Dec 2024  ",
        f"**Rows:** {len(df):,}  ",
        f"**Columns:** {len(df.columns)}  ",
        "",
        "---",
        "",
        "## Column definitions",
        "",
        "| Column | Dtype (raw) | Units | Description | Allowed range |",
        "|--------|-------------|-------|-------------|---------------|",
        "| timestamp | object (string) | ISO-8601 | Hourly observation time (PKT UTC+5) | 2024-01-01T00:00 – 2024-12-31T23:00 |",
        "| pm10 | float64 | µg/m³ | Particulate matter ≤10 µm | ≥ 0 |",
        "| pm2_5 | float64 | µg/m³ | Particulate matter ≤2.5 µm (fine particles) | ≥ 0, ≤ pm10 |",
        "| carbon_monoxide | float64 | µg/m³ | Carbon monoxide concentration | ≥ 0 |",
        "| nitrogen_dioxide | float64 | µg/m³ | Nitrogen dioxide | ≥ 0 |",
        "| sulphur_dioxide | float64 | µg/m³ | Sulphur dioxide | ≥ 0 |",
        "| ozone | float64 | µg/m³ | Ground-level ozone | ≥ 0 |",
        "| dust | float64 | µg/m³ | Mineral dust (Saharan/desert origin) | ≥ 0 |",
        "| uv_index | float64 | index | UV irradiance index | 0 – 11+ |",
        "| european_aqi | int64 | index | European Air Quality Index (AQI) | 0 – 500+ |",
        "| city | object | — | City name (6 cities) | Lahore/Karachi/Islamabad/Faisalabad/Multan/Peshawar |",
        "| latitude | float64 | °N | City centroid latitude | 24.86 – 34.02 |",
        "| longitude | float64 | °E | City centroid longitude | 67.00 – 74.36 |",
        "",
        "---",
        "",
        "## Column profiles",
        "",
    ]

    for col, info in profile.items():
        lines.append(f"### `{col}`")
        for key, val in info.items():
            if key == "sample_values":
                lines.append(f"- **sample values:** {val}")
            else:
                lines.append(f"- **{key}:** {val}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Data quality issues found in raw data",
        "",
    ]
    for issue in issues:
        lines.append(f"- {issue}")

    lines += [
        "",
        "---",
        "",
        "## Notes on expected zero values",
        "",
        "- `uv_index = 0` at night is **physically correct** (no sunlight).",
        "- `ozone = 0` at night is possible in heavily polluted urban air due to NO titration.",
        "- `nitrogen_dioxide = 0` or near-zero is plausible in areas with low traffic at night.",
        "- `dust = 0` indicates calm, dust-free conditions — valid.",
        "",
        "---",
        "",
        "*Generated automatically by `src/inspect_data.py`.*",
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[inspect] Schema written to '{out_path}'")


def main():
    """Run the full inspection pipeline and save the schema."""
    print("=" * 60)
    print("STEP 1 — Raw Data Inspection")
    print("=" * 60)

    df = load_raw()
    profile = profile_dataframe(df)
    issues = detect_quality_issues(df)

    print(f"\n[inspect] Columns  : {list(df.columns)}")
    print(f"[inspect] Dtypes   :\n{df.dtypes.to_string()}")
    print(f"\n[inspect] Quality issues detected ({len(issues)}):")
    for iss in issues:
        print(f"  → {iss}")

    write_schema_md(profile, issues, df)

    # save a quick summary report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "cities": df["city"].unique().tolist(),
        "date_range_raw": [df["timestamp"].min(), df["timestamp"].max()],
        "issues_count": len(issues),
    }
    import json
    summary_path = os.path.join(REPORTS_DIR, "inspection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[inspect] Summary saved to '{summary_path}'")
    print("\n[inspect] Done. ✓")
    return df


if __name__ == "__main__":
    main()
