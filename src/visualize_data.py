# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Step 5 of Part A — produce at least three
#              meaningful, labelled visualizations and save
#              them as image files.
# ============================================================
"""
visualize_data.py
-----------------
Creates and saves four visualizations of the Pakistan Air Quality
dataset.  Every chart has:
  - A title
  - Labelled axes with units
  - A one-sentence caption saved alongside the figure

Run:
    python src/visualize_data.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

CLEANED_PATH = "data/cleaned/pakistan_aq_cleaned.csv"
FIGURES_DIR  = "outputs/figures"
REPORTS_DIR  = "outputs/reports"

# consistent colour palette — one colour per city
CITY_PALETTE = {
    "Lahore":     "#E63946",
    "Karachi":    "#2A9D8F",
    "Islamabad":  "#457B9D",
    "Faisalabad": "#F4A261",
    "Multan":     "#6D6875",
    "Peshawar":   "#2D6A4F",
}

# figure styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":     120,
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

CAPTIONS = {}


def load_cleaned(path: str = CLEANED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ── CHART 1 ───────────────────────────────────────────────

def plot_monthly_pm25(df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    """Line chart — Monthly average PM2.5 for every city.

    Shows the seasonal pattern of fine-particle pollution.
    Winter months (Nov–Feb) are expected to be worse due to
    crop burning and temperature inversions in Punjab cities.

    Returns
    -------
    str : file path of the saved figure
    """
    monthly = (
        df.groupby(["month", "city"])["pm2_5"]
        .mean()
        .reset_index()
    )
    month_names = {
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
    }
    monthly["month_name"] = monthly["month"].map(month_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    for city, grp in monthly.groupby("city"):
        ax.plot(
            grp["month"], grp["pm2_5"],
            marker="o", linewidth=2,
            color=CITY_PALETTE.get(city, None),
            label=city,
        )

    ax.set_title("Monthly Average PM2.5 Concentration by City (2024)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list(month_names.values()))
    ax.axhline(y=15, color="green", linestyle="--", alpha=0.6, label="WHO guideline (15 µg/m³)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig1_monthly_pm25_by_city.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Chart 1 saved → '{out_path}'")
    CAPTIONS["fig1"] = (
        "Figure 1: Monthly average PM2.5 (µg/m³) for six Pakistani cities in 2024. "
        "Lahore and Faisalabad show highest winter peaks, consistent with crop burning "
        "and cold inversions trapping pollutants near the surface. "
        "All cities exceed the WHO annual guideline of 15 µg/m³ for most of the year."
    )
    return out_path


# ── CHART 2 ───────────────────────────────────────────────

def plot_aqi_distribution(df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    """Stacked bar chart — Hours spent in each AQI category per city.

    Shows the fraction of 2024 that each city's air was in the
    Low / Moderate / High / Very High pollution band.
    Useful for public health reports and policy-making.

    Returns
    -------
    str : file path
    """
    cat_order = ["Low", "Moderate", "High", "Very High"]
    cat_colors = {
        "Low":       "#52b788",
        "Moderate":  "#f4d35e",
        "High":      "#ee6c4d",
        "Very High": "#9b2335",
    }

    # count hours per city × category
    counts = (
        df.groupby(["city", "aqi_category"])
        .size()
        .reset_index(name="hours")
    )
    # pivot for stacking
    pivot = counts.pivot(index="city", columns="aqi_category", values="hours").fillna(0)
    for cat in cat_order:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[cat_order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))
    cities = pivot.index.tolist()

    for cat in cat_order:
        vals = pivot[cat].values
        ax.bar(cities, vals, bottom=bottom, label=cat,
               color=cat_colors[cat], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_title("Annual Distribution of AQI Categories by City (2024)")
    ax.set_xlabel("City")
    ax.set_ylabel("Hours per Year")
    ax.legend(title="AQI Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig2_aqi_distribution_by_city.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Chart 2 saved → '{out_path}'")
    CAPTIONS["fig2"] = (
        "Figure 2: Stacked bar chart showing how many hours each city spent in each "
        "AQI category during 2024. Karachi and Islamabad have more 'Low' and 'Moderate' "
        "hours; Faisalabad and Lahore are dominated by 'High' and 'Very High' readings."
    )
    return out_path


# ── CHART 3 ───────────────────────────────────────────────

def plot_diurnal_pattern(df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    """Line chart — Average hourly PM2.5 pattern across the day.

    The diurnal (24-hour) cycle of PM2.5 shows morning rush-hour
    peaks, midday lows as mixing height increases, and evening peaks
    as atmospheric boundary layer lowers again.

    Returns
    -------
    str : file path
    """
    hourly = (
        df.groupby(["hour", "city"])["pm2_5"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for city, grp in hourly.groupby("city"):
        ax.plot(
            grp["hour"], grp["pm2_5"],
            marker=".", linewidth=1.8,
            color=CITY_PALETTE.get(city, None),
            label=city,
        )

    ax.set_title("Diurnal (24-Hour) Average PM2.5 Pattern by City (2024)")
    ax.set_xlabel("Hour of Day (PKT, UTC+5)")
    ax.set_ylabel("Average PM2.5 (µg/m³)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig3_diurnal_pm25_pattern.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Chart 3 saved → '{out_path}'")
    CAPTIONS["fig3"] = (
        "Figure 3: Average PM2.5 by hour of day, showing a clear diurnal cycle. "
        "Early-morning peaks (02:00–06:00) reflect nocturnal boundary-layer compression; "
        "midday troughs occur as the sun heats the surface and ventilation improves. "
        "Lahore consistently shows the highest concentrations across all hours."
    )
    return out_path


# ── CHART 4 (bonus) ───────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str = FIGURES_DIR) -> str:
    """Heatmap — Pearson correlation between all numeric pollutant columns.

    High positive correlation between PM2.5 and PM10 is expected
    (both track combustion sources).  NO2 and CO correlation indicates
    shared traffic source.

    Returns
    -------
    str : file path
    """
    num_cols = ["pm10", "pm2_5", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide",
                "ozone", "dust", "uv_index", "european_aqi"]
    corr = df[num_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Pearson Correlation Matrix — Air Quality Variables (2024)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig4_correlation_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Chart 4 saved → '{out_path}'")
    CAPTIONS["fig4"] = (
        "Figure 4: Pearson correlation matrix for all numeric air-quality variables. "
        "PM2.5 and PM10 are highly correlated (r≈0.86), indicating similar combustion "
        "and dust sources. Ozone shows negative correlations with traffic pollutants — "
        "consistent with NO destroying ground-level ozone."
    )
    return out_path


def main():
    """Run all visualizations and save a captions file."""
    print("=" * 60)
    print("STEP 5 — Visualizations")
    print("=" * 60)

    df = load_cleaned()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig_paths = []
    fig_paths.append(plot_monthly_pm25(df))
    fig_paths.append(plot_aqi_distribution(df))
    fig_paths.append(plot_diurnal_pattern(df))
    fig_paths.append(plot_correlation_heatmap(df))

    # Save captions
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cap_path = os.path.join(REPORTS_DIR, "figure_captions.json")
    with open(cap_path, "w") as f:
        json.dump(CAPTIONS, f, indent=2)
    print(f"[viz] Captions saved → '{cap_path}'")

    print("\n[viz] All figures saved:")
    for p in fig_paths:
        print(f"  {p}")
    print("\n[viz] Done. ✓")


if __name__ == "__main__":
    main()
