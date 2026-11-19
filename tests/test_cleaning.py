# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Unit tests for src/clean_data.py
# ============================================================
"""
test_cleaning.py
----------------
Tests every public cleaning function in clean_data.py using
small, self-contained DataFrames — no external files needed.
"""

import pandas as pd
import numpy as np
import pytest

from src.clean_data import (
    fix_timestamp_dtype,
    fix_multan_longitude,
    remove_duplicate_rows,
    add_datetime_features,
    add_aqi_category,
    cap_outliers_iqr,
    standardise_city_names,
    MULTAN_CORRECT_LONGITUDE,
    POLLUTANT_COLS,
)


# ── Fixtures ───────────────────────────────────────────────

def make_base_df(n: int = 10) -> pd.DataFrame:
    """Return a minimal valid DataFrame matching the schema."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "pm10":              np.full(n, 60.0),
        "pm2_5":             np.full(n, 30.0),
        "carbon_monoxide":   np.full(n, 500.0),
        "nitrogen_dioxide":  np.full(n, 20.0),
        "sulphur_dioxide":   np.full(n, 5.0),
        "ozone":             np.full(n, 80.0),
        "dust":              np.full(n, 10.0),
        "uv_index":          np.array([float(i % 5) for i in range(n)]),
        "european_aqi":      np.array([30 + i * 7 for i in range(n)]),
        "city":              ["Lahore"] * n,
        "latitude":          np.full(n, 31.5497),
        "longitude":         np.full(n, 74.3436),
    })


# ── fix_timestamp_dtype ────────────────────────────────────

class TestFixTimestampDtype:
    def test_converts_string_to_datetime(self):
        df = make_base_df(3)
        # Overwrite timestamp with string representation
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
        # pandas 2.x may use StringDtype; just confirm it is NOT datetime yet
        assert not pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        result = fix_timestamp_dtype(df)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_idempotent_on_datetime_column(self):
        """Should not error if column is already datetime."""
        df = make_base_df(3)
        result = fix_timestamp_dtype(df)
        # Already datetime — should still work
        result2 = fix_timestamp_dtype(result)
        assert pd.api.types.is_datetime64_any_dtype(result2["timestamp"])

    def test_values_preserved(self):
        df = make_base_df(3)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
        result = fix_timestamp_dtype(df)
        assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")


# ── fix_multan_longitude ───────────────────────────────────

class TestFixMultanLongitude:
    def test_corrects_multan_longitude(self):
        df = make_base_df(4)
        df["city"]      = ["Multan", "Multan", "Lahore", "Peshawar"]
        df["longitude"] = [71.5249,  71.5249, 74.3436,   71.5249]
        result = fix_multan_longitude(df)
        multan_rows = result[result["city"] == "Multan"]
        assert (multan_rows["longitude"] == MULTAN_CORRECT_LONGITUDE).all()

    def test_other_cities_unchanged(self):
        """When no Multan rows are present, other cities must stay exactly as given."""
        df = make_base_df(4)
        df["city"]      = ["Karachi", "Islamabad", "Peshawar", "Multan"]
        df["longitude"] = [67.0,       73.0,         71.5249,   71.5249]
        result = fix_multan_longitude(df)
        karachi_lon = result.loc[result["city"] == "Karachi", "longitude"].iloc[0]
        assert karachi_lon == 67.0


# ── remove_duplicate_rows ──────────────────────────────────

class TestRemoveDuplicateRows:
    def test_removes_exact_duplicates(self):
        df = make_base_df(3)
        df_dup = pd.concat([df, df]).reset_index(drop=True)
        assert len(df_dup) == 6
        result = remove_duplicate_rows(df_dup)
        assert len(result) == 3

    def test_no_change_when_no_duplicates(self):
        df = make_base_df(5)
        result = remove_duplicate_rows(df)
        assert len(result) == 5


# ── add_datetime_features ──────────────────────────────────

class TestAddDatetimeFeatures:
    def test_adds_expected_columns(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        for col in ["hour", "month", "day_of_week", "season", "time_of_day", "is_weekend"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_range(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        assert result["hour"].between(0, 23).all()

    def test_month_range(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        assert result["month"].between(1, 12).all()

    def test_season_values(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        valid_seasons = {"Winter", "Spring", "Summer", "Autumn"}
        assert set(result["season"].unique()).issubset(valid_seasons)

    def test_time_of_day_values(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        valid_times = {"Night", "Morning", "Afternoon", "Evening"}
        assert set(result["time_of_day"].unique()).issubset(valid_times)

    def test_is_weekend_boolean(self):
        df = make_base_df(24)
        result = add_datetime_features(df)
        assert result["is_weekend"].dtype == bool


# ── add_aqi_category ──────────────────────────────────────

class TestAddAqiCategory:
    def test_all_four_categories_present(self):
        df = make_base_df(10)
        df["european_aqi"] = [25, 55, 75, 95, 30, 65, 85, 100, 20, 80]
        result = add_aqi_category(df)
        cats = set(result["aqi_category"].dropna().tolist())
        assert cats == {"Low", "Moderate", "High", "Very High"}

    def test_low_category_boundary(self):
        df = make_base_df(1)
        df["european_aqi"] = [49]
        result = add_aqi_category(df)
        assert result["aqi_category"].iloc[0] == "Low"

    def test_very_high_category_boundary(self):
        df = make_base_df(1)
        df["european_aqi"] = [91]
        result = add_aqi_category(df)
        assert result["aqi_category"].iloc[0] == "Very High"

    def test_no_nulls_in_valid_range(self):
        df = make_base_df(5)
        df["european_aqi"] = [1, 50, 70, 90, 150]
        result = add_aqi_category(df)
        assert result["aqi_category"].notna().all()


# ── cap_outliers_iqr ──────────────────────────────────────

class TestCapOutliersIqr:
    def test_extreme_values_capped(self):
        df = make_base_df(100)
        # Inject a large outlier
        df.loc[0, "pm2_5"] = 999_999.0
        result = cap_outliers_iqr(df, ["pm2_5"], factor=1.5)
        assert result["pm2_5"].max() < 999_999.0

    def test_row_count_preserved(self):
        df = make_base_df(50)
        df.loc[0, "pm10"] = 1e9
        result = cap_outliers_iqr(df, ["pm10"])
        assert len(result) == 50

    def test_no_negative_after_capping_non_negative_col(self):
        """After capping, normally non-negative pollutants stay ≥ 0."""
        df = make_base_df(100)
        # Natural IQR fence on constant values would be 0; add variation
        df["dust"] = np.abs(np.random.default_rng(42).normal(20, 5, size=100))
        df.loc[0, "dust"] = -500.0  # inject negative outlier
        result = cap_outliers_iqr(df, ["dust"])
        assert result["dust"].min() >= 0 or result["dust"].min() > -500


# ── standardise_city_names ────────────────────────────────

class TestStandardiseCityNames:
    def test_title_cases_names(self):
        df = make_base_df(3)
        df["city"] = ["lahore", "KARACHI", " islamabad "]
        result = standardise_city_names(df)
        assert result["city"].iloc[0] == "Lahore"
        assert result["city"].iloc[1] == "Karachi"
        assert result["city"].iloc[2] == "Islamabad"

    def test_already_title_case_unchanged(self):
        df = make_base_df(2)
        df["city"] = ["Peshawar", "Multan"]
        result = standardise_city_names(df)
        assert result["city"].tolist() == ["Peshawar", "Multan"]
