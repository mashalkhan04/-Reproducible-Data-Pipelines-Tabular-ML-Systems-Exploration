# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Unit tests for src/validate_data.py
# ============================================================
"""
test_validation.py
------------------
Tests every validator in validate_data.py:
  - Each passing case should NOT raise
  - Each failing case should raise ValueError
  - Quality gate raises SystemExit if pass rate < 95 %
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.validate_data import (
    validate_no_nulls,
    validate_non_negative,
    validate_pm_relationship,
    validate_timestamp_range,
    validate_city_set,
    validate_aqi_category_completeness,
    validate_coordinate_uniqueness,
    run_all_validations,
)

# Constants defined locally (match what validate_data uses internally)
REQUIRED_COLS   = ["timestamp", "pm10", "pm2_5", "city", "european_aqi"]
POLLUTANT_COLS  = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "dust", "uv_index"]
EXPECTED_CITIES = {"Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar"}


# ── Helper ─────────────────────────────────────────────────

def make_valid_df(n: int = 10) -> pd.DataFrame:
    """Return a minimal DataFrame that passes all validators."""
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar"]
    return pd.DataFrame({
        "timestamp":        pd.date_range("2024-01-15", periods=n, freq="h"),
        "pm10":             np.full(n, 60.0),
        "pm2_5":            np.full(n, 30.0),
        "carbon_monoxide":  np.full(n, 500.0),
        "nitrogen_dioxide": np.full(n, 20.0),
        "sulphur_dioxide":  np.full(n, 5.0),
        "ozone":            np.full(n, 80.0),
        "dust":             np.full(n, 10.0),
        "uv_index":         np.full(n, 2.0),
        "european_aqi":     np.full(n, 60),
        "city":             [cities[i % len(cities)] for i in range(n)],
        "latitude":         np.full(n, 31.0),
        "longitude":        [67.0 + i * 1.5 for i in range(n)],  # unique per row
        "aqi_category":     ["Moderate"] * n,
    })


# ── validate_no_nulls ──────────────────────────────────────

class TestValidateNoNulls:
    def test_passes_with_no_nulls(self):
        df = make_valid_df()
        # Should not raise
        validate_no_nulls(df, ["pm2_5", "city"])

    def test_raises_on_null(self):
        df = make_valid_df()
        df.loc[0, "pm2_5"] = np.nan
        with pytest.raises(ValueError, match="pm2_5"):
            validate_no_nulls(df, ["pm2_5"])

    def test_raises_on_null_in_city(self):
        df = make_valid_df()
        df.loc[0, "city"] = None
        with pytest.raises(ValueError):
            validate_no_nulls(df, ["city"])


# ── validate_non_negative ──────────────────────────────────

class TestValidateNonNegative:
    def test_passes_all_positive(self):
        df = make_valid_df()
        validate_non_negative(df, ["pm10", "pm2_5"])

    def test_passes_zero_values(self):
        df = make_valid_df()
        df["ozone"] = 0.0
        validate_non_negative(df, ["ozone"])

    def test_raises_on_negative(self):
        df = make_valid_df()
        df.loc[2, "pm10"] = -5.0
        with pytest.raises(ValueError, match="pm10"):
            validate_non_negative(df, ["pm10"])


# ── validate_pm_relationship ───────────────────────────────

class TestValidatePmRelationship:
    def test_passes_pm25_less_than_pm10(self):
        df = make_valid_df()
        df["pm2_5"] = 30.0
        df["pm10"]  = 60.0
        validate_pm_relationship(df)

    def test_passes_equal_pm25_pm10(self):
        df = make_valid_df()
        df["pm2_5"] = 60.0
        df["pm10"]  = 60.0
        validate_pm_relationship(df)

    def test_raises_when_pm25_exceeds_pm10(self):
        df = make_valid_df()
        df.loc[0, "pm2_5"] = 100.0
        df.loc[0, "pm10"]  = 50.0
        with pytest.raises(ValueError, match="PM2.5"):
            validate_pm_relationship(df)


# ── validate_timestamp_range ───────────────────────────────

class TestValidateTimestampRange:
    def test_passes_within_2024(self):
        df = make_valid_df()
        df["timestamp"] = pd.date_range("2024-06-01", periods=len(df), freq="h")
        validate_timestamp_range(df)

    def test_raises_on_timestamp_before_2024(self):
        df = make_valid_df()
        df.loc[0, "timestamp"] = pd.Timestamp("2023-12-31")
        with pytest.raises(ValueError, match="outside"):
            validate_timestamp_range(df)

    def test_raises_on_timestamp_after_2024(self):
        df = make_valid_df()
        df.loc[0, "timestamp"] = pd.Timestamp("2025-01-01")
        with pytest.raises(ValueError, match="outside"):
            validate_timestamp_range(df)


# ── validate_city_set ──────────────────────────────────────

class TestValidateCitySet:
    def test_passes_expected_cities(self):
        df = make_valid_df()
        df["city"] = df["city"].map(
            lambda c: c if c in EXPECTED_CITIES else list(EXPECTED_CITIES)[0]
        )
        validate_city_set(df)

    def test_raises_on_unexpected_city(self):
        df = make_valid_df()
        df.loc[0, "city"] = "MadeUpCity"
        with pytest.raises(ValueError, match="MadeUpCity"):
            validate_city_set(df)


# ── validate_aqi_category_completeness ────────────────────

class TestValidateAqiCategoryCompleteness:
    def test_passes_no_nulls(self):
        df = make_valid_df()
        validate_aqi_category_completeness(df)

    def test_raises_on_null_category(self):
        df = make_valid_df()
        df.loc[0, "aqi_category"] = np.nan
        with pytest.raises(ValueError, match="aqi_category"):
            validate_aqi_category_completeness(df)


# ── validate_coordinate_uniqueness ────────────────────────

class TestValidateCoordinateUniqueness:
    def test_passes_unique_longitudes(self):
        df = make_valid_df()
        # make_valid_df uses 67 + 1.5*i — all unique
        validate_coordinate_uniqueness(df)

    def test_raises_on_duplicate_longitude(self):
        df = make_valid_df()
        # Give two different cities the same longitude
        df.loc[df["city"] == "Lahore",   "longitude"] = 71.5
        df.loc[df["city"] == "Multan",   "longitude"] = 71.5
        with pytest.raises(ValueError, match="longitude"):
            validate_coordinate_uniqueness(df)


# ── run_all_validations ────────────────────────────────────

class TestRunAllValidations:
    def test_all_pass_on_valid_df(self):
        df = make_valid_df(60)  # 6 cities × 10 rows each
        # Should not raise
        result = run_all_validations(df, threshold_pct=95.0)
        assert result["pass_rate_pct"] == 100.0

    def test_quality_gate_raises_system_exit_when_too_many_failures(self):
        """Inject multiple bad rows so several validators fail → pass rate < 95 %."""
        df = make_valid_df(60)
        # 1) introduce nulls
        df.loc[0, "pm2_5"] = np.nan
        # 2) introduce negative
        df.loc[1, "pm10"] = -99.0
        # 3) PM2.5 > PM10
        df.loc[2, "pm2_5"] = 500.0
        df.loc[2, "pm10"] = 1.0
        # 4) out-of-range timestamp
        df.loc[3, "timestamp"] = pd.Timestamp("2023-01-01")
        # 5) unexpected city
        df.loc[4, "city"] = "Atlantis"
        # 6) duplicate longitude
        df.loc[df["city"] == "Lahore",   "longitude"] = 71.5
        df.loc[df["city"] == "Multan",   "longitude"] = 71.5
        # 7) null aqi_category
        df.loc[5, "aqi_category"] = np.nan

        with pytest.raises(SystemExit):
            run_all_validations(df, threshold_pct=95.0)

    def test_all_pass_returns_dict_with_pass_rate(self):
        df = make_valid_df(12)
        result = run_all_validations(df, threshold_pct=95.0)
        assert "pass_rate_pct" in result
        assert isinstance(result["pass_rate_pct"], float)
