# Assumptions & Decisions

**Name:** MASHAL KHAN  
**Roll Number:** TRN-2026-000349  
**Date:** 20 November 2026

---

## Dataset

| Assumption | Rationale |
|---|---|
| Data source: Open-Meteo Air Quality API (ECMWF CAMS model) | Free, no API key required, CC BY 4.0 licence, provides hourly data for Pakistan cities. The assignment requires a "Pakistan Air Quality" dataset — CAMS is the state-of-art global atmospheric reanalysis. |
| 6 cities: Lahore, Karachi, Islamabad, Faisalabad, Multan, Peshawar | These are Pakistan's six largest cities and represent north/south, industrial/residential, coastal/inland diversity. |
| Hourly resolution, full year 2024 | 52,704 rows = 6 cities × 8,784 hours (2024 is a leap year). Provides enough data for robust seasonal and diurnal analysis. |
| Licence: CC BY 4.0 (ECMWF CAMS) | Attribution required. Cited in README. No restrictions on academic use. |

---

## AQI Categorisation

The European AQI standard has 6 levels (Good / Fair / Moderate / Poor / Very Poor / Extremely Poor).  
Our dataset's range yielded only 3 "Good" rows, making 6 classes impractical for ML.

**Decision:** Merge into 4 practical categories:
- **Low** — AQI 0–49  
- **Moderate** — AQI 50–69  
- **High** — AQI 70–89  
- **Very High** — AQI ≥ 90  

This gives a reasonably balanced distribution (22–38 % per class) and matches public health communication levels used in Pakistan.

---

## Outlier Handling

**Decision: Winsorize (IQR capping) rather than delete.**

Rationale:
- Preserves row count (important for time-series completeness).
- Extreme values in AQ data are often physically real (e.g., crop-burning events, Saharan dust intrusions) but should be dampened so they do not dominate model training.
- IQR factor = 1.5 (standard Tukey fence). Factor 3.0 would retain more extremes but allow physically impossible negatives.

---

## Coordinate Error

Multan and Peshawar were assigned identical longitude `71.5249°` in the raw download due to a copy-paste error in the data-fetch script. Correct Multan longitude is `71.4687°E`. Fixed in `clean_data.py::fix_multan_longitude()`. Flagged as a data quality issue in `docs/schema.md`.

---

## Zero Values

Zero readings in `uv_index` (52 % of rows) and `ozone` (2.5 %) are **physically valid** — UV is zero at night, and ozone can drop to near-zero in certain conditions. These are **not** imputed. Nitrogen dioxide and dust zeros are also retained as valid low-concentration readings.

---

## ML Model

| Decision | Choice | Reason |
|---|---|---|
| Train/test split | 80 / 20, stratified | Stratification preserves class balance in both splits. |
| Preprocessing | Median imputation + StandardScaler for numerics, OHE for categoricals | Robust to any remaining nulls; OHE is correct for low-cardinality strings. |
| Three classifiers | Logistic Regression, Random Forest, Gradient Boosting | Covers linear, ensemble-bagging, ensemble-boosting — different bias-variance profiles. |
| Perfect accuracy note | “I removed european_aqi because it leaked the target. I also removed PM2.5 and PM10 to test whether cheaper gas sensors alone can predict air quality.”