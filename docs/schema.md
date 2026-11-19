# Data Schema — Pakistan Air Quality 2024

**Source:** Open-Meteo Air Quality API (https://open-meteo.com/en/docs/air-quality-api)  
**Underlying model:** ECMWF CAMS (Copernicus Atmosphere Monitoring Service)  
**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)  
**Access date:** 20 November 2026  
**Coverage:** Hourly readings for 6 Pakistani cities, 1 Jan – 31 Dec 2024  
**Rows:** 52,704  
**Columns:** 13  

---

## Column definitions

| Column | Dtype (raw) | Units | Description | Allowed range |
|--------|-------------|-------|-------------|---------------|
| timestamp | object (string) | ISO-8601 | Hourly observation time (PKT UTC+5) | 2024-01-01T00:00 – 2024-12-31T23:00 |
| pm10 | float64 | µg/m³ | Particulate matter ≤10 µm | ≥ 0 |
| pm2_5 | float64 | µg/m³ | Particulate matter ≤2.5 µm (fine particles) | ≥ 0, ≤ pm10 |
| carbon_monoxide | float64 | µg/m³ | Carbon monoxide concentration | ≥ 0 |
| nitrogen_dioxide | float64 | µg/m³ | Nitrogen dioxide | ≥ 0 |
| sulphur_dioxide | float64 | µg/m³ | Sulphur dioxide | ≥ 0 |
| ozone | float64 | µg/m³ | Ground-level ozone | ≥ 0 |
| dust | float64 | µg/m³ | Mineral dust (Saharan/desert origin) | ≥ 0 |
| uv_index | float64 | index | UV irradiance index | 0 – 11+ |
| european_aqi | int64 | index | European Air Quality Index (AQI) | 0 – 500+ |
| city | object | — | City name (6 cities) | Lahore/Karachi/Islamabad/Faisalabad/Multan/Peshawar |
| latitude | float64 | °N | City centroid latitude | 24.86 – 34.02 |
| longitude | float64 | °E | City centroid longitude | 67.00 – 74.36 |

---

## Column profiles

### `timestamp`
- **dtype:** str
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 8784

### `pm10`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 1227
- **min:** 4.3
- **max:** 527.1
- **mean:** 83.754
- **std:** 60.602
- **p25:** 43.3
- **p50:** 65.7
- **p75:** 104.3
- **zeros:** 0

### `pm2_5`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 1169
- **min:** 3.1
- **max:** 417.2
- **mean:** 51.701
- **std:** 46.529
- **p25:** 23.7
- **p50:** 36.8
- **p75:** 59.8
- **zeros:** 0

### `carbon_monoxide`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 4014
- **min:** 46.0
- **max:** 7860.0
- **mean:** 884.043
- **std:** 841.99
- **p25:** 331.0
- **p50:** 577.0
- **p75:** 1132.0
- **zeros:** 0

### `nitrogen_dioxide`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 1490
- **min:** 0.0
- **max:** 193.2
- **mean:** 29.491
- **std:** 28.073
- **p25:** 8.7
- **p50:** 19.7
- **p75:** 41.7
- **zeros:** 131

### `sulphur_dioxide`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 602
- **min:** 0.5
- **max:** 93.8
- **mean:** 11.063
- **std:** 8.296
- **p25:** 5.4
- **p50:** 8.6
- **p75:** 14.1
- **zeros:** 0

### `ozone`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 270
- **min:** 0.0
- **max:** 305.0
- **mean:** 80.1
- **std:** 56.218
- **p25:** 36.0
- **p50:** 68.0
- **p75:** 121.0
- **zeros:** 1333

### `dust`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 558
- **min:** 0.0
- **max:** 961.0
- **mean:** 38.255
- **std:** 63.659
- **p25:** 4.0
- **p50:** 15.0
- **p75:** 45.0
- **zeros:** 2237

### `uv_index`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 227
- **min:** 0.0
- **max:** 11.6
- **mean:** 1.351
- **std:** 2.258
- **p25:** 0.0
- **p50:** 0.0
- **p75:** 1.9
- **zeros:** 27424

### `european_aqi`
- **dtype:** int64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 187
- **min:** 19.0
- **max:** 225.0
- **mean:** 68.888
- **std:** 25.111
- **p25:** 51.0
- **p50:** 63.0
- **p75:** 79.0
- **zeros:** 0

### `city`
- **dtype:** str
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 6

### `latitude`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 6
- **min:** 24.861
- **max:** 34.015
- **mean:** 30.948
- **std:** 3.032
- **p25:** 30.157
- **p50:** 31.485
- **p75:** 33.684
- **zeros:** 0

### `longitude`
- **dtype:** float64
- **non_null:** 52704
- **null_count:** 0
- **null_pct:** 0.0
- **unique:** 5
- **min:** 67.001
- **max:** 74.359
- **mean:** 71.765
- **std:** 2.347
- **p25:** 71.525
- **p50:** 72.286
- **p75:** 73.135
- **zeros:** 0

---

## Data quality issues found in raw data

- DUPE   | No exact duplicate rows — OK.
- COORD  | Cities ['Multan', 'Peshawar'] share identical longitude 71.5249. This is a data-entry error; correct Multan longitude is ~71.4687°E.
- OUTL   | 'pm10' has 3,219 outliers (6.1%) outside IQR fence [-48.2, 195.8].
- OUTL   | 'pm2_5' has 4,561 outliers (8.7%) outside IQR fence [-30.4, 113.9].
- OUTL   | 'carbon_monoxide' has 3,549 outliers (6.7%) outside IQR fence [-870.5, 2333.5].
- OUTL   | 'nitrogen_dioxide' has 2,455 outliers (4.7%) outside IQR fence [-40.8, 91.2].
- OUTL   | 'sulphur_dioxide' has 2,829 outliers (5.4%) outside IQR fence [-7.6, 27.1].
- OUTL   | 'ozone' has 36 outliers (0.1%) outside IQR fence [-91.5, 248.5].
- OUTL   | 'dust' has 4,813 outliers (9.1%) outside IQR fence [-57.5, 106.5].
- OUTL   | 'uv_index' has 5,858 outliers (11.1%) outside IQR fence [-2.8, 4.8].
- OUTL   | 'european_aqi' has 2,468 outliers (4.7%) outside IQR fence [9.0, 121.0].
- ZERO   | 'nitrogen_dioxide' contains 131 zero values (0.2%). Night-time UV and ozone zeros are expected; others need review.
- ZERO   | 'ozone' contains 1,333 zero values (2.5%). Night-time UV and ozone zeros are expected; others need review.
- ZERO   | 'dust' contains 2,237 zero values (4.2%). Night-time UV and ozone zeros are expected; others need review.
- ZERO   | 'uv_index' contains 27,424 zero values (52.0%). Night-time UV and ozone zeros are expected; others need review.

---

## Notes on expected zero values

- `uv_index = 0` at night is **physically correct** (no sunlight).
- `ozone = 0` at night is possible in heavily polluted urban air due to NO titration.
- `nitrogen_dioxide = 0` or near-zero is plausible in areas with low traffic at night.
- `dust = 0` indicates calm, dust-free conditions — valid.

---

*Generated automatically by `src/inspect_data.py`.*