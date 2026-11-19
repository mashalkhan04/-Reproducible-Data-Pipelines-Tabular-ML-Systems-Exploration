# ============================================================
# Name       : MASHAL KHAN
# Roll Number: TRN-2026-000349
# Date       : 20 November 2026
# Purpose    : Part C1 — SQLite Indexing Experiment.
#              Demonstrates that adding an index on (city, season)
#              speeds up the GROUP BY query and shows how to use
#              EXPLAIN QUERY PLAN to verify index usage.
# ============================================================
"""
sqlite_feed_experiment.py  (Part C, Task C1)
--------------------------------------------
Experiment:
  1. Load cleaned data into SQLite (in-memory).
  2. Run the city × season GROUP BY query WITHOUT an index — time it.
  3. Add an index on (city, season).
  4. Run the same query WITH the index — time it.
  5. Use EXPLAIN QUERY PLAN to confirm the index is being used.
  6. Write results to docs/exploration.md (appended) and
     outputs/reports/sqlite_index_experiment.json.

This satisfies Part C1: "SQLite Streaming Feed" / engineering exploration.

Run:
    python src/sqlite_feed_experiment.py
"""

import sqlite3
import time
import json
import os
import pandas as pd

CLEANED_PATH  = "data/cleaned/pakistan_aq_cleaned.csv"
REPORTS_DIR   = "outputs/reports"
RESULTS_PATH  = os.path.join(REPORTS_DIR, "sqlite_index_experiment.json")

QUERY = """
SELECT
    city,
    season,
    ROUND(AVG(pm2_5), 2)                                          AS avg_pm2_5,
    MAX(european_aqi)                                             AS peak_aqi,
    COUNT(*)                                                      AS total_hours,
    SUM(CASE WHEN aqi_category = 'Very High' THEN 1 ELSE 0 END)  AS very_high_hours
FROM air_quality
GROUP BY city, season
ORDER BY avg_pm2_5 DESC;
"""

EXPLAIN_QUERY = "EXPLAIN QUERY PLAN " + QUERY


def load_to_sqlite(df: pd.DataFrame, conn: sqlite3.Connection):
    """Write DataFrame to 'air_quality' table in the given connection."""
    df.to_sql("air_quality", conn, if_exists="replace", index=False)


def run_timed(conn: sqlite3.Connection, query: str, label: str, runs: int = 5) -> float:
    """Run query multiple times, return mean elapsed time (ms)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        conn.execute(query).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    print(f"  {label:35s}: {mean_ms:.2f} ms  (avg of {runs} runs)")
    return round(mean_ms, 3)


def explain_plan(conn: sqlite3.Connection) -> list[str]:
    """Return EXPLAIN QUERY PLAN output lines."""
    rows = conn.execute(EXPLAIN_QUERY).fetchall()
    return [str(r) for r in rows]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("PART C1 — SQLite Indexing Experiment")
    print("=" * 60)

    df = pd.read_csv(CLEANED_PATH)
    print(f"[exp] Loaded {len(df):,} rows")

    # ── Without index ─────────────────────────────────────
    conn_no_idx = sqlite3.connect(":memory:")
    load_to_sqlite(df, conn_no_idx)

    plan_no_idx = explain_plan(conn_no_idx)
    print("\n[exp] EXPLAIN QUERY PLAN (no index):")
    for line in plan_no_idx:
        print(f"  {line}")

    t_no_idx = run_timed(conn_no_idx, QUERY, "Without index (full table scan)")
    conn_no_idx.close()

    # ── With index ────────────────────────────────────────
    conn_idx = sqlite3.connect(":memory:")
    load_to_sqlite(df, conn_idx)
    conn_idx.execute(
        "CREATE INDEX idx_city_season ON air_quality (city, season);"
    )

    plan_idx = explain_plan(conn_idx)
    print("\n[exp] EXPLAIN QUERY PLAN (with index):")
    for line in plan_idx:
        print(f"  {line}")

    t_idx = run_timed(conn_idx, QUERY, "With index (city, season)")
    conn_idx.close()

    # ── Results ───────────────────────────────────────────
    speedup = round(t_no_idx / t_idx, 2) if t_idx > 0 else "N/A"
    print(f"\n[exp] Speedup: {speedup}×")

    results = {
        "without_index_ms":    t_no_idx,
        "with_index_ms":       t_idx,
        "speedup_factor":      speedup,
        "explain_no_index":    plan_no_idx,
        "explain_with_index":  plan_idx,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp] Results saved → '{RESULTS_PATH}'")
    print("[exp] Done. ✓")


if __name__ == "__main__":
    main()
