-- Pakistan Air Quality — SQLite Analysis Query
-- Name: MASHAL KHAN  |  Roll: TRN-2026-000349
-- Date: 20 November 2026

-- Question: Which city-season combinations have the highest
-- average PM2.5, and how many 'Very High' AQI hours
-- did each experience?
--
-- Result interpretation: Lahore and Faisalabad in Winter have
-- the worst air quality, consistent with crop-burning and
-- temperature inversions trapping pollutants near ground level.

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
