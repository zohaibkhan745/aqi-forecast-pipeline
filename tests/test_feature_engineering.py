from datetime import datetime, timedelta, timezone
import unittest

from src.feature_engineering import (
    EXPECTED_COLUMNS,
    build_feature_row,
    compute_derived_features,
    compute_time_features,
)


class TestFeatureEngineering(unittest.TestCase):

    def test_compute_time_features(self):
        # 2026-07-31 14:30:00 UTC (Friday -> day_of_week=4, month=7 -> summer)
        ts_str = "2026-07-31T14:30:00+00:00"
        feats = compute_time_features(ts_str)

        self.assertEqual(feats["hour"], 14)
        self.assertEqual(feats["day_of_week"], 4)
        self.assertEqual(feats["day_of_month"], 31)
        self.assertEqual(feats["month"], 7)
        self.assertEqual(feats["is_weekend"], 0)
        self.assertEqual(feats["season"], "summer")

    def test_normal_case_with_full_history(self):
        now = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)
        current = {
            "fetched_at": now.isoformat(),
            "city": "London",
            "aqi": 100,
            "pm25": 25.0,
            "pm10": 50.0,
            "o3": 10.0,
            "no2": 15.0,
            "so2": 5.0,
            "co": 1.0,
            "temperature": 20.0,
            "humidity": 60.0,
            "wind_speed": 4.0,
            "pressure": 1012.0,
        }

        # Create history snapshots: 1h ago (aqi=80), 2h ago (aqi=60), 24h ago (aqi=50)
        history = [
            {"fetched_at": (now - timedelta(hours=24)).isoformat(), "aqi": 50},
            {"fetched_at": (now - timedelta(hours=2)).isoformat(), "aqi": 60},
            {"fetched_at": (now - timedelta(hours=1)).isoformat(), "aqi": 80},
        ]

        feature_row = build_feature_row(current, history)

        # Check column completeness
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, feature_row)

        # Check ratio (25 / 50 = 0.5)
        self.assertEqual(feature_row["pm25_pm10_ratio"], 0.5)

        # Check 1h change rate: (100 - 80) / 80 = 0.25
        self.assertEqual(feature_row["aqi_change_rate_1h"], 0.25)

        # Check 24h change rate: (100 - 50) / 50 = 1.0
        self.assertEqual(feature_row["aqi_change_rate_24h"], 1.0)

        # Check rolling mean 6h: (60 + 80 + 100) / 3 = 80.0
        self.assertEqual(feature_row["aqi_rolling_mean_6h"], 80.0)

        # Check rolling mean 24h: (50 + 60 + 80 + 100) / 4 = 72.5
        self.assertEqual(feature_row["aqi_rolling_mean_24h"], 72.5)

        # Target placeholder should be None
        self.assertIsNone(feature_row["target_aqi_3d"])

    def test_empty_history(self):
        current = {
            "fetched_at": "2026-07-31T12:00:00+00:00",
            "city": "London",
            "aqi": 75,
            "pm25": 15.0,
            "pm10": 30.0,
        }

        feature_row = build_feature_row(current, history=[])

        # Should not crash, derived rolling & change rates should be None
        self.assertIsNone(feature_row["aqi_change_rate_1h"])
        self.assertIsNone(feature_row["aqi_change_rate_24h"])
        self.assertIsNone(feature_row["aqi_rolling_mean_6h"])
        self.assertIsNone(feature_row["aqi_rolling_mean_24h"])

        # Ratio should still compute: 15 / 30 = 0.5
        self.assertEqual(feature_row["pm25_pm10_ratio"], 0.5)

        # Schema must contain all columns
        self.assertEqual(len(feature_row), len(EXPECTED_COLUMNS))

    def test_missing_pollutant_field(self):
        current = {
            "fetched_at": "2026-07-31T12:00:00+00:00",
            "city": "London",
            "aqi": 80,
            "pm25": 20.0,
            "pm10": None,  # Missing PM10
        }

        feature_row = build_feature_row(current, history=[])

        # pm25_pm10_ratio should be None when pm10 is None
        self.assertIsNone(feature_row["pm25_pm10_ratio"])
        self.assertEqual(feature_row["pm25"], 20.0)
        self.assertIsNone(feature_row["pm10"])


if __name__ == "__main__":
    unittest.main()
