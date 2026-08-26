import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path

from src.data_quality import (
    load_all_features,
    generate_gap_report,
    generate_summary_stats,
    ensure_processed_data_snapshot,
)


class TestDataQuality(unittest.TestCase):

    def test_load_all_features_sorting(self):
        mock_fs = MagicMock()
        mock_fg = MagicMock()
        mock_fs.get_or_create_feature_group.return_value = mock_fg

        df_input = pd.DataFrame([
            {"fetched_at": "2026-08-01T05:00:00+00:00", "city": "Lahore", "aqi": 120},
            {"fetched_at": "2026-08-01T01:00:00+00:00", "city": "Lahore", "aqi": 100},
            {"fetched_at": "2026-08-01T03:00:00+00:00", "city": "Lahore", "aqi": 110},
        ])
        mock_fg.read.return_value = df_input

        res_df = load_all_features(mock_fs, "aqi_features")
        self.assertEqual(len(res_df), 3)
        self.assertEqual(res_df.iloc[0]["fetched_at"], "2026-08-01T01:00:00+00:00")
        self.assertEqual(res_df.iloc[1]["fetched_at"], "2026-08-01T03:00:00+00:00")
        self.assertEqual(res_df.iloc[2]["fetched_at"], "2026-08-01T05:00:00+00:00")

    def test_generate_gap_report_with_gaps_and_nulls(self):
        df = pd.DataFrame([
            {"fetched_at": "2026-08-01T00:00:00+00:00", "city": "Lahore", "aqi": 100, "temp": 30.0, "missing_col": None},
            {"fetched_at": "2026-08-01T01:00:00+00:00", "city": "Lahore", "aqi": 105, "temp": 30.5, "missing_col": None},
            # Gap of 4 hours between 01:00 and 05:00
            {"fetched_at": "2026-08-01T05:00:00+00:00", "city": "Lahore", "aqi": 115, "temp": 32.0, "missing_col": None},
            {"fetched_at": "2026-08-01T06:00:00+00:00", "city": "Lahore", "aqi": 120, "temp": None, "missing_col": None},
        ])

        report = generate_gap_report(df)

        self.assertEqual(report["total_rows"], 4)
        self.assertEqual(report["date_range"]["start"], "2026-08-01T00:00:00+00:00")
        self.assertEqual(report["date_range"]["end"], "2026-08-01T06:00:00+00:00")
        self.assertEqual(report["expected_hourly_slots"], 7)  # 00, 01, 02, 03, 04, 05, 06 -> 7 slots
        self.assertAlmostEqual(report["completeness_pct"], (4 / 7) * 100.0, places=2)

        # Gap verification
        self.assertEqual(len(report["gaps"]), 1)
        gap = report["gaps"][0]
        self.assertEqual(gap["gap_start"], "2026-08-01T01:00:00+00:00")
        self.assertEqual(gap["gap_end"], "2026-08-01T05:00:00+00:00")
        self.assertEqual(gap["gap_hours"], 4.0)

        # Null analysis & attention flagging
        self.assertIn("missing_col", report["needs_attention"])
        self.assertEqual(report["null_percentages"]["missing_col"], 100.0)
        self.assertEqual(report["null_percentages"]["temp"], 25.0)
        self.assertNotIn("temp", report["needs_attention"])

    def test_generate_gap_report_empty(self):
        report = generate_gap_report(pd.DataFrame())
        self.assertEqual(report["total_rows"], 0)
        self.assertEqual(report["completeness_pct"], 0.0)
        self.assertEqual(len(report["gaps"]), 0)

    def test_generate_summary_stats(self):
        df = pd.DataFrame({
            "aqi": [50, 100, 150, 200],
            "pm25": [12.0, 35.0, 55.0, 150.0],
            "temperature": [25.0, 28.0, 30.0, 32.0],
            "city": ["Lahore", "Lahore", "Lahore", "Lahore"],  # non-numeric
        })

        stats = generate_summary_stats(df)
        self.assertIn("aqi", stats.index)
        self.assertIn("pm25", stats.index)
        self.assertIn("temperature", stats.index)
        self.assertNotIn("city", stats.index)

        # Check calculated values for AQI
        self.assertEqual(stats.loc["aqi", "min"], 50.0)
        self.assertEqual(stats.loc["aqi", "max"], 200.0)
        self.assertEqual(stats.loc["aqi", "mean"], 125.0)
        self.assertEqual(stats.loc["aqi", "50%"], 125.0)

    def test_ensure_processed_data_snapshot(self):
        df = pd.DataFrame({"aqi": [100, 110], "city": ["Lahore", "Lahore"]})
        tmp_csv = "data/processed/test_snapshot.csv"

        saved_path = ensure_processed_data_snapshot(df, file_path=tmp_csv)
        self.assertTrue(saved_path.exists())

        loaded_df = pd.read_csv(saved_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(loaded_df.iloc[0]["aqi"], 100)

        # Cleanup test file
        if saved_path.exists():
            saved_path.unlink()


if __name__ == "__main__":
    unittest.main()
