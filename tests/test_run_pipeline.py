import unittest
from unittest.mock import MagicMock, patch

from src.run_pipeline import run_pipeline


class TestRunPipeline(unittest.TestCase):

    @patch("src.run_pipeline.data_fetch.get_raw_snapshot")
    @patch("src.run_pipeline.feature_store.get_feature_store_connection")
    @patch("src.run_pipeline.feature_store.get_or_create_feature_group")
    @patch("src.run_pipeline.feature_store.read_recent_history")
    @patch("src.run_pipeline.feature_store.write_feature_row")
    def test_run_pipeline_success(
        self,
        mock_write_row,
        mock_read_history,
        mock_get_fg,
        mock_get_fs,
        mock_get_snapshot,
    ):
        mock_get_snapshot.return_value = {
            "fetched_at": "2026-07-31T12:00:00+00:00",
            "city": "London",
            "aqi": 50,
            "pm25": 12.0,
            "pm10": 24.0,
        }
        mock_read_history.return_value = []
        mock_write_row.return_value = True

        feature_row = run_pipeline(dry_run=False)

        self.assertEqual(feature_row["city"], "London")
        self.assertEqual(feature_row["aqi"], 50)
        self.assertEqual(feature_row["pm25_pm10_ratio"], 0.5)
        self.assertTrue(mock_write_row.called)

    @patch("src.run_pipeline.data_fetch.get_raw_snapshot")
    def test_run_pipeline_dry_run(self, mock_get_snapshot):
        mock_get_snapshot.return_value = {
            "fetched_at": "2026-07-31T12:00:00+00:00",
            "city": "Paris",
            "aqi": 60,
            "pm25": 15.0,
            "pm10": 30.0,
        }

        feature_row = run_pipeline(city_override="Paris", dry_run=True)

        self.assertEqual(feature_row["city"], "Paris")
        self.assertEqual(feature_row["aqi"], 60)


if __name__ == "__main__":
    unittest.main()
