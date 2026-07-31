from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.feature_store import (
    get_feature_store_connection,
    get_or_create_feature_group,
    read_recent_history,
    write_feature_row,
)


class TestFeatureStore(unittest.TestCase):

    @patch("src.feature_store.hopsworks")
    def test_get_feature_store_connection_success(self, mock_hopsworks):
        mock_project = MagicMock()
        mock_fs = MagicMock()
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        config = {
            "HOPSWORKS_API_KEY": "valid_key",
            "HOPSWORKS_PROJECT_NAME": "valid_project",
        }

        fs = get_feature_store_connection(config)
        self.assertEqual(fs, mock_fs)
        mock_hopsworks.login.assert_called_once_with(
            api_key_value="valid_key",
            project="valid_project",
        )

    @patch("src.feature_store.hopsworks")
    def test_get_feature_store_connection_failure(self, mock_hopsworks):
        mock_hopsworks.login.side_effect = Exception("Invalid API Key")

        config = {
            "HOPSWORKS_API_KEY": "bad_key",
            "HOPSWORKS_PROJECT_NAME": "bad_project",
        }

        with self.assertRaises(RuntimeError) as cm:
            get_feature_store_connection(config)

        self.assertIn("Hopsworks authentication failed", str(cm.exception))

    def test_get_or_create_feature_group(self):
        mock_fs = MagicMock()
        mock_fg = MagicMock()
        mock_fs.get_or_create_feature_group.return_value = mock_fg

        fg = get_or_create_feature_group(mock_fs, name="aqi_features", version=1)

        self.assertEqual(fg, mock_fg)
        mock_fs.get_or_create_feature_group.assert_called_once_with(
            name="aqi_features",
            version=1,
            primary_key=["city", "fetched_at"],
            description="AQI and weather features for forecasting",
            online_enabled=True,
        )

    def test_write_feature_row_success(self):
        mock_fg = MagicMock()
        mock_fg.name = "aqi_features"

        feature_row = {
            "city": "London",
            "fetched_at": "2026-07-31T12:00:00+00:00",
            "aqi": 50,
        }

        success = write_feature_row(mock_fg, feature_row)
        self.assertTrue(success)
        self.assertEqual(mock_fg.insert.call_count, 1)

    def test_read_recent_history(self):
        mock_fg = MagicMock()
        now = datetime.now(timezone.utc)

        sample_df = pd.DataFrame([
            {
                "city": "London",
                "fetched_at": (now).isoformat(),
                "aqi": 80,
            },
            {
                "city": "Paris",
                "fetched_at": (now).isoformat(),
                "aqi": 90,
            },
        ])
        mock_fg.read.return_value = sample_df

        history = read_recent_history(mock_fg, city="London", hours_back=48)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["city"], "London")
        self.assertEqual(history[0]["aqi"], 80)


if __name__ == "__main__":
    unittest.main()
