import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.backfill import (
    calculate_aqi_from_pm25,
    fetch_historical_aqicn,
    fetch_openweather_pollution_history,
    fetch_openweather_weather_history,
    simulate_backfill_from_current,
    backfill_and_store,
    parse_args,
)


class TestBackfill(unittest.TestCase):

    def test_calculate_aqi_from_pm25(self):
        # Edge and standard values
        self.assertIsNone(calculate_aqi_from_pm25(None))
        self.assertIsNone(calculate_aqi_from_pm25(-5.0))
        self.assertEqual(calculate_aqi_from_pm25(0.0), 0)
        self.assertEqual(calculate_aqi_from_pm25(12.0), 50)
        self.assertEqual(calculate_aqi_from_pm25(35.4), 100)
        self.assertEqual(calculate_aqi_from_pm25(55.4), 150)
        self.assertEqual(calculate_aqi_from_pm25(150.4), 200)
        self.assertEqual(calculate_aqi_from_pm25(250.4), 300)
        self.assertEqual(calculate_aqi_from_pm25(600.0), 500)

    @patch("src.backfill.requests.get")
    def test_fetch_historical_aqicn_no_history_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "data": {
                "aqi": 120,
                # No 'history' / 'daily' / 'series'
            },
        }
        mock_get.return_value = mock_response

        res = fetch_historical_aqicn("Lahore", "dummy_token")
        self.assertIsNone(res)

    @patch("src.backfill.requests.get")
    def test_fetch_historical_aqicn_error_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "data": "Invalid key",
        }
        mock_get.return_value = mock_response

        res = fetch_historical_aqicn("Lahore", "dummy_token")
        self.assertIsNone(res)

    @patch("src.backfill.requests.get")
    def test_fetch_historical_aqicn_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "data": {
                "history": [
                    {
                        "time": "2026-08-01T00:00:00+00:00",
                        "aqi": 150,
                        "pm25": 60.0,
                    },
                    {
                        "time": "2026-08-01T01:00:00+00:00",
                        "aqi": 155,
                        "pm25": 62.0,
                    },
                ]
            },
        }
        mock_get.return_value = mock_response

        res = fetch_historical_aqicn("Lahore", "dummy_token")
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["aqi"], 150)
        self.assertEqual(res[1]["aqi"], 155)

    @patch("src.backfill.requests.get")
    def test_fetch_openweather_pollution_history_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "list": [
                {
                    "dt": 1700000000,
                    "components": {
                        "pm2_5": 35.4,
                        "pm10": 50.0,
                        "o3": 20.0,
                        "no2": 15.0,
                        "so2": 5.0,
                        "co": 300.0,
                    },
                },
                {
                    "dt": 1700003600,
                    "components": {
                        "pm2_5": 55.4,
                        "pm10": 80.0,
                        "o3": 22.0,
                        "no2": 18.0,
                        "so2": 6.0,
                        "co": 350.0,
                    },
                },
            ]
        }
        mock_get.return_value = mock_response

        res = fetch_openweather_pollution_history(
            lat=31.5204,
            lon=74.3587,
            api_key="dummy_key",
            start_ts=1700000000,
            end_ts=1700003600,
            city_name="Lahore",
        )
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["pm25"], 35.4)
        self.assertEqual(res[0]["aqi"], 100)
        self.assertEqual(res[1]["pm25"], 55.4)
        self.assertEqual(res[1]["aqi"], 150)

    @patch("src.backfill.fetch_openweather_weather_history")
    @patch("src.backfill.fetch_openweather_pollution_history")
    def test_simulate_backfill_from_current(self, mock_pollution, mock_weather):
        dt1 = "2026-08-01T00:00:00+00:00"
        dt2 = "2026-08-01T01:00:00+00:00"
        dt3 = "2026-08-01T02:00:00+00:00"

        mock_pollution.return_value = [
            {"fetched_at": dt1, "city": "Lahore", "aqi": 100, "pm25": 35.4, "pm10": 50.0},
            {"fetched_at": dt2, "city": "Lahore", "aqi": 110, "pm25": 40.0, "pm10": 60.0},
            {"fetched_at": dt3, "city": "Lahore", "aqi": 120, "pm25": 45.0, "pm10": 70.0},
        ]
        mock_weather.return_value = {
            dt1: {"temperature": 30.0, "humidity": 60, "wind_speed": 4.0, "pressure": 1010},
            dt2: {"temperature": 31.0, "humidity": 58, "wind_speed": 4.2, "pressure": 1010},
            dt3: {"temperature": 32.0, "humidity": 55, "wind_speed": 4.5, "pressure": 1009},
        }

        dummy_config = {
            "CITY_NAME": "Lahore",
            "CITY_LAT": 31.5204,
            "CITY_LON": 74.3587,
            "OPENWEATHER_API_KEY": "dummy_key",
        }

        rows = simulate_backfill_from_current(dummy_config, days_back=5)
        self.assertEqual(len(rows), 3)

        # Verify columns from EXPECTED_COLUMNS
        for row in rows:
            self.assertIn("fetched_at", row)
            self.assertIn("city", row)
            self.assertIn("aqi", row)
            self.assertIn("temperature", row)
            self.assertIn("hour", row)
            self.assertIn("season", row)

        # 3rd row should have 6h rolling mean computed because history >= 3 points
        self.assertIsNotNone(rows[2]["aqi_rolling_mean_6h"])
        self.assertEqual(rows[2]["aqi_rolling_mean_6h"], 110.0)

    @patch("src.backfill.feature_store")
    @patch("src.backfill.simulate_backfill_from_current")
    @patch("src.backfill.fetch_historical_aqicn")
    def test_backfill_and_store_dry_run(self, mock_aqicn, mock_simulate, mock_fs):
        mock_aqicn.return_value = None
        mock_simulate.return_value = [
            {"fetched_at": "2026-08-01T00:00:00+00:00", "city": "Lahore", "aqi": 100},
            {"fetched_at": "2026-08-01T01:00:00+00:00", "city": "Lahore", "aqi": 105},
        ]

        dummy_config = {"CITY_NAME": "Lahore", "AQICN_API_TOKEN": "token"}

        rows = backfill_and_store(dummy_config, days_back=10, dry_run=True)
        self.assertEqual(len(rows), 2)
        # Should not connect to feature store in dry run
        mock_fs.get_feature_store_connection.assert_not_called()
        mock_fs.write_feature_row.assert_not_called()

    @patch("src.backfill.feature_store")
    @patch("src.backfill.simulate_backfill_from_current")
    @patch("src.backfill.fetch_historical_aqicn")
    def test_backfill_and_store_writes_to_feature_store(self, mock_aqicn, mock_simulate, mock_fs):
        mock_aqicn.return_value = None
        mock_simulate.return_value = [
            {"fetched_at": "2026-08-01T00:00:00+00:00", "city": "Lahore", "aqi": 100},
            {"fetched_at": "2026-08-01T01:00:00+00:00", "city": "Lahore", "aqi": 105},
        ]

        mock_fs_conn = MagicMock()
        mock_fg = MagicMock()
        mock_fs.get_feature_store_connection.return_value = mock_fs_conn
        mock_fs.get_or_create_feature_group.return_value = mock_fg
        mock_fs.write_feature_row.return_value = True

        dummy_config = {"CITY_NAME": "Lahore", "AQICN_API_TOKEN": "token"}

        rows = backfill_and_store(dummy_config, days_back=10, dry_run=False)
        self.assertEqual(len(rows), 2)
        mock_fs.get_feature_store_connection.assert_called_once_with(dummy_config)
        mock_fs.get_or_create_feature_group.assert_called_once_with(mock_fs_conn)
        self.assertEqual(mock_fs.write_feature_row.call_count, 2)


if __name__ == "__main__":
    unittest.main()
