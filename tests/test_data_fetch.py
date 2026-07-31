import unittest
from unittest.mock import MagicMock, patch

from src.data_fetch import (
    fetch_aqicn_data,
    fetch_openweather_data,
    get_raw_snapshot,
)


class TestDataFetch(unittest.TestCase):

    @patch("src.data_fetch.requests.get")
    def test_fetch_aqicn_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "data": {
                "aqi": 45,
                "iaqi": {
                    "pm25": {"v": 12.0},
                    "pm10": {"v": 25.0},
                },
            },
        }
        mock_get.return_value = mock_response

        result = fetch_aqicn_data("London", "dummy_token")
        self.assertEqual(result["aqi"], 45)
        self.assertEqual(result["iaqi"]["pm25"]["v"], 12.0)
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.data_fetch.requests.get")
    def test_fetch_aqicn_data_retry_and_fail(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "data": "Unknown station",
        }
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as cm:
            fetch_aqicn_data("UnknownCity", "dummy_token")

        self.assertIn("AQICN API returned status 'error'", str(cm.exception))
        self.assertEqual(mock_get.call_count, 3)

    @patch("src.data_fetch.requests.get")
    def test_fetch_openweather_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {
                "temp": 19.5,
                "humidity": 65,
                "pressure": 1013,
            },
            "wind": {
                "speed": 3.6,
            },
        }
        mock_get.return_value = mock_response

        result = fetch_openweather_data(51.5074, -0.1278, "dummy_key")
        self.assertEqual(result["temperature"], 19.5)
        self.assertEqual(result["humidity"], 65)
        self.assertEqual(result["wind_speed"], 3.6)
        self.assertEqual(result["pressure"], 1013)
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.data_fetch.fetch_openweather_data")
    @patch("src.data_fetch.fetch_aqicn_data")
    def test_get_raw_snapshot_fallback(self, mock_aqicn, mock_openweather):
        mock_aqicn.return_value = {
            "aqi": 50,
            "iaqi": {
                "pm25": {"v": 15.0},
            },
        }
        mock_openweather.side_effect = Exception("OpenWeather service down")

        dummy_config = {
            "CITY_NAME": "London",
            "AQICN_API_TOKEN": "token",
            "CITY_LAT": 51.5074,
            "CITY_LON": -0.1278,
            "OPENWEATHER_API_KEY": "key",
        }

        snapshot = get_raw_snapshot(dummy_config)

        self.assertEqual(snapshot["city"], "London")
        self.assertEqual(snapshot["aqi"], 50)
        self.assertEqual(snapshot["pm25"], 15.0)
        self.assertIsNone(snapshot["temperature"])
        self.assertIsNone(snapshot["humidity"])
        self.assertIn("fetched_at", snapshot)


if __name__ == "__main__":
    unittest.main()
