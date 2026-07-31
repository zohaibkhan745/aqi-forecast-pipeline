"""Data Fetching Module for AQI Predictor.

Provides functions to fetch raw air quality data from AQICN API
and meteorological data from OpenWeather API with exponential backoff retries.
"""

import logging
import time

from datetime import datetime, timezone

from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def fetch_aqicn_data(city: str, token: str) -> Dict[str, Any]:
    """Fetch raw air quality data for a given city using the AQICN API.

    Makes an HTTP GET request to the AQICN feed endpoint. If the request fails or
    returns a non-ok status, it retries up to 3 times with exponential backoff.

    Args:
        city (str): Name of the city to query (e.g., "London").
        token (str): AQICN API access token.

    Returns:
        dict: The parsed "data" payload from the AQICN JSON response.

    Raises:
        RuntimeError: If the API returns a status other than "ok".
        requests.RequestException: If HTTP request attempts fail after max retries.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    max_attempts = 3
    backoff_seconds = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Fetching AQICN data for city '%s' (Attempt %d/%d)", city, attempt, max_attempts)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            res_json = response.json()

            if res_json.get("status") != "ok":
                error_data = res_json.get("data", "Unknown AQICN API error")
                raise RuntimeError(f"AQICN API returned status '{res_json.get('status')}': {error_data}")

            logger.info("Successfully fetched AQICN data for city '%s'", city)
            return res_json.get("data", {})

        except Exception as err:
            logger.error("Attempt %d/%d failed for AQICN data (city: %s): %s", attempt, max_attempts, city, err)
            if attempt == max_attempts:
                raise
            time.sleep(backoff_seconds)
            backoff_seconds *= 2.0

    raise RuntimeError(f"Failed to fetch AQICN data for '{city}' after {max_attempts} attempts.")


def fetch_openweather_data(lat: float, lon: float, api_key: str) -> Dict[str, Optional[float]]:
    """Fetch current weather metrics for latitude/longitude using OpenWeather API.

    Makes an HTTP GET request to OpenWeather Current Weather endpoint.
    Retries up to 3 times with exponential backoff upon failure.

    Args:
        lat (float): Geographic latitude.
        lon (float): Geographic longitude.
        api_key (str): OpenWeather API key.

    Returns:
        dict: Dictionary containing temperature (°C), humidity (%),
              wind_speed (m/s), and pressure (hPa).

    Raises:
        requests.RequestException: If HTTP request attempts fail after max retries.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    max_attempts = 3
    backoff_seconds = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Fetching OpenWeather data for (lat: %s, lon: %s) (Attempt %d/%d)", lat, lon, attempt, max_attempts)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            res_json = response.json()

            main_data = res_json.get("main", {})
            wind_data = res_json.get("wind", {})

            result = {
                "temperature": main_data.get("temp"),
                "humidity": main_data.get("humidity"),
                "wind_speed": wind_data.get("speed"),
                "pressure": main_data.get("pressure"),
            }
            logger.info("Successfully fetched OpenWeather data for (lat: %s, lon: %s)", lat, lon)
            return result

        except Exception as err:
            logger.error("Attempt %d/%d failed for OpenWeather data (lat: %s, lon: %s): %s", attempt, max_attempts, lat, lon, err)
            if attempt == max_attempts:
                raise
            time.sleep(backoff_seconds)
            backoff_seconds *= 2.0

    raise RuntimeError(f"Failed to fetch OpenWeather data after {max_attempts} attempts.")


def get_raw_snapshot(config: Any) -> Dict[str, Any]:
    """Fetch and merge raw air quality and weather observations into a single snapshot.

    Reads configuration parameters from the given config module or dictionary.
    Includes a UTC timestamp field "fetched_at". If one API fails, a warning is logged
    and missing values are populated with None without failing the pipeline.

    Args:
        config (Any): Module or dict containing configuration constants:
                      CITY_NAME, AQICN_API_TOKEN, CITY_LAT, CITY_LON, OPENWEATHER_API_KEY.

    Returns:
        dict: Merged flat dictionary containing timestamp, location, AQI metrics, and weather metrics.
    """
    def _get_val(key: str) -> Any:
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)

    city = _get_val("CITY_NAME")
    aqicn_token = _get_val("AQICN_API_TOKEN")
    lat = _get_val("CITY_LAT")
    lon = _get_val("CITY_LON")
    ow_api_key = _get_val("OPENWEATHER_API_KEY")

    fetched_at = datetime.now(timezone.utc).isoformat()

    snapshot: Dict[str, Any] = {
        "fetched_at": fetched_at,
        "city": city,
        "aqi": None,
        "pm25": None,
        "pm10": None,
        "o3": None,
        "no2": None,
        "so2": None,
        "co": None,
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "pressure": None,
    }

    # Fetch AQICN Data
    try:
        aqicn_data = fetch_aqicn_data(city=city, token=aqicn_token)
        snapshot["aqi"] = aqicn_data.get("aqi")
        iaqi = aqicn_data.get("iaqi", {})
        for pollutant in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            pollutant_info = iaqi.get(pollutant)
            if isinstance(pollutant_info, dict):
                snapshot[pollutant] = pollutant_info.get("v")
    except Exception as err:
        logger.warning("Fallback triggered: Failed to retrieve AQICN data (%s). Filling AQI fields with None.", err)

    # Fetch OpenWeather Data
    try:
        ow_data = fetch_openweather_data(lat=float(lat), lon=float(lon), api_key=ow_api_key)
        snapshot.update(ow_data)
    except Exception as err:
        logger.warning("Fallback triggered: Failed to retrieve OpenWeather data (%s). Filling weather fields with None.", err)

    return snapshot


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # Configure root logger for script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Ensure parent project directory is in sys.path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import src.config as config
        logger.info("Executing get_raw_snapshot using loaded config...")
        snapshot = get_raw_snapshot(config)
        print("\n--- Raw Snapshot Output ---")
        print(json.dumps(snapshot, indent=2))
    except Exception as main_err:
        logger.error("Failed to execute data fetch snapshot: %s", main_err)
