"""
========================================================================================
HISTORICAL AQI DATA SOURCE & FALLBACK STRATEGY NOTICE (LAHORE):
========================================================================================
1. Primary Historical AQI Source:
   - AQICN (waqi.info) Free Tier: AQICN does not provide a public historical endpoint
     for free tier API tokens. Any attempt to query historical data via waqi.info will
     return empty or restricted status.

2. Fallback Historical Pollution API (Active Default):
   - OpenWeather Air Pollution History API:
     Endpoint: `http://api.openweathermap.org/data/2.5/air_pollution/history`
     Details: Available on OpenWeather free tier with up to 1 year of hourly historical
     air pollution data (CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3) for any geographic
     coordinates, including Lahore (lat: 31.5204, lon: 74.3587).
     PM2.5 / PM10 readings are converted to standardized US EPA AQI scale.

3. Historical Weather Data:
   - OpenWeather Historical Weather endpoint (`/data/2.5/history/city` or One Call 3.0).
   - If historical weather access requires paid tier or is unavailable, weather metrics
     will default to `None` without failing the pipeline or corrupting pollutant records.

4. Secondary Manual Fallback Strategy:
   - If free API limits are exceeded or remote history is unavailable, users can bootstrap
     historical training data using public historical datasets:
     a. Kaggle: "Lahore Air Quality Data" / "Air Quality Data in Pakistan"
     b. Pakistan EPA / US Embassy Air Quality Monitor public datasets (AirNow Gov).
   - IMPORTANT: We strictly DO NOT interpolate or fabricate fake AQI values to bridge gaps.
     Missing dates are skipped and explicitly logged.
========================================================================================
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

# Ensure parent directory is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import feature_engineering, feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_config_val(config: Any, key: str, default: Any = None) -> Any:
    """Helper to safely extract configuration value from module or dictionary."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def calculate_aqi_from_pm25(pm25: Optional[float]) -> Optional[int]:
    """Calculate US EPA Air Quality Index (AQI) from PM2.5 concentration (ug/m3).

    Uses official EPA standard breakpoints for PM2.5 (0-500 scale).

    Args:
        pm25 (float, optional): PM2.5 concentration in ug/m3.

    Returns:
        int | None: Calculated AQI value (0-500), or None if input is invalid.
    """
    if pm25 is None or pm25 < 0:
        return None

    # EPA PM2.5 breakpoints: (C_low, C_high, I_low, I_high)
    breakpoints: List[Tuple[float, float, int, int]] = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    c = round(float(pm25), 1)

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return int(round(aqi))

    if c > 500.4:
        return 500

    return None


def fetch_historical_aqicn(city_or_geo: str, token: str) -> Optional[List[Dict[str, Any]]]:
    """Attempt to query AQICN's historical data endpoint for the given city or geo coordinate.

    AQICN's free tier generally restricts historical API queries. If the endpoint is
    unavailable, unauthorized, or returns empty data, logs a warning and returns None.

    Args:
        city_or_geo (str): City name or geo identifier (e.g., "Lahore" or "geo:31.5204;74.3587").
        token (str): AQICN API token.

    Returns:
        list[dict] | None: List of historical snapshot dictionaries, or None if unavailable.
    """
    url = f"https://api.waqi.info/feed/{city_or_geo}/?token={token}"
    logger.info("Attempting to query AQICN historical feed for '%s'...", city_or_geo)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        res_json = response.json()

        if res_json.get("status") != "ok":
            logger.warning(
                "AQICN API returned non-ok status '%s' for historical query on '%s': %s",
                res_json.get("status"),
                city_or_geo,
                res_json.get("data"),
            )
            return None

        data = res_json.get("data", {})
        # Check if historical time series is provided
        history_records = data.get("history") or data.get("daily") or data.get("series")
        if not history_records:
            logger.warning(
                "AQICN endpoint responded with current/forecast feed only. "
                "No historical time-series data available for '%s' on free tier.",
                city_or_geo,
            )
            return None

        snapshots: List[Dict[str, Any]] = []
        for item in history_records:
            ts = item.get("time") or item.get("timestamp")
            if not ts:
                continue
            snapshot = {
                "fetched_at": ts,
                "city": city_or_geo,
                "aqi": item.get("aqi"),
                "pm25": item.get("pm25"),
                "pm10": item.get("pm10"),
                "o3": item.get("o3"),
                "no2": item.get("no2"),
                "so2": item.get("so2"),
                "co": item.get("co"),
                "temperature": item.get("temperature"),
                "humidity": item.get("humidity"),
                "wind_speed": item.get("wind_speed"),
                "pressure": item.get("pressure"),
            }
            snapshots.append(snapshot)

        return snapshots if snapshots else None

    except Exception as err:
        logger.warning(
            "AQICN historical query failed for '%s' (%s). Returning None.",
            city_or_geo,
            err,
        )
        return None


def fetch_openweather_pollution_history(
    lat: float, lon: float, api_key: str, start_ts: int, end_ts: int, city_name: str
) -> List[Dict[str, Any]]:
    """Fetch historical air pollution data from OpenWeather Air Pollution History API.

    Endpoint: /data/2.5/air_pollution/history

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): OpenWeather API key.
        start_ts (int): Start UNIX timestamp (seconds).
        end_ts (int): End UNIX timestamp (seconds).
        city_name (str): City name for labeling.

    Returns:
        list[dict]: Hourly raw snapshots containing pollutant readings and derived EPA AQI.
    """
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_ts,
        "end": end_ts,
        "appid": api_key,
    }

    logger.info(
        "Fetching OpenWeather air pollution history for (lat: %s, lon: %s) from %s to %s...",
        lat,
        lon,
        datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
    )

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        res_json = response.json()

        items = res_json.get("list", [])
        if not items:
            logger.warning("OpenWeather air pollution history returned empty list.")
            return []

        snapshots: List[Dict[str, Any]] = []
        for item in items:
            dt_sec = item.get("dt")
            if dt_sec is None:
                continue

            iso_ts = datetime.fromtimestamp(dt_sec, tz=timezone.utc).isoformat()
            components = item.get("components", {})
            pm25_val = components.get("pm2_5")
            pm10_val = components.get("pm10")

            # Derive EPA AQI from PM2.5 concentration, or fallback to PM10 if needed
            calculated_aqi = calculate_aqi_from_pm25(pm25_val)

            snapshots.append({
                "fetched_at": iso_ts,
                "city": city_name,
                "aqi": calculated_aqi,
                "pm25": pm25_val,
                "pm10": pm10_val,
                "o3": components.get("o3"),
                "no2": components.get("no2"),
                "so2": components.get("so2"),
                "co": components.get("co"),
                "temperature": None,
                "humidity": None,
                "wind_speed": None,
                "pressure": None,
            })

        logger.info("Successfully fetched %d historical pollution records from OpenWeather.", len(snapshots))
        return snapshots

    except Exception as err:
        logger.warning("Failed to fetch OpenWeather historical air pollution: %s", err)
        return []


def fetch_openweather_weather_history(
    lat: float, lon: float, api_key: str, start_ts: int, end_ts: int
) -> Dict[str, Dict[str, Optional[float]]]:
    """Attempt to fetch historical weather data from OpenWeather.

    If unavailable on free tier (e.g. 401/403), logs warning and returns empty map.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        api_key (str): OpenWeather API key.
        start_ts (int): Start UNIX timestamp (seconds).
        end_ts (int): End UNIX timestamp (seconds).

    Returns:
        dict[str, dict]: Mapping of ISO timestamp (or hourly key) to weather dictionary.
    """
    url = "https://api.openweathermap.org/data/2.5/history/city"
    params = {
        "lat": lat,
        "lon": lon,
        "type": "hour",
        "start": start_ts,
        "end": end_ts,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code in (401, 403):
            logger.warning(
                "OpenWeather Historical Weather API returned %d (Subscription required). "
                "Historical weather metrics will be omitted.",
                response.status_code,
            )
            return {}

        response.raise_for_status()
        res_json = response.json()
        items = res_json.get("list", [])

        weather_by_ts: Dict[str, Dict[str, Optional[float]]] = {}
        for item in items:
            dt_sec = item.get("dt")
            if dt_sec is None:
                continue
            iso_ts = datetime.fromtimestamp(dt_sec, tz=timezone.utc).isoformat()
            main = item.get("main", {})
            wind = item.get("wind", {})
            weather_by_ts[iso_ts] = {
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "wind_speed": wind.get("speed"),
                "pressure": main.get("pressure"),
            }

        return weather_by_ts

    except Exception as err:
        logger.warning("Historical weather query skipped or unavailable: %s", err)
        return {}


def simulate_backfill_from_current(config: Any, days_back: int = 30) -> List[Dict[str, Any]]:
    """Generate historical feature rows using free tier APIs without fabricating fake data.

    1. Queries OpenWeather Historical Air Pollution API for past `days_back` days.
    2. Queries OpenWeather Historical Weather if available, matching records on timestamps.
    3. Identifies missing dates, skips them without interpolation, and logs skipped dates.
    4. Builds feature rows in chronological order, feeding growing history to compute rolling features.

    Args:
        config (Any): Application config (module or dict).
        days_back (int, optional): Number of past days to backfill. Defaults to 30.

    Returns:
        list[dict]: Chronologically ordered list of model-ready feature rows.
    """
    city_name = _get_config_val(config, "CITY_NAME", "Lahore")
    lat = float(_get_config_val(config, "CITY_LAT", 31.5204))
    lon = float(_get_config_val(config, "CITY_LON", 74.3587))
    ow_api_key = _get_config_val(config, "OPENWEATHER_API_KEY", "")

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days_back)
    start_ts = int(start_time.timestamp())
    end_ts = int(now.timestamp())

    # Step 1: Fetch historical air pollution records
    pollution_snapshots = fetch_openweather_pollution_history(
        lat=lat,
        lon=lon,
        api_key=ow_api_key,
        start_ts=start_ts,
        end_ts=end_ts,
        city_name=city_name,
    )

    if not pollution_snapshots:
        logger.warning(
            "No historical pollution data could be retrieved for city '%s' (%d days back).",
            city_name,
            days_back,
        )
        return []

    # Step 2: Attempt to fetch historical weather
    weather_map = fetch_openweather_weather_history(
        lat=lat,
        lon=lon,
        api_key=ow_api_key,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    # Step 3: Merge weather into pollution snapshots and sort chronologically
    merged_snapshots: List[Dict[str, Any]] = []
    seen_dates = set()

    for p_snap in pollution_snapshots:
        ts_str = p_snap["fetched_at"]
        dt_val = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        seen_dates.add(dt_val.date().isoformat())

        weather_info = weather_map.get(ts_str)
        if weather_info:
            p_snap["temperature"] = weather_info.get("temperature")
            p_snap["humidity"] = weather_info.get("humidity")
            p_snap["wind_speed"] = weather_info.get("wind_speed")
            p_snap["pressure"] = weather_info.get("pressure")

        # Skip rows with no AQI and no pollutant data (do NOT fabricate fake values)
        if p_snap.get("aqi") is None and p_snap.get("pm25") is None:
            continue

        merged_snapshots.append(p_snap)

    # Log any completely missing dates in the expected window
    expected_dates = {
        (start_time + timedelta(days=i)).date().isoformat()
        for i in range(days_back + 1)
    }
    missing_dates = sorted(list(expected_dates - seen_dates))
    if missing_dates:
        logger.info(
            "Historical data missing for %d date(s) in the past %d days: %s. "
            "These dates will be skipped without fabrication or interpolation.",
            len(missing_dates),
            days_back,
            ", ".join(missing_dates),
        )

    # Sort merged snapshots chronologically (oldest to newest)
    merged_snapshots.sort(
        key=lambda x: datetime.fromisoformat(x["fetched_at"].replace("Z", "+00:00"))
    )

    # Step 4: Build feature rows chronologically with growing history
    feature_rows: List[Dict[str, Any]] = []
    accumulated_history: List[Dict[str, Any]] = []

    for snapshot in merged_snapshots:
        feature_row = feature_engineering.build_feature_row(snapshot, history=accumulated_history)
        feature_rows.append(feature_row)
        accumulated_history.append(snapshot)

    logger.info("Generated %d feature rows from historical backfill data.", len(feature_rows))
    return feature_rows


def backfill_and_store(
    config: Any, days_back: int = 30, dry_run: bool = False
) -> List[Dict[str, Any]]:
    """Orchestrate backfilling historical data and storing to the Hopsworks Feature Store.

    1. Tries `fetch_historical_aqicn` first.
    2. If unavailable, falls back to `simulate_backfill_from_current`.
    3. Writes each feature row to Hopsworks Feature Store in chronological order,
       logging progress every 50 rows.
    4. If dry_run is True, prints total row count and a sample of 3 rows instead of writing.

    Args:
        config (Any): Application config (module or dict).
        days_back (int, optional): Days back to backfill. Defaults to 30.
        dry_run (bool, optional): If True, previews data without writing. Defaults to False.

    Returns:
        list[dict]: List of generated feature rows.
    """
    city_name = _get_config_val(config, "CITY_NAME", "Lahore")
    aqicn_token = _get_config_val(config, "AQICN_API_TOKEN", "")

    logger.info(
        "Starting backfill process for '%s' (%d days back, dry_run=%s)...",
        city_name,
        days_back,
        dry_run,
    )

    feature_rows: List[Dict[str, Any]] = []

    # Strategy 1: Attempt AQICN historical feed
    aqicn_history = fetch_historical_aqicn(city_name, aqicn_token)
    if aqicn_history:
        logger.info("Using AQICN historical dataset (%d records).", len(aqicn_history))
        aqicn_history.sort(
            key=lambda x: datetime.fromisoformat(x["fetched_at"].replace("Z", "+00:00"))
        )
        accumulated_history: List[Dict[str, Any]] = []
        for snap in aqicn_history:
            row = feature_engineering.build_feature_row(snap, history=accumulated_history)
            feature_rows.append(row)
            accumulated_history.append(snap)
    else:
        # Strategy 2: Fall back to OpenWeather historical pollution backfill
        logger.info("Falling back to OpenWeather historical air pollution backfill...")
        feature_rows = simulate_backfill_from_current(config, days_back=days_back)

    if not feature_rows:
        logger.warning("No backfill rows were generated. Exiting backfill process.")
        return []

    total_rows = len(feature_rows)
    logger.info("Total backfill feature rows ready: %d", total_rows)

    if dry_run:
        print("\n==================================================")
        print(f" DRY RUN MODE - BACKFILL PREVIEW FOR {city_name.upper()}")
        print(f" Total Rows Generated: {total_rows}")
        print("==================================================")
        sample_count = min(3, total_rows)
        print(f"\n--- Sample of {sample_count} rows: ---")
        for i in range(sample_count):
            print(f"\n[Row {i + 1}/{sample_count}]")
            print(json.dumps(feature_rows[i], indent=2))
        print("\n==================================================")
        logger.info("Dry-run complete. No rows written to feature store.")
        return feature_rows

    # Persistence to Hopsworks Feature Store
    logger.info("Connecting to Hopsworks Feature Store for persistence...")
    fs = feature_store.get_feature_store_connection(config)
    fg = feature_store.get_or_create_feature_group(fs)

    written_count = 0
    for idx, row in enumerate(feature_rows, start=1):
        success = feature_store.write_feature_row(fg, row)
        if success:
            written_count += 1

        if idx % 50 == 0 or idx == total_rows:
            logger.info("Progress: %d/%d rows written to Feature Store.", idx, total_rows)

    logger.info(
        "Backfill completed successfully. Written %d/%d rows to Feature Store.",
        written_count,
        total_rows,
    )
    return feature_rows


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for backfill CLI."""
    parser = argparse.ArgumentParser(
        description="Backfill Feature Store with historical AQI and weather data."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview backfilled feature rows without writing to Hopsworks Feature Store",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for backfill script."""
    args = parse_args()

    try:
        import src.config as config
    except Exception as err:
        logger.error("Failed to load project config: %s", err)
        sys.exit(1)

    try:
        backfill_and_store(config=config, days_back=args.days, dry_run=args.dry_run)
    except Exception as err:
        logger.error("Backfill failed with error: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
