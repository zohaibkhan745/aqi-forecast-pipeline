"""
FEATURE LEAKAGE ANALYSIS / NOTE:
=================================
1. `target_aqi_3d`:
   - LEAKY TARGET VARIABLE. This field represents the future ground-truth Air Quality Index
     (3 days into the future). It is populated retrospectively for offline training labels
     and MUST NOT be included as an input feature during model training or real-time inference.

2. Future Observations & Centered Rolling Windows:
   - Any rolling statistics (e.g., `aqi_rolling_mean_6h`, `aqi_rolling_mean_24h`) or change rates
     (`aqi_change_rate_1h`, `aqi_change_rate_24h`) MUST be strictly backward-looking using
     past observations at or before `fetched_at`. Forward-looking or centered windows would
     cause severe data leakage.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Complete list of expected columns for consistent schema in feature store
EXPECTED_COLUMNS = [
    "fetched_at",
    "city",
    "aqi",
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "season",
    "aqi_change_rate_1h",
    "aqi_change_rate_24h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_24h",
    "pm25_pm10_ratio",
    "target_aqi_3d",
]


def _parse_timestamp(timestamp: Union[str, datetime]) -> datetime:
    """Helper to parse datetime objects or ISO strings into timezone-aware datetime."""
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp

    if isinstance(timestamp, str):
        clean_ts = timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    raise ValueError(f"Unsupported timestamp type/format: {timestamp}")


def _get_season(month: int) -> str:
    """Helper to determine season from month number (Northern Hemisphere)."""
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def compute_time_features(timestamp: Optional[Union[str, datetime]]) -> Dict[str, Any]:
    """Extract temporal features from a timestamp string or datetime object.

    Args:
        timestamp (str | datetime, optional): ISO format timestamp string or datetime object.

    Returns:
        dict: Temporal features containing hour, day_of_week, day_of_month, month,
              is_weekend (0/1), and season.
    """
    if not timestamp:
        return {
            "hour": None,
            "day_of_week": None,
            "day_of_month": None,
            "month": None,
            "is_weekend": None,
            "season": None,
        }

    try:
        dt = _parse_timestamp(timestamp)
        day_of_week = dt.weekday()  # Monday=0, Sunday=6
        is_weekend = 1 if day_of_week in (5, 6) else 0

        return {
            "hour": dt.hour,
            "day_of_week": day_of_week,
            "day_of_month": dt.day,
            "month": dt.month,
            "is_weekend": is_weekend,
            "season": _get_season(dt.month),
        }
    except Exception as err:
        logger.warning("Failed to compute time features for timestamp '%s': %s", timestamp, err)
        return {
            "hour": None,
            "day_of_week": None,
            "day_of_month": None,
            "month": None,
            "is_weekend": None,
            "season": None,
        }


def compute_derived_features(
    current: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Compute derived features (change rates, rolling means, pollutant ratio).

    Args:
        current (dict): Current raw snapshot dictionary.
        history (list[dict], optional): List of past snapshots ordered oldest to newest.

    Returns:
        dict: Derived features dictionary (aqi_change_rate_1h, aqi_change_rate_24h,
              aqi_rolling_mean_6h, aqi_rolling_mean_24h, pm25_pm10_ratio).
    """
    history = history or []

    derived: Dict[str, Any] = {
        "aqi_change_rate_1h": None,
        "aqi_change_rate_24h": None,
        "aqi_rolling_mean_6h": None,
        "aqi_rolling_mean_24h": None,
        "pm25_pm10_ratio": None,
    }

    # 1. PM2.5 to PM10 ratio
    pm25 = current.get("pm25")
    pm10 = current.get("pm10")
    if pm25 is not None and pm10 is not None:
        try:
            pm25_val = float(pm25)
            pm10_val = float(pm10)
            if pm10_val > 0:
                derived["pm25_pm10_ratio"] = round(pm25_val / pm10_val, 4)
        except (ValueError, TypeError):
            pass

    current_aqi = current.get("aqi")
    current_ts = current.get("fetched_at")

    if current_aqi is None or current_ts is None:
        return derived

    try:
        current_dt = _parse_timestamp(current_ts)
        current_aqi_val = float(current_aqi)
    except (ValueError, TypeError):
        return derived

    # Process history snapshots with valid timestamps and AQI values
    valid_history = []
    for snapshot in history:
        h_aqi = snapshot.get("aqi")
        h_ts = snapshot.get("fetched_at")
        if h_aqi is not None and h_ts is not None:
            try:
                h_dt = _parse_timestamp(h_ts)
                h_aqi_val = float(h_aqi)
                valid_history.append((h_dt, h_aqi_val))
            except (ValueError, TypeError):
                continue

    # 2. Rolling Means (require minimum 3 data points including current)
    # 6-hour rolling window
    points_6h = [
        val
        for dt_h, val in valid_history
        if 0 <= (current_dt - dt_h).total_seconds() <= 6 * 3600
    ] + [current_aqi_val]

    if len(points_6h) >= 3:
        derived["aqi_rolling_mean_6h"] = round(sum(points_6h) / len(points_6h), 2)

    # 24-hour rolling window
    points_24h = [
        val
        for dt_h, val in valid_history
        if 0 <= (current_dt - dt_h).total_seconds() <= 24 * 3600
    ] + [current_aqi_val]

    if len(points_24h) >= 3:
        derived["aqi_rolling_mean_24h"] = round(sum(points_24h) / len(points_24h), 2)

    # 3. Change Rates (percent change vs closest past snapshot to horizon)
    def _find_closest_past_aqi(target_hours: float, tolerance_hours: float) -> Optional[float]:
        candidates = []
        for dt_h, val in valid_history:
            hours_diff = (current_dt - dt_h).total_seconds() / 3600.0
            if abs(hours_diff - target_hours) <= tolerance_hours:
                candidates.append((abs(hours_diff - target_hours), val))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    # 1h change rate (tolerance: 0.5 hours)
    past_1h = _find_closest_past_aqi(target_hours=1.0, tolerance_hours=0.5)
    if past_1h is not None and past_1h > 0:
        derived["aqi_change_rate_1h"] = round((current_aqi_val - past_1h) / past_1h, 4)

    # 24h change rate (tolerance: 2.0 hours)
    past_24h = _find_closest_past_aqi(target_hours=24.0, tolerance_hours=2.0)
    if past_24h is not None and past_24h > 0:
        derived["aqi_change_rate_24h"] = round((current_aqi_val - past_24h) / past_24h, 4)

    return derived


def build_feature_row(
    raw_snapshot: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Combine raw snapshot, time features, and derived features into a standardized feature row.

    Ensures every column in EXPECTED_COLUMNS is present in the output dictionary.

    Args:
        raw_snapshot (dict): Output from data_fetch.get_raw_snapshot().
        history (list[dict], optional): List of past snapshots ordered oldest to newest.

    Returns:
        dict: A flat dictionary representing a complete model-ready feature row.
    """
    raw_snapshot = raw_snapshot or {}
    history = history or []

    time_feats = compute_time_features(raw_snapshot.get("fetched_at"))
    derived_feats = compute_derived_features(raw_snapshot, history)

    combined = {}
    combined.update(raw_snapshot)
    combined.update(time_feats)
    combined.update(derived_feats)
    combined["target_aqi_3d"] = raw_snapshot.get("target_aqi_3d", None)

    # Standardize column structure matching EXPECTED_COLUMNS
    feature_row = {}
    for col in EXPECTED_COLUMNS:
        feature_row[col] = combined.get(col, None)

    return feature_row
