"""Feature Pipeline Execution Script for AQI Predictor.

Ties together data fetching, feature engineering, and Hopsworks Feature Store persistence.
"""

import argparse
import json
import logging
import sys
import os

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List

# Ensure parent directory is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config, data_fetch, feature_engineering, feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AQI Predictor Feature Pipeline"
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Optional city name override (defaults to CITY_NAME in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline and print feature row without writing to Hopsworks Feature Store",
    )
    return parser.parse_args()


def run_pipeline(city_override: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """Execute end-to-end feature pipeline.

    Args:
        city_override (str, optional): City name override.
        dry_run (bool): If True, prints feature row without writing to Feature Store.

    Returns:
        dict: Generated feature row.
    """
    city_name = city_override or config.CITY_NAME

    runtime_config = config
    if city_override:
        runtime_config = {
            "CITY_NAME": city_override,
            "AQICN_API_TOKEN": config.AQICN_API_TOKEN,
            "CITY_LAT": config.CITY_LAT,
            "CITY_LON": config.CITY_LON,
            "OPENWEATHER_API_KEY": config.OPENWEATHER_API_KEY,
            "HOPSWORKS_API_KEY": config.HOPSWORKS_API_KEY,
            "HOPSWORKS_PROJECT_NAME": config.HOPSWORKS_PROJECT_NAME,
        }

    logger.info("Starting Feature Pipeline for city: '%s' (Dry-run: %s)", city_name, dry_run)

    history: List[Dict[str, Any]] = []
    fs = None
    fg = None

    # Step A & B: Feature Store Connection & Recent History Retrieval
    if not dry_run:
        logger.info("Connecting to Hopsworks Feature Store...")
        fs = feature_store.get_feature_store_connection(runtime_config)
        fg = feature_store.get_or_create_feature_group(fs)
        logger.info("Reading recent history (last 48h) for '%s'...", city_name)
        history = feature_store.read_recent_history(fg, city=city_name, hours_back=48)
    else:
        logger.info("Dry-run active. Attempting optional history retrieval if Feature Store is reachable...")
        try:
            fs = feature_store.get_feature_store_connection(runtime_config)
            fg = feature_store.get_or_create_feature_group(fs)
            history = feature_store.read_recent_history(fg, city=city_name, hours_back=48)
        except Exception as fs_err:
            logger.warning("Dry-run: Feature Store connection skipped or unavailable (%s). Using empty history.", fs_err)
            history = []

    # Step C: Fetch Raw Data Snapshot
    logger.info("Fetching raw data snapshot for '%s'...", city_name)
    raw_snapshot = data_fetch.get_raw_snapshot(runtime_config)

    # Step D: Compute Time & Derived Features
    logger.info("Computing temporal & derived features...")
    feature_row = feature_engineering.build_feature_row(raw_snapshot, history)

    # Step E: Write Feature Row to Store (or Print in Dry-Run)
    if dry_run:
        logger.info("Dry-run mode: Feature row generated successfully.")
        print("\n--- Generated Feature Row ---")
        print(json.dumps(feature_row, indent=2))
    else:
        logger.info("Writing feature row to Hopsworks Feature Group...")
        success = feature_store.write_feature_row(fg, feature_row)
        if not success:
            raise RuntimeError("Failed to write feature row to Hopsworks Feature Group.")

    # Step F: Log Pipeline Summary
    derived_status = "Computable" if feature_row.get("aqi_rolling_mean_6h") is not None else "None (Insufficient History)"
    logger.info("--- Pipeline Summary ---")
    logger.info("Timestamp: %s", feature_row.get("fetched_at"))
    logger.info("City: %s", feature_row.get("city"))
    logger.info("Current AQI: %s", feature_row.get("aqi"))
    logger.info("Derived Rolling Features Status: %s", derived_status)

    return feature_row


def _safe_print(text: str) -> None:
    """Print text safely, falling back gracefully if terminal encoding does not support emojis."""
    try:
        print(text)
    except UnicodeEncodeError:
        fallback_text = text.replace("✅", "[SUCCESS]").replace("❌", "[FAILED]")
        print(fallback_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))


def main() -> None:
    """Main CLI entrypoint with error handling."""
    logger.info("--- Environment Variables Health Check ---")
    for var in config.REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if val:
            logger.info("%s: ✓ loaded", var)
        else:
            logger.warning("%s: ✗ missing", var)
            
    args = parse_args()
    city_name = args.city or getattr(config, "CITY_NAME", "Unknown")
    run_time = datetime.now(timezone.utc).isoformat()

    try:
        run_pipeline(city_override=args.city, dry_run=args.dry_run)
        _safe_print(f"\n✅ Pipeline run complete for {city_name} at {run_time}")
    except Exception as err:
        logger.error("Pipeline execution failed: %s", err)
        _safe_print(f"\n❌ Pipeline run failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
