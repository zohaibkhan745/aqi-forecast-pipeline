"""
HOPSWORKS FREE TIER NOTICE:
===========================
Hopsworks free tier (Serverless/Community Edition) enforces storage and compute resource quotas.
If feature group insertion or data reading fails with storage, memory, or quota error codes,
please check your Hopsworks project dashboard or consult the official documentation
(https://docs.hopsworks.ai/) to manage project storage or clean up obsolete feature groups.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import hopsworks
except ImportError:
    hopsworks = None

logger = logging.getLogger(__name__)


def get_feature_store_connection(config: Any) -> Any:
    """Log into Hopsworks and return the Feature Store object.

    Args:
        config (Any): Config module or dictionary containing HOPSWORKS_API_KEY
                      and HOPSWORKS_PROJECT_NAME.

    Returns:
        hopsworks.feature_store.FeatureStore: Connected Hopsworks feature store instance.

    Raises:
        RuntimeError: If authentication or connection to Hopsworks fails.
        ImportError: If the hopsworks package is not installed.
    """
    if hopsworks is None:
        raise ImportError("The 'hopsworks' package is required to connect to the feature store.")

    def _get_conf(key: str) -> Any:
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)

    api_key = _get_conf("HOPSWORKS_API_KEY")
    project_name = _get_conf("HOPSWORKS_PROJECT_NAME")

    if not api_key or not project_name:
        raise ValueError("HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set in config.")

    try:
        logger.info("Connecting to Hopsworks project '%s'...", project_name)
        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name,
        )
        fs = project.get_feature_store()
        logger.info("Successfully connected to Hopsworks Feature Store for project '%s'", project_name)
        return fs
    except Exception as err:
        error_msg = (
            f"Hopsworks authentication failed for project '{project_name}'. "
            "Please verify that your HOPSWORKS_API_KEY is active and valid, "
            f"and that HOPSWORKS_PROJECT_NAME exists in your account. Details: {err}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from err


def get_or_create_feature_group(
    fs: Any,
    name: str = "aqi_features",
    version: int = 1,
    primary_key: Optional[List[str]] = None,
) -> Any:
    """Retrieve an existing Hopsworks Feature Group or create it if it does not exist.

    Args:
        fs (Any): Hopsworks Feature Store instance.
        name (str, optional): Name of the feature group. Defaults to "aqi_features".
        version (int, optional): Version of the feature group. Defaults to 1.
        primary_key (list[str], optional): Primary keys. Defaults to ["city", "fetched_at"].

    Returns:
        Any: The Hopsworks Feature Group object.
    """
    if primary_key is None:
        primary_key = ["city", "fetched_at"]

    try:
        fg = fs.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=primary_key,
            description="AQI and weather features for forecasting",
            online_enabled=True,
        )
        logger.info("Retrieved or created Feature Group '%s' (v%d)", name, version)
        return fg
    except Exception as err:
        logger.error("Failed to get or create Feature Group '%s': %s", name, err)
        raise


def write_feature_row(fg: Any, feature_row: Dict[str, Any]) -> bool:
    """Convert a single feature dictionary to a DataFrame and insert into Hopsworks.

    Args:
        fg (Any): Hopsworks Feature Group object.
        feature_row (dict): Standardized feature dictionary.

    Returns:
        bool: True if writing succeeded, False otherwise.
    """
    try:
        df = pd.DataFrame([feature_row])
        logger.info("Writing feature row into Hopsworks Feature Group '%s'...", getattr(fg, "name", "fg"))
        fg.insert(df)
        logger.info(
            "Successfully inserted feature row for city '%s' at '%s'",
            feature_row.get("city"),
            feature_row.get("fetched_at"),
        )
        return True
    except Exception as err:
        logger.error("Failed to write feature row to Hopsworks Feature Group: %s", err)
        return False


def read_recent_history(fg: Any, city: str, hours_back: int = 48) -> List[Dict[str, Any]]:
    """Read recent historical feature rows for a city from the Feature Group.

    Args:
        fg (Any): Hopsworks Feature Group object.
        city (str): Name of the city to filter.
        hours_back (int, optional): Number of past hours of data to fetch. Defaults to 48.

    Returns:
        list[dict]: List of feature dictionaries sorted oldest to newest.
    """
    try:
        df = fg.read()
        if df is None or df.empty:
            logger.info("No feature data found in Feature Group.")
            return []

        # Filter by city if column present
        if "city" in df.columns:
            df = df[df["city"].astype(str).str.lower() == str(city).lower()]

        if df.empty or "fetched_at" not in df.columns:
            return []

        # Convert fetched_at to datetime for cutoff filtering and sorting
        df = df.copy()
        df["_fetched_at_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        df = df[df["_fetched_at_dt"] >= cutoff_time]
        df = df.sort_values(by="_fetched_at_dt", ascending=True)
        df = df.drop(columns=["_fetched_at_dt"])

        # Convert NaN values to None
        records = df.to_dict(orient="records")
        clean_records = []
        for rec in records:
            clean_rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
            clean_records.append(clean_rec)

        logger.info("Read %d history rows for city '%s' (past %d hours)", len(clean_records), city, hours_back)
        return clean_records

    except Exception as err:
        logger.error("Failed to read recent history for city '%s': %s", city, err)
        return []

def get_or_create_prediction_log_group(fs: Any) -> Any:
    """Retrieve or create Hopsworks Feature Group for logging predictions."""
    try:
        fg = fs.get_or_create_feature_group(
            name="aqi_prediction_log",
            version=1,
            primary_key=["city", "forecast_timestamp"],
            description="Log of past AQI predictions against actuals",
            online_enabled=True,
        )
        logger.info("Retrieved or created Prediction Log Feature Group (v1)")
        return fg
    except Exception as err:
        logger.error("Failed to get or create Prediction Log FG: %s", err)
        return None

def write_prediction_log(df: pd.DataFrame, config: Any = None) -> bool:
    """Writes prediction log to Hopsworks and local CSV fallback."""
    # Write to local CSV fallback
    try:
        from pathlib import Path
        local_dir = Path(__file__).resolve().parent.parent / "data"
        local_dir.mkdir(exist_ok=True)
        csv_path = local_dir / "prediction_log.csv"
        
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            combined = pd.concat([existing_df, df]).drop_duplicates(subset=["city", "forecast_timestamp"], keep="last")
            combined.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)
        logger.info(f"Successfully appended {len(df)} rows to local prediction log CSV.")
    except Exception as e:
        logger.warning(f"Failed to write local prediction log CSV: {e}")

    # Write to Hopsworks
    try:
        fs = get_feature_store_connection(config)
        fg = get_or_create_prediction_log_group(fs)
        if fg is not None:
            logger.info("Writing into Hopsworks prediction log FG...")
            fg.insert(df)
            return True
    except Exception as err:
        logger.error("Failed to write prediction log to Hopsworks: %s", err)
        
    return False

def read_prediction_log(config: Any = None, days_back: int = 30) -> pd.DataFrame:
    """Reads prediction log from Hopsworks or local CSV fallback."""
    try:
        fs = get_feature_store_connection(config)
        fg = get_or_create_prediction_log_group(fs)
        df = fg.read()
        
        if df is not None and not df.empty:
            df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"], utc=True)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            df = df[df["forecast_timestamp"] >= cutoff]
            return df.sort_values("forecast_timestamp")
    except Exception as err:
        logger.warning("Failed to read from Hopsworks Prediction Log: %s. Trying local CSV...", err)
        
    # Local fallback
    try:
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent / "data" / "prediction_log.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"], utc=True)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            df = df[df["forecast_timestamp"] >= cutoff]
            return df.sort_values("forecast_timestamp")
    except Exception as e:
        logger.error(f"Failed to read local prediction log: {e}")
        
    return pd.DataFrame()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import src.config as config

        logger.info("Connecting to feature store...")
        fs = get_feature_store_connection(config)
        fg = get_or_create_feature_group(fs)

        print("\n--- Feature Group Schema ---")
        try:
            for feature in fg.schema:
                print(f" - {feature.name}: {feature.type}")
        except Exception:
            print("Feature Group:", fg)

        df = fg.read()
        row_count = len(df) if df is not None else 0
        print(f"\nTotal rows currently stored: {row_count}")

    except Exception as main_err:
        logger.error("Feature store demo failed: %s", main_err)
