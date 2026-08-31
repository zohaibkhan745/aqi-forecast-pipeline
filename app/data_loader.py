"""
Data Loader for Streamlit Dashboard.
Handles fetching models and feature data from Hopsworks.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config, feature_store

logger = logging.getLogger(__name__)

# Fallback models mapping if offline or registry fails
FALLBACK_DIR = project_root / "models"

@st.cache_resource
def load_latest_model_and_scaler(config_dict=None):
    """
    Connects to Hopsworks model registry and downloads the latest version of the model.
    Downloads scaler.pkl and imputer.pkl if they exist alongside it.
    """
    if config_dict is None:
        config_dict = config
        
    try:
        import hopsworks
        import joblib
        
        # Determine if we should even try online mode (useful for testing)
        if not os.getenv("HOPSWORKS_API_KEY"):
            raise ValueError("HOPSWORKS_API_KEY not found in environment.")
            
        project_name = getattr(config_dict, "HOPSWORKS_PROJECT_NAME", os.getenv("HOPSWORKS_PROJECT_NAME"))
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=project_name
        )
        mr = project.get_model_registry()
        
        model = mr.get_best_model("aqi_forecaster", "rmse", "min")
        if model is None:
            model = mr.get_model("aqi_forecaster")  # get latest if no metrics
            
        model_dir = model.download()
        
        # Identify framework
        is_lstm = model.framework == "TENSORFLOW"
        
        loaded_model = None
        scaler = None
        imputer = None
        
        if is_lstm:
            from tensorflow.keras.models import load_model
            model_path = Path(model_dir) / "lstm_model.keras"
            scaler_path = Path(model_dir) / "scaler.pkl"
            
            if model_path.exists():
                loaded_model = load_model(str(model_path))
            if scaler_path.exists():
                scaler = joblib.load(str(scaler_path))
        else:
            model_path = Path(model_dir) / "best_baseline_model.pkl"
            imputer_path = Path(model_dir) / "imputer.pkl"
            
            if model_path.exists():
                loaded_model = joblib.load(str(model_path))
            if imputer_path.exists():
                imputer = joblib.load(str(imputer_path))
                
        version = model.version
        updated_at = model.created
        
        return loaded_model, scaler, imputer, version, updated_at
        
    except Exception as e:
        logger.warning(f"Could not load model from Hopsworks registry: {e}. Falling back to local models directory.")
        # Fallback to local files
        import joblib
        
        loaded_model = None
        scaler = None
        imputer = None
        is_lstm = False
        
        # Check if local files exist
        if (FALLBACK_DIR / "lstm_model.keras").exists():
            from tensorflow.keras.models import load_model
            loaded_model = load_model(str(FALLBACK_DIR / "lstm_model.keras"))
            is_lstm = True
            if (FALLBACK_DIR / "scaler.pkl").exists():
                scaler = joblib.load(str(FALLBACK_DIR / "scaler.pkl"))
        elif (FALLBACK_DIR / "best_baseline_model.pkl").exists():
            loaded_model = joblib.load(str(FALLBACK_DIR / "best_baseline_model.pkl"))
            if (FALLBACK_DIR / "imputer.pkl").exists():
                imputer = joblib.load(str(FALLBACK_DIR / "imputer.pkl"))
                
        if loaded_model is None:
            raise RuntimeError("Could not load any model from Hopsworks or local fallback directory.")
            
        return loaded_model, scaler, imputer, "Local Fallback", "Unknown"


@st.cache_data(ttl=3600)
def load_recent_features(hours_back=48):
    """
    Pulls recent feature rows from the feature store.
    Cached for 1 hour to match pipeline cadence.
    """
    try:
        # Use existing feature store logic
        fs = feature_store.get_feature_store_connection(config)
        fg = feature_store.get_or_create_feature_group(fs)
        
        city = getattr(config, "CITY_NAME", os.getenv("CITY_NAME", "Lahore"))
        
        history_records = feature_store.read_recent_history(fg, city=city, hours_back=hours_back)
        
        if not history_records:
            return pd.DataFrame()
            
        df = pd.DataFrame(history_records)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"])
        df = df.sort_values("fetched_at", ascending=True).reset_index(drop=True)
        return df
        
    except Exception as e:
        logger.warning(f"Could not fetch features from Hopsworks: {e}. Attempting local fallback.")
        # Fallback to local csv if available (useful for offline testing)
        local_path = project_root / "data" / "processed" / "aqi_features_snapshot.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)
            df["fetched_at"] = pd.to_datetime(df["fetched_at"])
            df = df.sort_values("fetched_at", ascending=True).tail(hours_back).reset_index(drop=True)
            return df
            
        raise RuntimeError("Could not load feature history from Hopsworks or local fallback.")
