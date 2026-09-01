import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging
from datetime import datetime

from app import data_loader, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AQI Forecast API",
    description="API for fetching 72-hour AQI forecasts and feature explanations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching model and features
_model = None
_scaler = None
_imputer = None
_model_version = None
_model_updated_at = None
_last_features = None
_last_features_time = None

def _get_model_and_features():
    global _model, _scaler, _imputer, _model_version, _model_updated_at, _last_features, _last_features_time
    
    # Reload features if we don't have them or they are old
    now = datetime.now()
    if _last_features is None or _last_features_time is None or (now - _last_features_time).total_seconds() > 3600:
        logger.info("Loading recent features...")
        try:
            # We bypass streamlit's cache_data by calling the internal logic or just relying on it if it works in FastAPI context
            _last_features = data_loader.load_recent_features(hours_back=120)
            _last_features_time = now
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise HTTPException(status_code=500, detail="Failed to load feature data.")
            
    # Load model if not loaded
    if _model is None:
        logger.info("Loading model...")
        try:
            _model, _scaler, _imputer, _model_version, _model_updated_at = data_loader.load_latest_model_and_scaler()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load model.")
            
    return _model, _scaler, _imputer, _last_features, _model_version, _model_updated_at

@app.get("/health")
def health_check():
    """Returns API health and loaded model version status."""
    try:
        model, _, _, _, version, updated_at = _get_model_and_features()
        return {
            "status": "healthy",
            "model_version": version,
            "model_updated_at": str(updated_at) if updated_at != "Unknown" else "Unknown"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/forecast")
def get_forecast():
    """Fetches recent features and returns the 72-hour forecast sequence."""
    try:
        model, scaler, imputer, recent_features, _, _ = _get_model_and_features()
        
        if recent_features.empty:
            raise HTTPException(status_code=404, detail="No recent features found.")
            
        forecast_df = predict.generate_3day_forecast(model, scaler, imputer, recent_features)
        
        # Convert to dictionary format
        forecasts = []
        for _, row in forecast_df.iterrows():
            forecasts.append({
                "timestamp": row["forecast_timestamp"].isoformat(),
                "predicted_aqi": row["predicted_aqi"],
                "category": row["aqi_category"]
            })
            
        # Also return current state
        latest_row = recent_features.iloc[-1]
        current_aqi = latest_row.get("aqi")
        
        return {
            "current_aqi": float(current_aqi) if pd.notna(current_aqi) else None,
            "current_aqi_category": predict.get_aqi_category(current_aqi),
            "forecasts": forecasts
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain")
def get_explanation():
    """Computes SHAP values for the most recent prediction and returns feature importance scores."""
    try:
        model, scaler, imputer, recent_features, _, _ = _get_model_and_features()
        
        if recent_features.empty:
            raise HTTPException(status_code=404, detail="No recent features found.")
            
        importance_dict = predict.generate_shap_explanation(model, scaler, imputer, recent_features)
        
        if not importance_dict:
            raise HTTPException(status_code=500, detail="Failed to generate SHAP explanation. Model might not be supported.")
            
        return {
            "feature_importance": importance_dict
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
