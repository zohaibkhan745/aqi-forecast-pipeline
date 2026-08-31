"""
Inference Module for Streamlit Dashboard.
Handles generating forecasts from recent feature data.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

def get_aqi_category(aqi: float) -> str:
    """
    Maps numeric AQI to EPA color categories.
    """
    if pd.isna(aqi):
        return "Unknown"
    
    aqi = int(round(aqi))
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_color(category: str) -> str:
    """
    Maps EPA category to color for UI.
    """
    colors = {
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023",
        "Unknown": "#cccccc"
    }
    return colors.get(category, "#cccccc")

def generate_3day_forecast(model, scaler, imputer, recent_features: pd.DataFrame) -> pd.DataFrame:
    """
    Produces predictions for the next 3 days at 12-hour intervals.
    Handles both sklearn and Keras models transparently.
    Returns a DataFrame: forecast_timestamp, predicted_aqi, aqi_category
    """
    if recent_features.empty:
        raise ValueError("Cannot forecast: Recent features DataFrame is empty.")
        
    df = recent_features.copy()
    
    # We predict 72 hours ahead from the 'current' time.
    # The models are trained to predict 72 hours from any given feature row.
    # So if we take the latest feature row, its prediction is for T+72h.
    # If we want a progression over the next 3 days (e.g., T+12, T+24 ... T+72),
    # we can use the historical feature rows from (T-60h, T-48h, ..., T).
    # Since the target is shifted by 72h, predicting on the row from 48h ago gives the prediction for T+24h.
    
    # Ensure data is sorted ascending by fetched_at
    df = df.sort_values("fetched_at").reset_index(drop=True)
    
    # Get the latest timestamp
    latest_time = pd.to_datetime(df.iloc[-1]["fetched_at"])
    
    # We need predictions for the future: T+12, T+24, T+36, T+48, T+60, T+72
    # To predict T+72, we use row T.
    # To predict T+60, we use row T-12.
    # To predict T+48, we use row T-24.
    # To predict T+12, we use row T-60.
    
    target_horizons_hours = [12, 24, 36, 48, 60, 72]
    forecasts = []
    
    is_keras = hasattr(model, "predict") and "keras" in str(type(model)).lower()
    is_sklearn = not is_keras
    
    # Prepare features for the model
    # Drop non-numeric and target columns
    X_df = df.drop(columns=["target_aqi_3d", "fetched_at", "city", "season", "_sort_dt"], errors="ignore")
    # Also drop categorical that weren't encoded or keep them if they are numeric
    X_df = X_df.select_dtypes(include=[np.number])
    
    if is_sklearn:
        # Classical model (Ridge/RF)
        # Apply imputer
        if imputer:
            X = imputer.transform(X_df)
        else:
            X = X_df.values
            
        preds = model.predict(X)
        
    else:
        # LSTM Model
        # Requires 24-hour sequences.
        lookback = 24
        
        if len(X_df) < lookback:
            raise ValueError(f"Not enough data for LSTM. Need {lookback} rows, got {len(X_df)}.")
            
        if scaler:
            X_scaled = scaler.transform(X_df)
        else:
            X_scaled = X_df.values
            
        # We can create predictions for all possible sequences
        preds = []
        for i in range(len(X_scaled) - lookback + 1):
            seq = X_scaled[i : i + lookback]
            # shape (1, 24, features)
            pred = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0][0]
            preds.append(pred)
            
        # Pad the beginning with NaNs so it aligns with the dataframe rows
        preds = [np.nan] * (lookback - 1) + preds
        
    df["predicted_future_aqi"] = preds
    
    # Now extract the specific horizons
    for horizon in target_horizons_hours:
        # We need the row at T - (72 - horizon)
        hours_to_look_back = 72 - horizon
        
        # Find the row closest to (latest_time - hours_to_look_back)
        target_past_time = latest_time - timedelta(hours=hours_to_look_back)
        
        # Calculate time differences
        time_diffs = abs(pd.to_datetime(df["fetched_at"]) - target_past_time)
        closest_idx = time_diffs.idxmin()
        
        # Ensure the match is reasonably close (e.g. within 2 hours)
        if time_diffs[closest_idx] <= timedelta(hours=2):
            predicted_val = df.iloc[closest_idx]["predicted_future_aqi"]
            forecast_time = latest_time + timedelta(hours=horizon)
            
            forecasts.append({
                "forecast_timestamp": forecast_time,
                "predicted_aqi": float(predicted_val) if not pd.isna(predicted_val) else None,
                "aqi_category": get_aqi_category(predicted_val)
            })
            
    forecast_df = pd.DataFrame(forecasts)
    
    # If some values couldn't be predicted (e.g., due to LSTM lookback padding), filter them
    forecast_df = forecast_df.dropna(subset=["predicted_aqi"]).reset_index(drop=True)
    
    return forecast_df
