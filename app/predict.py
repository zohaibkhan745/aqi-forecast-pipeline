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
    
    # Ensure all remaining columns are treated as numeric (Hopsworks/CSV can sometimes return them as object if they contain NaNs)
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        
    X_df = X_df.select_dtypes(include=[np.number])
    
    # Align columns to what the model/imputer/scaler expects
    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif imputer and hasattr(imputer, "feature_names_in_"):
        expected_cols = list(imputer.feature_names_in_)
    elif scaler and hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        
    if expected_cols is not None:
        X_df = X_df.reindex(columns=expected_cols)
    
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

def generate_shap_explanation(model, scaler, imputer, recent_features: pd.DataFrame) -> dict:
    """
    Computes SHAP values for the most recent feature row.
    Returns a dictionary mapping feature names to their absolute SHAP importance.
    """
    import shap
    
    if recent_features.empty:
        return {}
        
    df = recent_features.copy()
    df = df.sort_values("fetched_at").reset_index(drop=True)
    
    is_keras = hasattr(model, "predict") and "keras" in str(type(model)).lower()
    
    X_df = df.drop(columns=["target_aqi_3d", "fetched_at", "city", "season", "_sort_dt"], errors="ignore")
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.select_dtypes(include=[np.number])
    
    # Align columns to what the model/imputer/scaler expects
    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif imputer and hasattr(imputer, "feature_names_in_"):
        expected_cols = list(imputer.feature_names_in_)
    elif scaler and hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        
    if expected_cols is not None:
        X_df = X_df.reindex(columns=expected_cols)
        
    feature_names = X_df.columns.tolist()
    
    if is_keras:
        # LSTM Model
        lookback = 24
        if len(X_df) < lookback:
            return {}
            
        if scaler:
            X_scaled = scaler.transform(X_df)
        else:
            X_scaled = X_df.values
            
        seq = X_scaled[-lookback:]
        # Use a background of a few recent sequences to speed up DeepExplainer
        background = []
        for i in range(max(0, len(X_scaled) - lookback - 10), len(X_scaled) - lookback):
            background.append(X_scaled[i:i+lookback])
            
        if not background:
            # Not enough data for a background, fallback to dummy
            background = [np.zeros_like(seq)]
            
        background = np.array(background)
        
        try:
            # GradientExplainer works well for Keras sequences
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(np.expand_dims(seq, axis=0))
            
            # shape could be (1, 24, features) depending on SHAP version/model
            # we want the aggregate impact per feature
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
                
            if len(shap_vals.shape) == 3:
                # Average over the lookback window
                shap_vals = np.abs(shap_vals[0]).mean(axis=0)
            else:
                shap_vals = np.abs(shap_vals[0])
                
        except Exception as e:
            print(f"SHAP explanation failed for LSTM: {e}")
            return {}
            
    else:
        # Classical model (Ridge/RF)
        if imputer:
            X = imputer.transform(X_df)
        else:
            X = X_df.values
            
        instance = X[-1].reshape(1, -1)
        background = X[-100:] if len(X) > 100 else X
        
        try:
            if type(model).__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance)
            elif type(model).__name__ in ['Ridge', 'LinearRegression']:
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(instance)
            else:
                # Generic explainer
                explainer = shap.Explainer(model, background)
                shap_values = explainer(instance).values
                
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[0])
            else:
                shap_vals = np.abs(shap_values[0])
                
        except Exception as e:
            print(f"SHAP explanation failed for sklearn: {e}")
            return {}
            
    # Combine feature names with their absolute SHAP importance
    importance_dict = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    # Sort by importance descending
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    return importance_dict
