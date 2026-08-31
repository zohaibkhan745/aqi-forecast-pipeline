"""
Model Comparison Module
Compares Ridge Regression, Random Forest, and LSTM models against a persistence baseline.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_all_models():
    """
    Loads best_baseline_model, lstm_model, baseline_imputer, and lstm_scaler.
    """
    models_dir = project_root / "models"
    models_dict = {}
    
    baseline_path = models_dir / "best_baseline_model.pkl"
    imputer_path = models_dir / "imputer.pkl"
    
    if baseline_path.exists():
        models_dict["baseline"] = joblib.load(baseline_path)
        if imputer_path.exists():
            models_dict["imputer"] = joblib.load(imputer_path)
    
    lstm_path = models_dir / "lstm_model.keras"
    scaler_path = models_dir / "scaler.pkl"
    
    if lstm_path.exists() and load_model is not None:
        models_dict["lstm"] = load_model(lstm_path)
        if scaler_path.exists():
            models_dict["scaler"] = joblib.load(scaler_path)
            
    return models_dict

def run_full_comparison(test_df: pd.DataFrame, models_dict: dict):
    """
    Evaluates every model on the test set.
    Returns a DataFrame with RMSE, MAE, R2, beats_persistence.
    """
    results = []
    
    y_test = test_df["target_aqi_3d"].values
    
    # 1. Persistence Baseline
    if "aqi" in test_df.columns:
        pers_preds = test_df["aqi"].values
        pers_rmse = float(np.sqrt(mean_squared_error(y_test, pers_preds)))
        pers_mae = mean_absolute_error(y_test, pers_preds)
        pers_r2 = r2_score(y_test, pers_preds)
        results.append({
            "model_name": "Persistence",
            "rmse": pers_rmse,
            "mae": pers_mae,
            "r2": pers_r2,
            "beats_persistence": False
        })
    else:
        pers_rmse = float('inf')
        
    # 2. Baseline Model
    if "baseline" in models_dict and "imputer" in models_dict:
        X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number])
        X = models_dict["imputer"].transform(X_df)
        preds = models_dict["baseline"].predict(X)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({
            "model_name": type(models_dict["baseline"]).__name__,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "beats_persistence": rmse < pers_rmse
        })
        
    # 3. LSTM
    if "lstm" in models_dict and "scaler" in models_dict:
        lookback = 24
        if len(test_df) > lookback:
            X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number]).ffill().bfill()
            X_scaled = models_dict["scaler"].transform(X_df.values)
            
            X_seq = []
            for i in range(len(X_scaled) - lookback):
                X_seq.append(X_scaled[i : i + lookback])
            X_seq = np.array(X_seq)
            
            y_seq = y_test[lookback - 1 : len(y_test) - 1]
            if len(X_seq) == len(y_seq):
                preds = models_dict["lstm"].predict(X_seq, verbose=0).flatten()
                rmse = float(np.sqrt(mean_squared_error(y_seq, preds)))
                mae = mean_absolute_error(y_seq, preds)
                r2 = r2_score(y_seq, preds)
                
                if "aqi" in test_df.columns:
                    pers_preds_seq = test_df["aqi"].values[lookback - 1 : len(y_test) - 1]
                    local_pers_rmse = float(np.sqrt(mean_squared_error(y_seq, pers_preds_seq)))
                else:
                    local_pers_rmse = pers_rmse
                    
                results.append({
                    "model_name": "LSTM",
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "beats_persistence": rmse < local_pers_rmse
                })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("rmse", ascending=True).reset_index(drop=True)
    return df_res

def plot_predictions_vs_actual(test_df: pd.DataFrame, models_dict: dict):
    """
    Plots predicted vs actual over time.
    """
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    y_test = test_df["target_aqi_3d"].values
    if "fetched_at" in test_df.columns:
        dates = pd.to_datetime(test_df["fetched_at"])
    else:
        dates = range(len(test_df))
        
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_test, label="Actual AQI (Target 3d)", color='black', linewidth=2)
    
    if "aqi" in test_df.columns:
        plt.plot(dates, test_df["aqi"].values, label="Persistence", linestyle='--', alpha=0.7)
        
    if "baseline" in models_dict and "imputer" in models_dict:
        X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number])
        X = models_dict["imputer"].transform(X_df)
        preds = models_dict["baseline"].predict(X)
        plt.plot(dates, preds, label=type(models_dict["baseline"]).__name__, alpha=0.8)
        
    if "lstm" in models_dict and "scaler" in models_dict:
        lookback = 24
        if len(test_df) > lookback:
            X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number]).ffill().bfill()
            X_scaled = models_dict["scaler"].transform(X_df.values)
            
            X_seq = []
            for i in range(len(X_scaled) - lookback):
                X_seq.append(X_scaled[i : i + lookback])
            X_seq = np.array(X_seq)
            
            if len(X_seq) > 0:
                preds = models_dict["lstm"].predict(X_seq, verbose=0).flatten()
                plot_dates = dates.iloc[lookback - 1 : len(dates) - 1] if isinstance(dates, pd.Series) else dates[lookback - 1 : len(dates) - 1]
                plt.plot(plot_dates, preds, label="LSTM", alpha=0.8)
                
    plt.title("Predictions vs Actual AQI")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    
    save_path = reports_dir / "model_comparison_predictions.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved predictions plot to {save_path}")

def select_best_model(comparison_df: pd.DataFrame):
    """
    Returns name of best model based on RMSE.
    """
    if comparison_df.empty:
        return None
        
    valid_models = comparison_df[comparison_df["beats_persistence"] == True]
    
    if valid_models.empty:
        logger.warning("No model beats the persistence baseline!")
        return None
        
    best_model = valid_models.iloc[0]["model_name"]
    return best_model

def extract_predictions_for_log(test_df: pd.DataFrame, models_dict: dict, best_model_name: str) -> pd.DataFrame:
    """Extracts predictions from the winning model to save into the prediction log."""
    if "fetched_at" not in test_df.columns or "target_aqi_3d" not in test_df.columns:
        return pd.DataFrame()
        
    dates = pd.to_datetime(test_df["fetched_at"])
    y_test = test_df["target_aqi_3d"].values
    city = test_df["city"].values[0] if "city" in test_df.columns and len(test_df) > 0 else "Unknown"
    
    # We will compute forecast timestamp by adding 72 hours to the fetched_at time.
    forecast_timestamps = dates + pd.Timedelta(hours=72)
    
    preds = None
    aligned_forecast_timestamps = forecast_timestamps
    aligned_y_test = y_test
    
    if "LSTM" in best_model_name.upper():
        if "lstm" in models_dict and "scaler" in models_dict:
            lookback = 24
            if len(test_df) > lookback:
                X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number]).ffill().bfill()
                X_scaled = models_dict["scaler"].transform(X_df.values)
                
                X_seq = []
                for i in range(len(X_scaled) - lookback):
                    X_seq.append(X_scaled[i : i + lookback])
                X_seq = np.array(X_seq)
                
                if len(X_seq) > 0:
                    preds = models_dict["lstm"].predict(X_seq, verbose=0).flatten()
                    aligned_forecast_timestamps = forecast_timestamps.iloc[lookback - 1 : len(forecast_timestamps) - 1]
                    aligned_y_test = y_test[lookback - 1 : len(y_test) - 1]
    else:
        # Sklearn model
        if "baseline" in models_dict and "imputer" in models_dict:
            X_df = test_df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore").select_dtypes(include=[np.number])
            X = models_dict["imputer"].transform(X_df)
            preds = models_dict["baseline"].predict(X)
            
    if preds is None:
        return pd.DataFrame()
        
    log_df = pd.DataFrame({
        "city": [city] * len(preds),
        "forecast_timestamp": aligned_forecast_timestamps,
        "predicted_aqi": preds,
        "actual_aqi": aligned_y_test,
        "model_version": best_model_name
    })
    
    return log_df

def main():
    data_dir = project_root / "data" / "processed"
    if not (data_dir / "test.csv").exists():
        logger.error("Test data not found.")
        sys.exit(1)
        
    test_df = pd.read_csv(data_dir / "test.csv")
    models_dict = load_all_models()
    
    if not models_dict:
        logger.error("No trained models found.")
        sys.exit(1)
        
    logger.info("Running full comparison...")
    comparison_df = run_full_comparison(test_df, models_dict)
    
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (TEST SET)")
    print("=" * 60)
    
    table_data = []
    for _, row in comparison_df.iterrows():
        table_data.append([
            row["model_name"],
            f"{row['rmse']:.2f}",
            f"{row['mae']:.2f}",
            f"{row['r2']:.2f}",
            "Yes" if row["beats_persistence"] else "No"
        ])
        
    print(tabulate(table_data, headers=["Model", "RMSE", "MAE", "R²", "Beats Persistence"], tablefmt="github"))
    
    plot_predictions_vs_actual(test_df, models_dict)
    
    best_model = select_best_model(comparison_df)
    if best_model:
        print(f"\nRecommendation: Register {best_model} for production.")
        
        # Log predictions for dashboard overlay
        try:
            import src.config as config
            from src.feature_store import write_prediction_log
            log_df = extract_predictions_for_log(test_df, models_dict, best_model)
            if not log_df.empty:
                write_prediction_log(log_df, config)
        except Exception as e:
            logger.warning(f"Could not write prediction log: {e}")
            
    else:
        print("\nRecommendation: DO NOT register any model. Improve feature engineering or try other architectures.")
        
    reports_dir = project_root / "reports"
    comparison_df.to_csv(reports_dir / "model_comparison_metrics.csv", index=False)

if __name__ == "__main__":
    main()
