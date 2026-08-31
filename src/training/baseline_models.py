"""
Baseline Models Module
Trains Ridge Regression and Random Forest models and compares them against a persistence baseline.
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def prepare_imputed_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits imputer on train, transforms train, val, and test.
    Returns: X_train_imp, X_val_imp, X_test_imp, imputer
    """
    imputer = SimpleImputer(strategy="median")
    
    # We must ensure X_train_imp remains a DataFrame to pass the test's `.isnull()` checks
    columns = X_train.columns
    
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=columns)
    
    return X_train_imp, X_val_imp, X_test_imp, imputer

def train_ridge_regression(X_train, y_train, X_val, y_val, alphas=None):
    """
    Grid searches Ridge Regression over alpha values.
    Returns best model and dictionary of validation RMSEs.
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
        
    best_model = None
    best_rmse = float("inf")
    results = {}
    
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        results[alpha] = rmse
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            
    return best_model, results

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators_list=None, max_depth_list=None):
    """
    Grid searches Random Forest over n_estimators and max_depth.
    Returns best model, dictionary of validation RMSEs, and feature importances.
    """
    if n_estimators_list is None:
        n_estimators_list = [100, 200]
    if max_depth_list is None:
        max_depth_list = [10, 20, None]
        
    best_model = None
    best_rmse = float("inf")
    results = {}
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            results[(n_estimators, max_depth)] = rmse
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                
    # Sort feature importances descending
    importances = best_model.feature_importances_
    feature_names = X_train.columns if hasattr(X_train, 'columns') else range(len(importances))
    feature_importances = {name: float(imp) for name, imp in zip(feature_names, importances)}
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
        
    return best_model, results, feature_importances

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and computes a persistence baseline.
    """
    # Model predictions
    preds = model.predict(X_test)
    model_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    model_mae = float(mean_absolute_error(y_test, preds))
    model_r2 = float(r2_score(y_test, preds))
    
    # Persistence baseline (using 'aqi' from X_test if available)
    if hasattr(X_test, 'columns') and "aqi" in X_test.columns:
        persistence_preds = X_test["aqi"]
        pers_rmse = float(np.sqrt(mean_squared_error(y_test, persistence_preds)))
        pers_mae = float(mean_absolute_error(y_test, persistence_preds))
        pers_r2 = float(r2_score(y_test, persistence_preds))
    else:
        pers_rmse, pers_mae, pers_r2 = None, None, None
        
    improvement_rmse = (pers_rmse - model_rmse) if pers_rmse is not None else None
    beats_persistence = (model_rmse < pers_rmse) if pers_rmse is not None else False
        
    return {
        "model_metrics": {
            "rmse": model_rmse,
            "mae": model_mae,
            "r2": model_r2
        },
        "persistence_metrics": {
            "rmse": pers_rmse,
            "mae": pers_mae,
            "r2": pers_r2
        },
        "improvement_rmse": improvement_rmse,
        "beats_persistence": bool(beats_persistence)
    }

def main():
    data_dir = project_root / "data" / "processed"
    if not (data_dir / "train.csv").exists():
        logger.error("Training data not found. Please run data_prep.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    def extract_xy(df):
        y = df["target_aqi_3d"]
        X = df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        return X, y
        
    X_tr_raw, y_train = extract_xy(train_df)
    X_va_raw, y_val = extract_xy(val_df)
    X_te_raw, y_test = extract_xy(test_df)
    
    X_train, X_val, X_test, imputer = prepare_imputed_features(X_tr_raw, X_va_raw, X_te_raw)
    
    logger.info("Training Ridge Regression...")
    ridge_model, ridge_results = train_ridge_regression(X_train, y_train, X_val, y_val)
    logger.info(f"Ridge Validation RMSEs: {ridge_results}")
    
    logger.info("Training Random Forest...")
    rf_model, rf_results, feature_importances = train_random_forest(X_train, y_train, X_val, y_val)
    logger.info(f"RF Validation RMSEs: {rf_results}")
    
    # Evaluate model uses X_te_raw so it can access 'aqi' for persistence
    ridge_eval = evaluate_model(ridge_model, X_te_raw, y_test)
    rf_eval = evaluate_model(rf_model, X_te_raw, y_test)
    
    # Compare
    table = []
    pers_metrics = ridge_eval["persistence_metrics"]
    table.append(["Persistence (Baseline)", pers_metrics["rmse"], pers_metrics["mae"], pers_metrics["r2"]])
    
    rm_metrics = ridge_eval["model_metrics"]
    table.append(["Ridge Regression", rm_metrics["rmse"], rm_metrics["mae"], rm_metrics["r2"]])
    
    rfm_metrics = rf_eval["model_metrics"]
    table.append(["Random Forest", rfm_metrics["rmse"], rfm_metrics["mae"], rfm_metrics["r2"]])
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 60)
    print(tabulate(table, headers=["Model", "RMSE", "MAE", "R²"], tablefmt="github"))
    
    best_model_name = "Ridge Regression" if rm_metrics["rmse"] < rfm_metrics["rmse"] else "Random Forest"
    best_model = ridge_model if best_model_name == "Ridge Regression" else rf_model
    best_rmse = min(rm_metrics["rmse"], rfm_metrics["rmse"])
    
    pers_rmse = pers_metrics["rmse"]
    if pers_rmse is not None:
        if best_rmse < pers_rmse:
            diff = pers_rmse - best_rmse
            print(f"\nConclusion: {best_model_name} won and beat the persistence baseline by {diff:.2f} RMSE.")
        else:
            print(f"\nWARNING: {best_model_name} failed to beat the persistence baseline! "
                  f"Model RMSE ({best_rmse:.2f}) >= Persistence RMSE ({pers_rmse:.2f}). "
                  "The model is not learning anything useful yet.")
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_baseline_model.pkl"
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best baseline model to {best_model_path}")

    # Also save the imputer for future inference
    imputer_path = models_dir / "imputer.pkl"
    joblib.dump(imputer, imputer_path)
    logger.info(f"Saved imputer to {imputer_path}")

if __name__ == "__main__":
    main()
