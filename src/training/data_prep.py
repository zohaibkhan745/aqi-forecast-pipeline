"""
Data Preparation Module for ML Modeling.

IMPORTANT: Chronological Splitting for Time Series
For time series forecasting, chronological (not random) splitting is critical. 
Randomly shuffling data before splitting would cause data leakage, allowing the model 
to train on future information to predict past events, which violates the reality of 
how the model will operate in production. We must strictly split on a timeline: 
past -> train, recent past -> val, present -> test.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def create_forecast_target(df: pd.DataFrame, horizon_hours: int = 72, target_col: str = "aqi") -> pd.DataFrame:
    """
    Sorts by timestamp ascending and creates a new target column by shifting 
    the target_col forward by horizon_hours.
    Drops rows at the end of the dataset where the shifted target is NaN.
    """
    df = df.copy()
    if df.empty or target_col not in df.columns or "fetched_at" not in df.columns:
        return df
        
    df["_sort_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.sort_values(by="_sort_dt", ascending=True).reset_index(drop=True)
    df = df.drop(columns=["_sort_dt"])
    
    if horizon_hours == 72:
        new_target_col = "target_aqi_3d"
    else:
        new_target_col = f"target_aqi_{horizon_hours}h"
        
    df[new_target_col] = df[target_col].shift(-horizon_hours)
    
    # Drop rows at the end where target is NaN
    df = df.dropna(subset=[new_target_col]).reset_index(drop=True)
    
    return df

def remove_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that encode future information or are data-quality artifacts
    (e.g., is_imputed_*, *_outlier flags).
    """
    df = df.copy()
    columns_to_drop = []
    
    for col in df.columns:
        if "is_imputed" in col.lower() or "outlier" in col.lower() or "imputed_" in col.lower():
            columns_to_drop.append(col)
            
    if columns_to_drop:
        logger.info(f"Removing leaky/data-quality columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
        
    return df

def time_based_train_test_split(df: pd.DataFrame, test_size: float = 0.2, target_col: str = "target_aqi_3d") -> tuple:
    """
    Splits the dataset chronologically into train and test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    if "fetched_at" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
        df = df.sort_values(by="_sort_dt", ascending=True).reset_index(drop=True)
        df = df.drop(columns=["_sort_dt"])
        
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    cols_to_drop = [target_col]
    for col in ["fetched_at", "city"]:
        if col in df.columns:
            cols_to_drop.append(col)
        
    X_train = train_df.drop(columns=cols_to_drop, errors="ignore")
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=cols_to_drop, errors="ignore")
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def time_based_train_val_test_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, target_col: str = "target_aqi_3d") -> dict:
    """
    Splits the dataset chronologically into train, val, and test sets.
    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test.
    """
    if "fetched_at" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
        df = df.sort_values(by="_sort_dt", ascending=True).reset_index(drop=True)
        df = df.drop(columns=["_sort_dt"])
        
    val_split_idx = int(len(df) * (1 - (val_size + test_size)))
    test_split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:test_split_idx]
    test_df = df.iloc[test_split_idx:]
    
    cols_to_drop = [target_col]
    for col in ["fetched_at", "city"]:
        if col in df.columns:
            cols_to_drop.append(col)
        
    return {
        "X_train": train_df.drop(columns=cols_to_drop, errors="ignore"),
        "y_train": train_df[target_col],
        "X_val": val_df.drop(columns=cols_to_drop, errors="ignore"),
        "y_val": val_df[target_col],
        "X_test": test_df.drop(columns=cols_to_drop, errors="ignore"),
        "y_test": test_df[target_col]
    }

def save_processed_splits(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, output_dir: str = "data/processed") -> dict:
    """
    Chronologically splits the dataframe and saves train, val, test CSVs.
    Returns a dict with paths.
    """
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    
    if "fetched_at" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
        df = df.sort_values(by="_sort_dt", ascending=True).reset_index(drop=True)
        df = df.drop(columns=["_sort_dt"])
        
    val_split_idx = int(len(df) * (1 - (val_size + test_size)))
    test_split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:test_split_idx]
    test_df = df.iloc[test_split_idx:]
    
    train_path = out_path / "train.csv"
    val_path = out_path / "val.csv"
    test_path = out_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

def main():
    input_path = project_root / "data" / "processed" / "aqi_features_snapshot.csv"
    if not input_path.exists():
        logger.error(f"Clean CSV file not found: {input_path}")
        sys.exit(1)
        
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Create forecast target
    df = create_forecast_target(df, horizon_hours=72, target_col="aqi")
    
    # 2. Remove leaky features
    df = remove_leaky_features(df)
    
    # 3. Save splits
    save_processed_splits(df)

if __name__ == "__main__":
    main()
