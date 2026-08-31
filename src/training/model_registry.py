"""
Model Registry Module
Registers the winning model in Hopsworks Model Registry.
"""

import logging
import sys
from pathlib import Path
import os
import shutil
import tempfile

import pandas as pd

try:
    import hopsworks
except ImportError:
    hopsworks = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_model_registry(config):
    """
    Connects to Hopsworks and returns the model registry object.
    """
    if hopsworks is None:
        raise ImportError("The 'hopsworks' package is required.")
        
    def _get_conf(key: str):
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)

    api_key = _get_conf("HOPSWORKS_API_KEY")
    project_name = _get_conf("HOPSWORKS_PROJECT_NAME")

    if not api_key or not project_name:
        raise ValueError("HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set in config.")
        
    logger.info("Connecting to Hopsworks project '%s'...", project_name)
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )
    mr = project.get_model_registry()
    logger.info("Successfully connected to Hopsworks Model Registry.")
    return mr

def register_model(mr, model_path: str, model_name: str, metrics: dict, description: str, is_lstm: bool = False):
    """
    Registers the model artifact to Hopsworks.
    """
    temp_dir = tempfile.mkdtemp()
    
    # Copy the model file itself
    shutil.copy(model_path, os.path.join(temp_dir, Path(model_path).name))
    
    if is_lstm:
        scaler_path = Path(model_path).parent / "scaler.pkl"
        if scaler_path.exists():
            shutil.copy(scaler_path, os.path.join(temp_dir, "scaler.pkl"))
            
    # For baseline
    if not is_lstm:
        imputer_path = Path(model_path).parent / "imputer.pkl"
        if imputer_path.exists():
            shutil.copy(imputer_path, os.path.join(temp_dir, "imputer.pkl"))
            
    framework = "TENSORFLOW" if is_lstm else "SKLEARN"
    
    hw_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description
    )
    
    hw_model.save(temp_dir)
    logger.info(f"Successfully registered model '{model_name}' (Version: {hw_model.version}) to Hopsworks.")
    
    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    return hw_model

def get_latest_model_version(mr, model_name: str):
    """
    Retrieves the most recent version of a named model.
    """
    model = mr.get_model(model_name)
    return model

def main():
    try:
        import src.config as config
    except ImportError:
        logger.error("Failed to load project config.")
        sys.exit(1)
        
    reports_dir = project_root / "reports"
    metrics_path = reports_dir / "model_comparison_metrics.csv"
    
    if not metrics_path.exists():
        logger.error("Model comparison metrics not found. Run model_comparison.py first.")
        sys.exit(1)
        
    df = pd.read_csv(metrics_path)
    
    # Determine winner
    from src.training.model_comparison import select_best_model
    best_model_name_internal = select_best_model(df)
    
    if not best_model_name_internal:
        logger.warning("select_best_model returned None. No model beat persistence.")
        logger.info("Refusing to register. Please improve feature engineering or try different models instead.")
        sys.exit(0)
        
    best_row = df[df["model_name"] == best_model_name_internal].iloc[0]
    metrics = {
        "RMSE": float(best_row["rmse"]),
        "MAE": float(best_row["mae"]),
        "R2": float(best_row["r2"])
    }
    
    is_lstm = "LSTM" in best_model_name_internal
    
    models_dir = project_root / "models"
    if is_lstm:
        model_path = models_dir / "lstm_model.keras"
    else:
        model_path = models_dir / "best_baseline_model.pkl"
        
    if not model_path.exists():
        logger.error(f"Best model artifact not found at {model_path}")
        sys.exit(1)
        
    logger.info("Connecting to Hopsworks...")
    
    # IMPORTANT: Model registry versioning matters here because the CI/CD pipeline (Week 4) 
    # will retrain daily and push new versions — so the web app should always pull 
    # the "latest" version rather than hardcoding a version number.
    mr = get_model_registry(config)
    
    model_name = "aqi_forecaster"
    description = f"AQI forecasting model ({best_model_name_internal}). Predicts 3 days ahead."
    
    logger.info(f"Registering model: {model_name}...")
    hw_model = register_model(mr, str(model_path), model_name, metrics, description, is_lstm)
    
    print("\n" + "=" * 60)
    print("MODEL REGISTRATION SUCCESS")
    print("=" * 60)
    print(f"Registered Name  : {model_name}")
    print(f"Version          : {hw_model.version}")
    print(f"Framework        : {'TensorFlow' if is_lstm else 'scikit-learn'}")
    print(f"Metrics attached : {metrics}")
    
if __name__ == "__main__":
    main()
