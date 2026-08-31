"""
Deep Learning Model Module
Trains an LSTM for AQI forecasting.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    tf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 24):
    """
    Converts flat feature rows into sequences of `lookback` past hours per sample.
    Handles the boundary correctly so no sample uses data from before the start of the given split.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i : i + lookback])
        # The target corresponds to the time step at the end of the sequence
        y_seq.append(y[i + lookback - 1])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(n_features: int, lookback: int = 24):
    """
    Builds the LSTM model.
    """
    if tf is None:
        raise ImportError("Tensorflow is not installed.")
        
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, batch_size=32):
    """
    Trains the LSTM with early stopping and reduce LR on plateau.
    """
    model = build_lstm_model(n_features=X_train_seq.shape[2], lookback=X_train_seq.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    )
    
    # Configure TF to run on CPU if GPU issues happen
    tf.config.set_visible_devices([], 'GPU')
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """
    Plots train vs val loss and saves to reports.
    """
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('LSTM Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = reports_dir / "lstm_training_curve.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved training curve to {save_path}")

def evaluate_model(model, X_test_seq, y_test_seq, persistence_preds_seq=None):
    """
    Evaluates the model on test data and computes a persistence baseline.
    """
    preds = model.predict(X_test_seq).flatten()
    model_rmse = float(np.sqrt(mean_squared_error(y_test_seq, preds)))
    model_mae = mean_absolute_error(y_test_seq, preds)
    model_r2 = r2_score(y_test_seq, preds)
    
    if persistence_preds_seq is not None:
        pers_rmse = float(np.sqrt(mean_squared_error(y_test_seq, persistence_preds_seq)))
        pers_mae = mean_absolute_error(y_test_seq, persistence_preds_seq)
        pers_r2 = r2_score(y_test_seq, persistence_preds_seq)
    else:
        pers_rmse, pers_mae, pers_r2 = None, None, None
        
    return {
        "model_metrics": {
            "RMSE": model_rmse,
            "MAE": model_mae,
            "R2": model_r2
        },
        "persistence_metrics": {
            "RMSE": pers_rmse,
            "MAE": pers_mae,
            "R2": pers_r2
        }
    }

def main():
    if tf is None:
        logger.error("TensorFlow not installed. Please install tensorflow to run deep models.")
        sys.exit(1)
        
    data_dir = project_root / "data" / "processed"
    if not (data_dir / "train.csv").exists():
        logger.error("Training data not found.")
        sys.exit(1)
        
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    def prep_for_lstm(df):
        y = df["target_aqi_3d"].values
        X_df = df.drop(columns=["target_aqi_3d", "fetched_at", "city"], errors="ignore")
        X_df = X_df.select_dtypes(include=[np.number])
        X_df = X_df.ffill().bfill()
        return X_df.values, y
        
    X_train_raw, y_train = prep_for_lstm(train_df)
    X_val_raw, y_val = prep_for_lstm(val_df)
    X_test_raw, y_test = prep_for_lstm(test_df)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    lookback = 24
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
    
    logger.info(f"Train sequences shape: {X_train_seq.shape}")
    logger.info("Training LSTM...")
    model, history = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)
    
    plot_training_history(history)
    
    if "aqi" in test_df.columns:
        aqi_values = test_df["aqi"].values
        persistence_preds_seq = aqi_values[lookback - 1 : len(aqi_values) - 1]
    else:
        persistence_preds_seq = None
        
    eval_metrics = evaluate_model(model, X_test_seq, y_test_seq, persistence_preds_seq)
    
    logger.info("LSTM Evaluation Results:")
    logger.info(f"Model RMSE: {eval_metrics['model_metrics']['RMSE']:.2f}")
    logger.info(f"Model MAE: {eval_metrics['model_metrics']['MAE']:.2f}")
    logger.info(f"Model R2: {eval_metrics['model_metrics']['R2']:.2f}")
    
    if persistence_preds_seq is not None:
        logger.info(f"Persistence RMSE: {eval_metrics['persistence_metrics']['RMSE']:.2f}")
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "lstm_model.keras"
    model.save(model_path)
    logger.info(f"Saved LSTM model to {model_path}")
    
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved LSTM scaler to {scaler_path}")

if __name__ == "__main__":
    main()
