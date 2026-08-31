import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.training.data_prep import (
    create_forecast_target,
    remove_leaky_features,
    time_based_train_test_split,
    time_based_train_val_test_split,
    save_processed_splits,
)


class TestDataPrep(unittest.TestCase):

    def setUp(self):
        # Create 100 hourly records
        self.dates = pd.date_range(start="2026-01-01 00:00:00", periods=100, freq="h", tz="UTC")
        self.df = pd.DataFrame({
            "fetched_at": self.dates.astype(str),
            "city": "Lahore",
            "aqi": np.arange(100, 200),
            "pm25": np.arange(50, 150),
            "temperature": np.linspace(20, 30, 100),
            "humidity": np.linspace(50, 70, 100),
            "is_imputed_gap": [0] * 100,  # Leaky/cleaning flag
            "is_outlier": [0] * 100,      # Leaky/cleaning flag
        })

    def test_create_forecast_target(self):
        # Horizon 10 hours for testing
        df_target = create_forecast_target(self.df, horizon_hours=10, target_col="aqi")
        
        # 100 rows - 10 trailing rows = 90 rows
        self.assertEqual(len(df_target), 90)
        self.assertIn("target_aqi_10h", df_target.columns)
        
        # Check that target at row 0 equals aqi at row 10 (100 + 10 = 110)
        self.assertEqual(df_target.iloc[0]["target_aqi_10h"], 110)
        self.assertEqual(df_target.iloc[5]["target_aqi_10h"], 115)

    def test_create_forecast_target_72h(self):
        df_target = create_forecast_target(self.df, horizon_hours=72, target_col="aqi")
        
        # 100 rows - 72 trailing rows = 28 rows
        self.assertEqual(len(df_target), 28)
        self.assertIn("target_aqi_3d", df_target.columns)
        self.assertEqual(df_target.iloc[0]["target_aqi_3d"], 172)

    def test_remove_leaky_features(self):
        df_clean = remove_leaky_features(self.df)
        self.assertNotIn("is_imputed_gap", df_clean.columns)
        self.assertNotIn("is_outlier", df_clean.columns)
        self.assertIn("aqi", df_clean.columns)
        self.assertIn("pm25", df_clean.columns)

    def test_time_based_train_test_split(self):
        df_target = create_forecast_target(self.df, horizon_hours=10, target_col="aqi")
        df_clean = remove_leaky_features(df_target)

        X_train, X_test, y_train, y_test = time_based_train_test_split(
            df_clean, test_size=0.2, target_col="target_aqi_10h"
        )

        total_rows = len(df_clean)  # 90
        expected_train_len = int(total_rows * 0.8)  # 72
        expected_test_len = total_rows - expected_train_len  # 18

        self.assertEqual(len(X_train), expected_train_len)
        self.assertEqual(len(y_train), expected_train_len)
        self.assertEqual(len(X_test), expected_test_len)
        self.assertEqual(len(y_test), expected_test_len)

        # Target should not be in feature matrices
        self.assertNotIn("target_aqi_10h", X_train.columns)
        self.assertNotIn("target_aqi_10h", X_test.columns)

        # Metadata should be excluded from X
        self.assertNotIn("fetched_at", X_train.columns)
        self.assertNotIn("city", X_train.columns)

    def test_time_based_train_val_test_split(self):
        df_target = create_forecast_target(self.df, horizon_hours=10, target_col="aqi")
        df_clean = remove_leaky_features(df_target)

        splits = time_based_train_val_test_split(
            df_clean, val_size=0.15, test_size=0.15, target_col="target_aqi_10h"
        )

        self.assertIn("X_train", splits)
        self.assertIn("y_train", splits)
        self.assertIn("X_val", splits)
        self.assertIn("y_val", splits)
        self.assertIn("X_test", splits)
        self.assertIn("y_test", splits)

        total_rows = len(df_clean)  # 90
        train_len = len(splits["X_train"])
        val_len = len(splits["X_val"])
        test_len = len(splits["X_test"])

        self.assertEqual(train_len + val_len + test_len, total_rows)
        self.assertNotIn("target_aqi_10h", splits["X_train"].columns)

    def test_save_processed_splits(self):
        df_target = create_forecast_target(self.df, horizon_hours=10, target_col="aqi")
        out_dir = "data/processed/test_splits"

        saved_paths = save_processed_splits(df_target, val_size=0.15, test_size=0.15, output_dir=out_dir)
        
        self.assertTrue(saved_paths["train"].exists())
        self.assertTrue(saved_paths["val"].exists())
        self.assertTrue(saved_paths["test"].exists())

        # Cleanup
        for path in saved_paths.values():
            if path.exists():
                path.unlink()
        Path(out_dir).rmdir()


if __name__ == "__main__":
    unittest.main()
