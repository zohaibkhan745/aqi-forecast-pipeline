import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.training.baseline_models import (
    prepare_imputed_features,
    train_ridge_regression,
    train_random_forest,
    evaluate_model,
)


class TestBaselineModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        # Synthetic feature matrix
        self.X_train = pd.DataFrame({
            "aqi": np.linspace(50, 200, n) + np.random.normal(0, 5, n),
            "pm25": np.linspace(25, 100, n) + np.random.normal(0, 3, n),
            "wind_speed": np.random.uniform(1, 8, n),
            "temperature": np.random.uniform(15, 35, n),
        })
        self.y_train = pd.Series(self.X_train["aqi"] * 1.05 + np.random.normal(0, 10, n), name="target_aqi_3d")

        # Validation set
        n_val = 30
        self.X_val = pd.DataFrame({
            "aqi": np.linspace(60, 210, n_val) + np.random.normal(0, 5, n_val),
            "pm25": np.linspace(30, 105, n_val) + np.random.normal(0, 3, n_val),
            "wind_speed": np.random.uniform(1, 8, n_val),
            "temperature": np.random.uniform(15, 35, n_val),
        })
        self.y_val = pd.Series(self.X_val["aqi"] * 1.05 + np.random.normal(0, 10, n_val), name="target_aqi_3d")

        # Test set
        n_test = 30
        self.X_test = pd.DataFrame({
            "aqi": np.linspace(70, 220, n_test) + np.random.normal(0, 5, n_test),
            "pm25": np.linspace(35, 110, n_test) + np.random.normal(0, 3, n_test),
            "wind_speed": np.random.uniform(1, 8, n_test),
            "temperature": np.random.uniform(15, 35, n_test),
        })
        self.y_test = pd.Series(self.X_test["aqi"] * 1.05 + np.random.normal(0, 10, n_test), name="target_aqi_3d")

    def test_prepare_imputed_features(self):
        # Insert NaNs into train and test
        X_tr = self.X_train.copy()
        X_tr.loc[0, "pm25"] = np.nan
        X_te = self.X_test.copy()
        X_te.loc[0, "wind_speed"] = np.nan

        X_tr_imp, X_va_imp, X_te_imp, imputer = prepare_imputed_features(X_tr, self.X_val, X_te)

        self.assertFalse(X_tr_imp.isnull().any().any())
        self.assertFalse(X_va_imp.isnull().any().any())
        self.assertFalse(X_te_imp.isnull().any().any())
        self.assertEqual(list(X_tr_imp.columns), list(self.X_train.columns))

    def test_train_ridge_regression(self):
        alphas = [0.1, 1.0, 10.0, 100.0]
        model, grid_results = train_ridge_regression(
            self.X_train, self.y_train, self.X_val, self.y_val, alphas=alphas
        )

        self.assertIsNotNone(model)
        self.assertEqual(len(grid_results), 4)
        for a in alphas:
            self.assertIn(a, grid_results)
            self.assertIsInstance(grid_results[a], float)

        # Check prediction shape
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_train_random_forest(self):
        n_estimators = [10, 20]  # small for fast test
        max_depths = [5, None]
        model, grid_results, importances = train_random_forest(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_estimators_list=n_estimators, max_depth_list=max_depths
        )

        self.assertIsNotNone(model)
        self.assertEqual(len(grid_results), 4)
        self.assertEqual(len(importances), len(self.X_train.columns))

        # Check that importances are sorted descending
        imp_values = list(importances.values())
        self.assertTrue(all(imp_values[i] >= imp_values[i+1] for i in range(len(imp_values)-1)))

    def test_evaluate_model(self):
        model, _, _ = train_random_forest(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_estimators_list=[10], max_depth_list=[5]
        )

        eval_res = evaluate_model(model, self.X_test, self.y_test)

        self.assertIn("model_metrics", eval_res)
        self.assertIn("persistence_metrics", eval_res)
        self.assertIn("rmse", eval_res["model_metrics"])
        self.assertIn("mae", eval_res["model_metrics"])
        self.assertIn("r2", eval_res["model_metrics"])
        self.assertIn("improvement_rmse", eval_res)
        self.assertIn("beats_persistence", eval_res)
        self.assertIsInstance(eval_res["beats_persistence"], (bool, np.bool_))


if __name__ == "__main__":
    unittest.main()
