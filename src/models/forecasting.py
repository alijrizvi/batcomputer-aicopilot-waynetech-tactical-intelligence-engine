# ==============================================================================
# MODULE: FORECASTING ENGINE (ARIMA vs. XGBoost)
# Architecture: Deep Module Interface for Time-Series Evaluation
# Methodology: Rolling Step Expanding Window Cross-Validation
# ==============================================================================

import warnings
from typing import Any, Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")


def evaluate_forecasting_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon: int = 30,
    step_shift: int = 15
) -> Dict[str, Any]:
    """Evaluates ARIMA vs. XGBoost using an Expanding Window Time-Series Split.
    
    Args:
        df: Input DataFrame containing daily kit_demand and engineered Gold features.
        n_splits: Number of expanding window backtest folds.
        horizon: Forecast horizon (in days) per fold.
        step_shift: Number of days to slide the training start forward per fold.
        
    Returns:
        Dict containing average RMSE/MAPE metrics, forecast comparison DataFrame,
        and the winning model name.
    """
    total_rows = len(df)
    min_train_size = total_rows - (n_splits * step_shift + horizon)
    
    if min_train_size < 30:
        raise ValueError("Insufficient dataset size for requested splits, step shift, and horizon.")

    arima_rmse_list, xgb_rmse_list = [], []
    arima_mape_list, xgb_mape_list = [], []
    
    last_actuals, last_arima_preds, last_xgb_preds = [], [], []

    # Predictor Set for XGBoost
    feature_cols = [
        "unit_cost", "threat_score", "demand_lag_1d", 
        "demand_lag_7d", "threat_roll_std_7d", "is_weekend"
    ]
    
    clean_df = df.copy()
    for col in feature_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].bfill().ffill()

    # Expanding Window Split Loop with Staggered Calendar Shifts
    for fold in range(n_splits):
        train_end = min_train_size + (fold * step_shift)
        test_end = train_end + horizon
        
        train_data = clean_df.iloc[:train_end]
        test_data = clean_df.iloc[train_end:test_end]
        
        if len(test_data) < horizon:
            break

        y_train = train_data["kit_demand"]
        y_test = test_data["kit_demand"]

        # --- 1. ARIMA Baseline (Order: 1, 1, 1) ---
        try:
            arima_fit = ARIMA(y_train, order=(1, 1, 1)).fit()
            arima_pred = arima_fit.forecast(steps=horizon)
        except Exception:
            arima_pred = pd.Series([y_train.iloc[-1]] * horizon, index=y_test.index)

        # --- 2. XGBoost Machine Learning Engine ---
        X_train, y_tr = train_data[feature_cols], train_data["kit_demand"]
        X_test = test_data[feature_cols]

        xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        xgb_model.fit(X_train, y_tr)
        xgb_pred = xgb_model.predict(X_test)

        # Calculate fold metrics
        arima_rmse_list.append(np.sqrt(mean_squared_error(y_test, arima_pred)))
        xgb_rmse_list.append(np.sqrt(mean_squared_error(y_test, xgb_pred)))
        
        arima_mape_list.append(mean_absolute_percentage_error(y_test, arima_pred))
        xgb_mape_list.append(mean_absolute_percentage_error(y_test, xgb_pred))

        if fold == n_splits - 1:
            last_actuals = y_test.values
            last_arima_preds = arima_pred.values if isinstance(arima_pred, (np.ndarray, pd.Series)) else arima_pred
            last_xgb_preds = xgb_pred

    avg_arima_rmse = float(np.mean(arima_rmse_list))
    avg_xgb_rmse = float(np.mean(xgb_rmse_list))
    avg_arima_mape = float(np.mean(arima_mape_list))
    avg_xgb_mape = float(np.mean(xgb_mape_list))

    best_model = "XGBoost" if avg_xgb_rmse < avg_arima_rmse else "ARIMA"

    forecast_comparison_df = pd.DataFrame({
        "actual_demand": last_actuals,
        "arima_pred": last_arima_preds,
        "xgb_pred": last_xgb_preds
    })

    return {
        "metrics": {
            "arima_rmse": avg_arima_rmse,
            "xgboost_rmse": avg_xgb_rmse,
            "arima_mape": avg_arima_mape,
            "xgboost_mape": avg_xgb_mape
        },
        "forecasts": forecast_comparison_df,
        "best_model": best_model
    }