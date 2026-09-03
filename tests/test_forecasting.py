# ==============================================================================
# TEST SUITE: FORECASTING ENGINE (TDD RED PHASE)
# Target Module: src.models.forecasting
# ==============================================================================

import pytest
import numpy as np
import pandas as pd
from src.data.generator import generate_gotham_data

# Explicitly import the deep module interface function before it exists!
from src.models.forecasting import evaluate_forecasting_models


@pytest.fixture
def sample_gold_data() -> pd.DataFrame:
    """Fixture providing a valid 100-day dataset simulating Gold Parquet features."""
    df = generate_gotham_data(days=100, seed=42)
    # Engineer basic temporal features expected by Gold layer
    df["log_kit_demand"] = np.log(df["kit_demand"])
    df["log_unit_cost"] = np.log(df["unit_cost"])
    df["demand_lag_1d"] = df["kit_demand"].shift(1)
    df["demand_lag_7d"] = df["kit_demand"].shift(7)
    df["threat_roll_std_7d"] = df["threat_score"].shift(1).rolling(7).std()
    return df.dropna().reset_index(drop=True)


def test_evaluate_forecasting_models_structure(sample_gold_data: pd.DataFrame):
    """Verify that evaluate_forecasting_models returns expected dictionary structure."""
    results = evaluate_forecasting_models(
        df=sample_gold_data,
        n_splits=3,
        horizon=14
    )
    
    # 1. Assert top-level keys exist
    assert "metrics" in results
    assert "forecasts" in results
    assert "best_model" in results
    
    # 2. Assert metrics contain RMSE and MAPE for both models
    metrics = results["metrics"]
    assert "arima_rmse" in metrics
    assert "xgboost_rmse" in metrics
    assert "arima_mape" in metrics
    assert "xgboost_mape" in metrics
    
    # 3. Assert metrics are valid non-negative floats
    assert metrics["xgboost_rmse"] >= 0
    assert metrics["arima_rmse"] >= 0
    
    # 4. Assert forecast DataFrame contains required columns
    forecast_df = results["forecasts"]
    assert isinstance(forecast_df, pd.DataFrame)
    assert "arima_pred" in forecast_df.columns
    assert "xgb_pred" in forecast_df.columns
    assert "actual_demand" in forecast_df.columns