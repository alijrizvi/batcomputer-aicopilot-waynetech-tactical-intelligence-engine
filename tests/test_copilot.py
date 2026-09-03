# ==============================================================================
# TEST SUITE: ANOMALY ENGINE & BATCOMPUTER COPILOT (TDD RED PHASE)
# Target Module: src.models.copilot
# ==============================================================================

import pytest
import numpy as np
import pandas as pd

# Explicitly import deep module function before implementation exists!
from src.models.copilot import detect_threat_anomalies, generate_batcomputer_tactical_brief


@pytest.fixture
def sample_system_payload():
    """Fixture providing mock system outputs from Slices 1, 2, and 3."""
    df = pd.DataFrame({
        "date": pd.date_range(start="2026-01-01", periods=30, freq="D"),
        "threat_score": [20]*28 + [95, 22],  # Day 29 is an anomaly
        "kit_demand": [5.0]*28 + [12.0, 5.1]
    })
    
    elasticity_dict = {"price_elasticity": -0.491, "r2_score": 0.82}
    forecast_dict = {"metrics": {"xgboost_rmse": 0.710, "arima_rmse": 1.841}, "best_model": "XGBoost"}
    hypothesis_dict = {
        "ab_test": {"t_stat": 4.28, "p_value": 0.0001, "cohens_d": 0.612, "is_significant": True},
        "risk_simulation": {"stockout_probability": 0.042, "ci_lower_95": 3.12, "ci_upper_95": 8.85}
    }
    
    return df, elasticity_dict, forecast_dict, hypothesis_dict


def test_detect_threat_anomalies(sample_system_payload):
    """Verify statistical Z-score anomaly detector flags severe spikes."""
    df, _, _, _ = sample_system_payload
    anomalies_df = detect_threat_anomalies(df, z_threshold=2.5)
    
    assert isinstance(anomalies_df, pd.DataFrame)
    assert "is_anomaly" in anomalies_df.columns
    assert "z_score" in anomalies_df.columns
    assert anomalies_df["is_anomaly"].sum() >= 1


def test_generate_batcomputer_tactical_brief_structure(sample_system_payload):
    """Verify copilot returns structured tactical intelligence payload."""
    df, elasticity_dict, forecast_dict, hypothesis_dict = sample_system_payload
    anomalies_df = detect_threat_anomalies(df)
    
    brief = generate_batcomputer_tactical_brief(
        elasticity_dict=elasticity_dict,
        forecast_dict=forecast_dict,
        hypothesis_dict=hypothesis_dict,
        anomalies_df=anomalies_df
    )
    
    assert "headline" in brief
    assert "executive_summary" in brief
    assert "tactical_recommendations" in brief
    assert "threat_status" in brief
    assert isinstance(brief["tactical_recommendations"], list)