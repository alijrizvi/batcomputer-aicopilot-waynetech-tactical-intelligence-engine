# Mathematical foundation and Feature Engineering pipeline for synthetic data generation in Gotham District 1.

"""src/data/generator.py

Synthetic Data Generation Pipeline for Gotham District 1.
Encodes true log-log demand elasticity and builds time-series features.
"""

from typing import Optional
import numpy as np
import pandas as pd


def generate_gotham_data(
    days: int = 365,
    seed: Optional[int] = 42,
    base_demand: float = 5.5,
    true_elasticity: float = -1.25,
) -> pd.DataFrame:
    """Generates synthetic daily time-series data for Medical Kit Demand.

    Formula: ln(Demand) = base_demand + true_elasticity * ln(Cost) + Threat_Impact + Noise

    Args:
        days (int): Exact number of usable daily time steps to return.
        seed (Optional[int]): Random seed for reproducibility.
        base_demand (float): Intercept (beta_0) in log space.
        true_elasticity (float): True underlying elasticity coefficient (beta_1).

    Returns:
        pd.DataFrame: Engineered daily time-series dataset with exactly `days` rows.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Warm-up Buffer: Generate extra days to absorb 7-day rolling window NaNs
    warmup_days = 7
    total_days = days + warmup_days

    # 2. Base Temporal Index
    dates = pd.date_range(start="2026-01-01", periods=total_days, freq="D")

    # 3. Exogenous Features (Cost & Threat Score)
    unit_cost = np.random.uniform(20.0, 80.0, total_days)
    threat_score = np.random.uniform(10.0, 95.0, total_days)

    # 4. Log-Log Mechanics with Noise & Exogenous Threat Uplift
    threat_effect = 0.005 * threat_score
    noise = np.random.normal(0, 0.05, total_days)

    log_demand = base_demand + (true_elasticity * np.log(unit_cost)) + threat_effect + noise
    kit_demand = np.exp(log_demand)

    # Assemble Base DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "unit_cost": unit_cost,
            "threat_score": threat_score,
            "kit_demand": kit_demand,
        }
    )

    # 5. Time-Series Feature Engineering for ML (XGBoost)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Rolling Window & Lag Features
    df["demand_lag_1d"] = df["kit_demand"].shift(1)
    df["demand_lag_7d"] = df["kit_demand"].shift(7)
    df["demand_roll_mean_7d"] = df["kit_demand"].shift(1).rolling(window=7).mean()
    df["demand_roll_std_7d"] = df["kit_demand"].shift(1).rolling(window=7).std()

    # Drop warm-up NaN rows and reset index to guarantee exactly `days` rows
    df = df.dropna().reset_index(drop=True)

    return df