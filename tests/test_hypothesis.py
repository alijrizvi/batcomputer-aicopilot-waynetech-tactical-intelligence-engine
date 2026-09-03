# ==============================================================================
# TEST SUITE: HYPOTHESIS & RISK ENGINE (TDD RED PHASE)
# Target Module: src.models.hypothesis
# ==============================================================================

import pytest
import numpy as np
import pandas as pd

# Explicitly import interface function before implementation exists!
from src.models.hypothesis import run_district_hypothesis_suite


@pytest.fixture
def sample_experiment_data() -> pd.DataFrame:
    """Fixture simulating a 2-District A/B test with 3 pricing tiers."""
    np.random.seed(42)
    n = 100
    
    # District 1: Control (Standard Demand)
    control_demand = np.random.normal(loc=5.0, scale=1.2, size=n)
    # District 2: Treatment (Policy Intervention Lift)
    treatment_demand = np.random.normal(loc=6.2, scale=1.4, size=n)
    
    df = pd.DataFrame({
        "district_id": ["District_1"] * n + ["District_2"] * n,
        "kit_demand": np.concatenate([control_demand, treatment_demand]),
        "price_tier": np.random.choice(["Tier_Base", "Tier_Minus_5", "Tier_Minus_10"], size=2*n)
    })
    return df


def test_run_district_hypothesis_suite_structure(sample_experiment_data: pd.DataFrame):
    """Verify hypothesis suite returns valid statistical results & Monte Carlo bounds."""
    results = run_district_hypothesis_suite(
        df=sample_experiment_data,
        n_simulations=500
    )
    
    # 1. Top-Level Keys
    assert "ab_test" in results
    assert "anova_test" in results
    assert "risk_simulation" in results
    
    # 2. Welch's t-test Validation
    ab = results["ab_test"]
    assert "t_stat" in ab
    assert "p_value" in ab
    assert "cohens_d" in ab
    assert ab["p_value"] <= 0.05  # Significant difference built into fixture
    
    # 3. One-Way ANOVA Validation
    anova = results["anova_test"]
    assert "f_stat" in anova
    assert "p_value" in anova
    
    # 4. Monte Carlo Risk Bounds Validation
    risk = results["risk_simulation"]
    assert "mean_demand" in risk
    assert "ci_lower_95" in risk
    assert "ci_upper_95" in risk
    assert "stockout_probability" in risk
    assert 0.0 <= risk["stockout_probability"] <= 1.0