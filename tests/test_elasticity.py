# OLS Elasticity Unit Test
# This test verifies that the log-log regression function correctly estimates the elasticity coefficient (beta1 = -1.5) 
# from log-transformed inputs.

import numpy as np
import pandas as pd
from src.models.elasticity import fit_log_log_elasticity


def test_fit_log_log_elasticity_known_coefficient():
    """Verify log-log regression recovers known slope beta_1 = -1.5."""
    # Arrange: Construct deterministic dataset: ln(y) = 5.0 - 1.5 * ln(x)
    np.random.seed(42)
    costs = np.random.uniform(10, 100, 100)
    demands = np.exp(5.0 - 1.5 * np.log(costs) + np.random.normal(0, 0.01, 100))
    df = pd.DataFrame({"unit_cost": costs, "kit_demand": demands})

    # Act
    results = fit_log_log_elasticity(
        df, cost_col="unit_cost", demand_col="kit_demand"
    )

    # Assert: Verify returned metrics
    assert "elasticity_coef" in results
    assert np.isclose(results["elasticity_coef"], -1.5, atol=0.1)
    assert results["r_squared"] > 0.95