# OLS Elasticity Logic

from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_log_log_elasticity(
    df: pd.DataFrame, cost_col: str = "unit_cost", demand_col: str = "kit_demand"
) -> Dict[str, Any]:
    """Fits an OLS Log-Log regression model to estimate price elasticity of demand.

    Formula: ln(Demand) = beta_0 + beta_1 * ln(Cost) + epsilon
    """
    # Log transformations
    log_x = np.log(df[cost_col])
    log_y = np.log(df[demand_col])

    # Add constant for OLS intercept
    X = sm.add_constant(log_x)

    # Fit Ordinary Least Squares model
    model = sm.OLS(log_y, X).fit()

    return {
        "elasticity_coef": float(model.params.iloc[1]),
        "intercept": float(model.params.iloc[0]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues.iloc[1]),
    }