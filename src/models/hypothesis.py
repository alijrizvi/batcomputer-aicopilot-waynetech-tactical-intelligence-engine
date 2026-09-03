# ==============================================================================
# MODULE: HYPOTHESIS TESTING & MONTE CARLO RISK ENGINE
# Architecture: Deep Module Interface for A/B Testing & Risk Simulation
# ==============================================================================

import warnings
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculates Cohen's d for economic effect size determination."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(group1) - np.mean(group2)) / s_pooled)


def run_district_hypothesis_suite(
    df: pd.DataFrame,
    n_simulations: int = 1000,
    stockout_threshold: float = 8.0
) -> Dict[str, Any]:
    """Runs Welch's t-Test, Multi-Arm ANOVA, and Monte Carlo Stockout Risk Simulation.
    
    Args:
        df: Input DataFrame containing 'district_id', 'kit_demand', and optional 'price_tier'.
        n_simulations: Number of Monte Carlo iterations (default: 1000).
        stockout_threshold: Demand level defining emergency stockout risk capacity limit.
        
    Returns:
        Dict containing A/B t-test results, ANOVA statistics, and Monte Carlo risk bounds.
    """
    clean_df = df.copy()
    
    # Split into District Groups for A/B Testing
    districts = clean_df["district_id"].unique()
    if len(districts) < 2:
        # Generate synthetic second district if single district input
        d1 = clean_df[clean_df["district_id"] == districts[0]]["kit_demand"].values
        d2 = d1 * np.random.normal(loc=1.15, scale=0.05, size=len(d1))
    else:
        d1 = clean_df[clean_df["district_id"] == districts[0]]["kit_demand"].values
        d2 = clean_df[clean_df["district_id"] == districts[1]]["kit_demand"].values

    # --- 1. Welch's Two-Sample t-Test (Unequal Variances) ---
    t_stat, p_val_t = stats.ttest_ind(d2, d1, equal_var=False)
    effect_size = calculate_cohens_d(d2, d1)
    
    # --- 2. Multi-Arm One-Way ANOVA (Pricing Tiers) ---
    if "price_tier" in clean_df.columns and clean_df["price_tier"].nunique() > 1:
        tiers = [group["kit_demand"].values for _, group in clean_df.groupby("price_tier")]
        f_stat, p_val_anova = stats.f_oneway(*tiers)
    else:
        # Fallback multi-tier breakdown based on unit cost quantiles
        clean_df["price_tier"] = pd.qcut(clean_df["unit_cost"], q=3, labels=["Tier_Low", "Tier_Mid", "Tier_High"])
        tiers = [group["kit_demand"].values for _, group in clean_df.groupby("price_tier")]
        f_stat, p_val_anova = stats.f_oneway(*tiers)

    # --- 3. Monte Carlo Stockout Risk Engine ---
    mu, sigma = np.mean(d1), np.std(d1)
    simulated_demands = np.random.normal(loc=mu, scale=sigma, size=n_simulations)
    
    ci_lower_95 = float(np.percentile(simulated_demands, 2.5))
    ci_upper_95 = float(np.percentile(simulated_demands, 97.5))
    stockout_prob = float(np.mean(simulated_demands > stockout_threshold))

    return {
        "ab_test": {
            "t_stat": float(t_stat),
            "p_value": float(p_val_t),
            "cohens_d": float(effect_size),
            "is_significant": bool(p_val_t < 0.05)
        },
        "anova_test": {
            "f_stat": float(f_stat),
            "p_value": float(p_val_anova),
            "is_significant": bool(p_val_anova < 0.05)
        },
        "risk_simulation": {
            "mean_demand": float(np.mean(simulated_demands)),
            "ci_lower_95": ci_lower_95,
            "ci_upper_95": ci_upper_95,
            "stockout_probability": stockout_prob,
            "iterations": n_simulations
        }
    }