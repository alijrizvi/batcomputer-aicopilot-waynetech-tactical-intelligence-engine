# This test ensures that the synthetic data generator for Gotham City produces a clean, correctly shaped time-series dataset
#  with the expected structure and valid data types.

import pandas as pd
from src.data.generator import generate_gotham_data


def test_generate_gotham_data_structure():
    """Verify generated dataset shape, core columns, and data integrity."""
    # 1. Act: Generate exactly 365 days of engineered time-series data
    df = generate_gotham_data(days=365, seed=42)

    # 2. Assert: Validate shape
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 365

    # 3. Assert: Verify core required columns exist in the DataFrame
    required_columns = {"date", "unit_cost", "kit_demand"}
    assert required_columns.issubset(set(df.columns))

    # 4. Assert: Validate data positivity
    assert (df["unit_cost"] > 0).all()
    assert (df["kit_demand"] > 0).all()