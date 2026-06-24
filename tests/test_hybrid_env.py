import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.hybrid_env import HybridTradingEnv


@pytest.fixture
def mock_feature_data():
    """Generates 50 rows of dummy market data to test environment boundaries."""
    dates = pd.date_range(start="2026-06-01", periods=50, freq="15min")
    data = {
        "time": dates,
        "env_high": np.random.uniform(2010, 2015, 50),
        "env_low": np.random.uniform(2000, 2005, 50),
        "env_close": np.random.uniform(2005, 2010, 50),
        "dist_ema_50": np.random.uniform(-0.01, 0.01, 50),  # Mock stationary feature
    }
    return pd.DataFrame(data)


def test_environment_scaler_persistence(mock_feature_data, tmp_path):
    scaler_file = tmp_path / "mock_scaler.npz"

    # 1. Initialize env - it should create the scaler file
    env1 = HybridTradingEnv(
        mock_feature_data, window_size=5, scaler_path=str(scaler_file)
    )
    assert os.path.exists(scaler_file), "Environment failed to generate scaler file."

    # Check that standard deviation is calculated correctly
    assert (
        len(env1.feature_std) == 1
    ), "Should only track 1 non-env feature (dist_ema_50)"

    # 2. Modify dummy data drastically to simulate a totally different test set
    mock_feature_data["dist_ema_50"] = mock_feature_data["dist_ema_50"] * 1000

    # 3. Initialize second env pointing to the same scaler file
    env2 = HybridTradingEnv(
        mock_feature_data, window_size=5, scaler_path=str(scaler_file)
    )

    # Assert that env2 loaded the original std dev, rather
    # than recalculating it on the modified test data
    np.testing.assert_array_equal(
        env1.feature_std,
        env2.feature_std,
        err_msg="Distribution Shift Leak! Environment recalculated stats instead of loading them.",
    )


def test_atr_lookahead_bias(mock_feature_data, tmp_path):
    scaler_file = tmp_path / "mock_scaler.npz"
    env = HybridTradingEnv(
        mock_feature_data, window_size=5, scaler_path=str(scaler_file)
    )

    atr_values = env.df["env_atr"].values

    # The first 14 rows should have uniform ATR due to the strict backfill limit
    assert (
        atr_values[0] == atr_values[13]
    ), "Initial ATR window was not correctly restricted."

    # Rows after the 14th should be dynamic and differ from the backfilled block
    assert (
        atr_values[0] != atr_values[25]
    ), "ATR calculation is statically leaking future global volatility."
