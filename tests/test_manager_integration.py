import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.hybrid_env import HybridTradingEnv
from models.oracle_lstm import OracleLSTM


@pytest.fixture
def mock_integration_setup(tmp_path):
    """Sets up dummy files representing the output of the Supervised Oracle pipeline."""
    data_length = 100
    dates = pd.date_range(start="2026-06-01", periods=data_length, freq="15min")
    highs = np.random.uniform(2010, 2015, data_length)
    lows = np.random.uniform(2000, 2005, data_length)
    data = {
        "time": dates,
        "env_open": np.random.uniform(lows, highs, data_length),
        "env_high": highs,
        "env_low": lows,
        "env_close": np.random.uniform(lows, highs, data_length),
        "dist_ema_50": np.random.uniform(-0.02, 0.02, data_length),
    }
    df = pd.DataFrame(data)

    # Generate dummy Z-score scaler
    scaler_path = tmp_path / "dummy_scaler.npz"
    np.savez(scaler_path, mean=np.array([0.0]), std=np.array([1.0]))

    # Generate dummy Oracle .pth weights
    oracle_path = tmp_path / "dummy_oracle.pth"
    model = OracleLSTM(input_dim=1)
    torch.save(model.state_dict(), oracle_path)

    return df, str(oracle_path), str(scaler_path)


def test_sac_manager_step_and_learn(mock_integration_setup, tmp_path):
    df, oracle_path, scaler_path = mock_integration_setup

    def make_env():
        return HybridTradingEnv(
            df, window_size=5, oracle_path=oracle_path, scaler_path=scaler_path
        )

    vec_env = DummyVecEnv([make_env])

    # Initialize Soft Actor-Critic agent
    # learning_starts=10 forces the agent to begin gradient descent almost immediately
    model = SAC("MlpPolicy", vec_env, verbose=0, learning_starts=10, batch_size=8)

    try:
        # Run a micro-training loop to ensure PyTorch and SB3 hand-off succeeds
        model.learn(total_timesteps=20)
    except Exception as e:
        pytest.fail(f"SAC Manager failed during the learn step: {e}")

    # Ensure the architecture allows for clean weight serialization
    save_path = tmp_path / "test_sac.zip"
    model.save(save_path)
    assert os.path.exists(save_path), "Manager SAC model failed to save to disk."
