import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import SAC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.oracle_lstm import OracleLSTM
from models.train_hybrid import run_forward_test


@pytest.fixture
def mock_forward_test_files(tmp_path):
    """Sets up a complete ecosystem of files for the final inference test."""
    # 1. Dummy Market Data
    data_length = 50
    data = {
        "time": pd.date_range(start="2026-07-01", periods=data_length, freq="15min"),
        "env_high": np.random.uniform(2010, 2015, data_length),
        "env_low": np.random.uniform(2000, 2005, data_length),
        "env_close": np.random.uniform(2005, 2010, data_length),
        "dist_ema_50": np.random.uniform(-0.02, 0.02, data_length),
    }
    df_path = tmp_path / "mock_test_data.csv"
    pd.DataFrame(data).to_csv(df_path, index=False)

    # 2. Dummy Oracle & Scaler
    scaler_path = tmp_path / "dummy_scaler.npz"
    np.savez(scaler_path, mean=np.array([0.0]), std=np.array([1.0]))

    oracle_path = tmp_path / "dummy_oracle.pth"
    model = OracleLSTM(input_dim=1)
    torch.save(model.state_dict(), oracle_path)

    # 3. Dummy SAC Model
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environment.hybrid_env import HybridTradingEnv

    env = DummyVecEnv(
        [
            lambda: HybridTradingEnv(
                pd.DataFrame(data),
                window_size=5,
                oracle_path=str(oracle_path),
                scaler_path=str(scaler_path),
            )
        ]
    )
    sac_model = SAC("MlpPolicy", env, learning_starts=0, batch_size=8)
    sac_path = tmp_path / "dummy_sac.zip"
    sac_model.save(sac_path)

    return str(df_path), str(oracle_path), str(scaler_path), str(sac_path)


def test_forward_test_execution(mock_forward_test_files, monkeypatch):
    test_data_path, oracle_path, scaler_path, sac_path = mock_forward_test_files

    # Disable matplotlib plotting during unit tests
    monkeypatch.setenv("HEADLESS_TESTING", "1")

    try:
        run_forward_test(
            model_path=sac_path,
            test_data_path=test_data_path,
            oracle_path=oracle_path,
            scaler_path=scaler_path,
            session="ALL",
        )
    except Exception as e:
        pytest.fail(f"Forward testing pipeline failed: {e}")
