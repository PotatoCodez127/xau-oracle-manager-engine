import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.oracle_lstm import TradingSequenceDataset, train_oracle


@pytest.fixture
def mock_labeled_data(tmp_path):
    """Creates a tiny synthetic labeled dataset to verify the training loop."""
    data_length = 50
    data = {
        "env_close": np.random.uniform(2000, 2050, data_length),
        "dist_ema_50": np.random.uniform(-0.02, 0.02, data_length),
        "dxy_pct_change_15m": np.random.uniform(-0.005, 0.005, data_length),
        "target": np.random.choice([0, 1, 2], size=data_length),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "mock_labeled_features.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_dataset_scaler_export(mock_labeled_data, tmp_path):
    scaler_path = tmp_path / "test_scaler.npz"

    # Initialize the dataset which should trigger the scaler export
    dataset = TradingSequenceDataset(
        mock_labeled_data, window_size=5, scaler_path=str(scaler_path)
    )

    assert os.path.exists(scaler_path), "Dataset failed to export the Z-score scaler."

    scaler_data = np.load(scaler_path)
    assert (
        "mean" in scaler_data and "std" in scaler_data
    ), "Scaler missing mean or std arrays."
    assert (
        len(scaler_data["mean"]) == 2
    ), "Scaler should only track the 2 non-env stationary features."


def test_oracle_training_loop(mock_labeled_data, tmp_path):
    scaler_path = tmp_path / "test_scaler.npz"
    original_cwd = os.getcwd()

    # Change working dir to tmp_path so the .pth saves there cleanly
    os.chdir(tmp_path)

    try:
        # Run a micro-training loop
        model_path = train_oracle(
            csv_path=mock_labeled_data,
            scaler_path=str(scaler_path),
            epochs=2,
            batch_size=8,
        )

        assert os.path.exists(
            model_path
        ), "Training loop completed but failed to save .pth weights."

        # Verify the weights can be loaded
        state_dict = torch.load(model_path, weights_only=True)
        assert len(state_dict) > 0, "Saved state dictionary is empty."

    finally:
        os.chdir(original_cwd)
