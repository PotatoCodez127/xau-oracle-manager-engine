import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the tests folder can import from the features folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.label_targets import generate_labels

@pytest.fixture
def mock_market_data():
    """
    Creates a 7-row synthetic dataframe testing three scenarios.
    """
    data = {
        'env_high':  [100.0, 105.0,  95.0, 110.0, 115.0,  90.0,  80.0],
        'env_low':   [100.0, 100.0,  90.0, 100.0, 105.0,  85.0,  70.0],
        'env_close': [100.0, 102.0,  92.0, 105.0, 110.0,  88.0,  75.0],
        'env_atr':   [  2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0]
    }
    return pd.DataFrame(data)

def test_generate_labels_logic(mock_market_data):
    # Set max_hold to 4 so the loop (len - max_hold) can evaluate index 0, 1, and 2.
    df_labeled = generate_labels(mock_market_data, max_hold=4, rr_ratio=2.0, spread=0.0)
    
    targets = df_labeled['target'].values
    
    # ROW 0 SCENARIO (Index 0): Long Win
    # Entry = 100.0, Long TP = 104.0, Long SL = 98.0
    # Next candle High hits 105.0 (TP Hit) before Low hits 98.0.
    assert targets[0] == 1, f"Expected Long Win (1) at index 0, got {targets[0]}"

    # ROW 1 SCENARIO (Index 1): Short Win
    # Entry = 102.0, Short TP = 98.0, Short SL = 104.0
    # Next candle Low hits 90.0 (TP Hit) before High hits 104.0.
    assert targets[1] == 2, f"Expected Short Win (2) at index 1, got {targets[1]}"
    
    # ROW 2 SCENARIO (Index 2): Long Win
    # Entry = 92.0, Long TP = 96.0, Long SL = 90.0
    # Next candle High hits 110.0 (TP Hit) before Low hits 90.0.
    assert targets[2] == 1, f"Expected Long Win (1) at index 2, got {targets[2]}"