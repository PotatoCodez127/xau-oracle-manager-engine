import os
import sys

import numpy as np
import pandas as pd

# Adjust path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.hybrid_env import HybridTradingEnv


def test_environment_mechanics():
    print("Loading test dataset...")
    # Use your labeled features file for a quick test
    df = pd.read_csv("../data/processed/master_features_15m.csv")

    env = HybridTradingEnv(
        df=df,
        session="LONDON",
        window_size=30,
        oracle_path="../models/oracle_lstm.pth",
        scaler_path="../models/oracle_scaler.npz",
    )

    obs, info = env.reset()

    print("\n--- INITIATING RANDOM AGENT STRESS TEST ---")
    trades_executed = 0

    for i in range(1000):  # Test over 1000 candles
        # Generate a random action: [Direction, TP_Multiplier] between -1.0 and 1.0
        random_action = env.action_space.sample()

        # Force the action to be large enough to trigger a trade (> 0.25) periodically
        if i % 50 == 0:
            random_action[0] = 0.8 if np.random.rand() > 0.5 else -0.8

        previous_position = env.position
        obs, reward, terminated, truncated, info = env.step(random_action)

        # Detect Trade Entry
        if previous_position == 0 and env.position != 0:
            print(
                f"Step {info['step']}: 🟢 ENTERED {'LONG' if env.position == 1 else 'SHORT'} | "
                f"Entry: {env.entry_price:.2f} | SL: {env.sl_price:.2f} | TP: {env.tp_price:.2f}"
            )
            trades_executed += 1

        # Detect Trade Exit
        elif previous_position != 0 and env.position == 0:
            print(
                f"Step {info['step']}: 🔴 EXITED TRADE | Reward Triggered: {reward:.4f} | "
                f"Bars Held: {env.bars_held}\n"
            )

        if terminated or truncated:
            print("Environment terminated early (Margin Call).")
            break

    print(f"\nTest Complete. Total trades cycled: {trades_executed}")


if __name__ == "__main__":
    test_environment_mechanics()
