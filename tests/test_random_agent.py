import os
import sys

import numpy as np
import pandas as pd

# Adjust path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.hybrid_env import HybridTradingEnv


def test_environment_mechanics():
    print("Generating isolated test dataset...")
    dates = pd.date_range(start="2026-06-01", periods=1000, freq="15min")
    highs = np.random.uniform(2010, 2015, 1000)
    lows = np.random.uniform(2000, 2005, 1000)
    df = pd.DataFrame(
        {
            "time": dates,
            "env_open": np.random.uniform(lows, highs, 1000),
            "env_high": highs,
            "env_low": lows,
            "env_close": np.random.uniform(lows, highs, 1000),
            "dist_ema_50": np.random.uniform(-0.01, 0.01, 1000),
        }
    )

    env = HybridTradingEnv(
        df=df,
        session="ALL",
        window_size=30,
        oracle_path="dummy_oracle.pth",
        scaler_path="dummy_scaler.npz",
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
