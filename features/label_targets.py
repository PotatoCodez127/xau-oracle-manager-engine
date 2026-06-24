import os

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR) based on hidden env_ prices."""
    high_low = df["env_high"] - df["env_low"]
    high_close = np.abs(df["env_high"] - df["env_close"].shift())
    low_close = np.abs(df["env_low"] - df["env_close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def generate_labels(
    df: pd.DataFrame, max_hold: int = 32, rr_ratio: float = 2.0, spread: float = 0.15
) -> pd.DataFrame:
    """
    Sweeps the dataset and labels future price action.
    0 = Hold/Noise, 1 = Long 2R Hit, 2 = Short 2R Hit.
    """
    print(
        f"Generating forward-looking labels (Max Hold: {max_hold}, RR: {rr_ratio})..."
    )
    df = df.copy()

    # Only calculate ATR if it hasn't been provided (prevents overwriting test mocks)
    if "env_atr" not in df.columns:
        df["env_atr"] = calculate_atr(df)

    # Extract to fast numpy arrays
    close_p = df["env_close"].values
    high_p = df["env_high"].values
    low_p = df["env_low"].values
    atr_v = df["env_atr"].values

    targets = np.zeros(len(df), dtype=int)

    # Iterate through all rows (except the very end where we can't look ahead)
    for i in range(len(df) - max_hold):
        if np.isnan(atr_v[i]):
            continue

        entry_price = close_p[i]
        atr = atr_v[i]

        # Avoid zero-volatility math errors
        if atr < 0.1:
            atr = 0.5

        # Define strict structural brackets
        long_tp = entry_price + (atr * rr_ratio) + spread
        long_sl = entry_price - atr - spread

        short_tp = entry_price - (atr * rr_ratio) - spread
        short_sl = entry_price + atr + spread

        long_valid = True
        short_valid = True
        target = 0

        # Look ahead into the future up to the max holding period
        for j in range(1, max_hold + 1):
            future_idx = i + j
            f_high = high_p[future_idx]
            f_low = low_p[future_idx]

            # Evaluate Long Edge
            if long_valid:
                if f_low <= long_sl:
                    long_valid = False
                elif f_high >= long_tp:
                    target = 1
                    break

            # Evaluate Short Edge
            if short_valid:
                if f_high >= short_sl:
                    short_valid = False
                elif f_low <= short_tp:
                    target = 2
                    break

            # If both directions stopped out, break early to save compute
            if not long_valid and not short_valid:
                break

        targets[i] = target

    df["target"] = targets

    # Drop rows where we couldn't calculate ATR or look ahead
    df_clean = df.dropna().iloc[:-max_hold].copy()

    print(
        f"Labeling complete. Longs: {np.sum(df_clean['target'] == 1)} |"
        f" Shorts: {np.sum(df_clean['target'] == 2)} | "
        f"Hold/Noise: {np.sum(df_clean['target'] == 0)}"
    )
    return df_clean


if __name__ == "__main__":
    input_path = "../data/processed/master_features_15m.csv"
    output_path = "../data/processed/labeled_features_15m.csv"

    if os.path.exists(input_path):
        print("Loading master features...")
        df = pd.read_csv(
            input_path,
            index_col=(
                "time" if "time" in pd.read_csv(input_path, nrows=0).columns else 0
            ),
        )

        df_labeled = generate_labels(df, max_hold=32, rr_ratio=2.0)

        df_labeled.to_csv(output_path)
        print(f"Labeled dataset saved to {output_path}")
    else:
        print(f"File not found: {input_path}")
