import numpy as np
import pandas as pd


def calculate_wick_zones(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Identifies Wick-to-Body S&R zones for a specific timeframe.
    window: the number of candles needed to confirm a swing high/low.
    """
    df = df.copy()

    # Calculate rolling maximums/minimums.
    # We use a trailing window and shift to avoid lookahead bias.
    # A peak is confirmed if the high from 'window' periods ago is the max of the rolling window.
    df["rolling_max"] = df["high"].rolling(window=window * 2 + 1, center=False).max()
    df["rolling_min"] = df["low"].rolling(window=window * 2 + 1, center=False).min()

    # Identify Peaks and Troughs
    # The peak actually happened 'window' periods ago.
    is_swing_high = df["high"].shift(window) == df["rolling_max"]
    is_swing_low = df["low"].shift(window) == df["rolling_min"]

    # Initialize Zone Columns
    df["res_zone_top"] = np.nan
    df["res_zone_bottom"] = np.nan
    df["sup_zone_top"] = np.nan
    df["sup_zone_bottom"] = np.nan

    # Extract Wick-to-Body dimensions for Swing Highs (Resistance)
    # Top of zone = tip of wick (High)
    # Bottom of zone = highest body point (Max of Open/Close)
    res_mask = is_swing_high
    df.loc[res_mask, "res_zone_top"] = df["high"].shift(window)
    df.loc[res_mask, "res_zone_bottom"] = (
        df[["open", "close"]].shift(window).max(axis=1)
    )

    # Extract Wick-to-Body dimensions for Swing Lows (Support)
    # Bottom of zone = tip of wick (Low)
    # Top of zone = lowest body point (Min of Open/Close)
    sup_mask = is_swing_low
    df.loc[sup_mask, "sup_zone_bottom"] = df["low"].shift(window)
    df.loc[sup_mask, "sup_zone_top"] = df[["open", "close"]].shift(window).min(axis=1)

    # Forward fill the active zones until a new one is formed
    df["res_zone_top"] = df["res_zone_top"].ffill()
    df["res_zone_bottom"] = df["res_zone_bottom"].ffill()
    df["sup_zone_top"] = df["sup_zone_top"].ffill()
    df["sup_zone_bottom"] = df["sup_zone_bottom"].ffill()

    return df


def add_15m_ema(df_15m: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    """
    Calculates the exponential moving average exclusively for the 15m timeframe.
    """
    df_15m[f"ema_{period}"] = df_15m["close"].ewm(span=period, adjust=False).mean()
    return df_15m
