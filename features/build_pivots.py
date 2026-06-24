import pandas as pd


def calculate_daily_levels(df_m1: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Daily EQ and Standard Pivots from M1 data.
    Assumes df_m1 has a DatetimeIndex and ['open', 'high', 'low', 'close']
    """
    # Resample to Daily to get the day's HLOC
    daily = (
        df_m1.resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    # Calculate Previous Day's metrics (Shift by 1 to prevent lookahead bias!)
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)

    # 1. Daily EQ (Middle of previous daily candle)
    daily["daily_eq"] = (daily["prev_high"] + daily["prev_low"]) / 2.0

    # 2. Standard Pivots
    daily["pivot"] = (
        daily["prev_high"] + daily["prev_low"] + daily["prev_close"]
    ) / 3.0
    daily["R1"] = (2 * daily["pivot"]) - daily["prev_low"]
    daily["S1"] = (2 * daily["pivot"]) - daily["prev_high"]

    # Keep only the calculated levels
    daily_levels = daily[["daily_eq", "pivot", "R1", "S1"]].dropna()

    # Merge back to the M1 timeframe (forward fill so every minute knows the current day's levels)
    # We use merge_asof to safely align timestamps
    df_m1 = df_m1.sort_index()
    daily_levels = daily_levels.sort_index()

    merged = pd.merge_asof(
        df_m1, daily_levels, left_index=True, right_index=True, direction="backward"
    )

    return merged
