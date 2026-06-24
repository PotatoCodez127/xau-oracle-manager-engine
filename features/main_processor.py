import pandas as pd
from build_pivots import calculate_daily_levels
from build_zones import add_15m_ema, calculate_wick_zones


def resample_hloc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples M1 data to higher timeframes safely."""
    return (
        df.resample(timeframe)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def convert_to_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw prices to stationary percentage distances for the AI,
    but preserves raw prices under an 'env_' prefix strictly for PnL math.
    """
    print("Converting raw prices to stationary percentage distances...")
    stationary_df = pd.DataFrame(index=df.index)

    # --- ENVIRONMENT VARIABLES (HIDDEN FROM AI) ---
    # The environment needs these to calculate actual dollar PnL
    stationary_df["env_open"] = df["open"]
    stationary_df["env_high"] = df["high"]
    stationary_df["env_low"] = df["low"]
    stationary_df["env_close"] = df["close"]

    # --- AI OBSERVATION VARIABLES (STATIONARY) ---
    stationary_df["close_ret"] = df["close"].pct_change()
    stationary_df["high_ret"] = (df["high"] - df["open"]) / df["open"]
    stationary_df["low_ret"] = (df["low"] - df["open"]) / df["open"]

    exclude_cols = ["open", "high", "low", "close", "dxy_pct_change_15m"]
    price_level_cols = [c for c in df.columns if c not in exclude_cols]

    for col in price_level_cols:
        # Distance metric: (Level - Current Price) / Current Price
        stationary_df[f"dist_{col}"] = (df[col] - df["close"]) / df["close"]

    if "dxy_pct_change_15m" in df.columns:
        stationary_df["dxy_pct_change_15m"] = df["dxy_pct_change_15m"]

    return stationary_df.dropna()


def build_master_dataset(xau_m1_path: str, dxy_m1_path: str) -> pd.DataFrame:
    """Orchestrates the creation of the ML Feature state."""
    print("Loading raw M1 data...")
    df_m1 = pd.read_csv(xau_m1_path, index_col="time", parse_dates=True).sort_index()

    df_m1 = calculate_daily_levels(df_m1)

    print("Building 15m Execution Base & 50 EMA...")
    df_15m = resample_hloc(df_m1, "15min")
    df_15m = add_15m_ema(df_15m, period=50)
    df_15m = calculate_wick_zones(df_15m, window=5)

    df_15m = df_15m.rename(
        columns=lambda x: f"{x}_15m" if "zone" in x or "rolling" in x else x
    )

    print("Mapping 30m and 4H structural zones...")
    df_30m = resample_hloc(df_m1, "30min")
    df_30m = calculate_wick_zones(df_30m, window=5)
    df_30m = df_30m.rename(columns=lambda x: f"{x}_30m" if "zone" in x else x)

    df_4h = resample_hloc(df_m1, "4h")
    df_4h = calculate_wick_zones(df_4h, window=5)
    df_4h = df_4h.rename(columns=lambda x: f"{x}_4h" if "zone" in x else x)

    print("Processing DXY correlation metrics...")
    try:
        dxy_m1 = pd.read_csv(
            dxy_m1_path, index_col="time", parse_dates=True
        ).sort_index()
        dxy_15m = resample_hloc(dxy_m1, "15min")
        dxy_15m["dxy_pct_change_15m"] = dxy_15m["close"].pct_change()
    except FileNotFoundError:
        print("Warning: DXY file not found. Generating zeros for testing purposes.")
        dxy_15m = pd.DataFrame(index=df_15m.index)
        dxy_15m["dxy_pct_change_15m"] = 0.0

    print("Stitching timelines together...")
    daily_cols = (
        df_m1[["daily_eq", "pivot", "R1", "S1"]].resample("15min").last().ffill()
    )
    master = pd.merge_asof(df_15m, daily_cols, left_index=True, right_index=True)

    htf_cols_30m = [c for c in df_30m.columns if "zone" in c]
    master = pd.merge_asof(
        master, df_30m[htf_cols_30m], left_index=True, right_index=True
    )

    htf_cols_4h = [c for c in df_4h.columns if "zone" in c]
    master = pd.merge_asof(
        master, df_4h[htf_cols_4h], left_index=True, right_index=True
    )

    master = pd.merge_asof(
        master, dxy_15m[["dxy_pct_change_15m"]], left_index=True, right_index=True
    )
    master["dxy_pct_change_15m"] = master["dxy_pct_change_15m"].fillna(0.0)
    master = master.dropna()

    # CRITICAL INJECTION: Convert raw prices to stationary features before returning
    master = convert_to_stationary(master)

    print("Master Feature Dataset Complete.")
    return master


if __name__ == "__main__":
    print("Starting Master Processor for TRAINING...")
    # Point to the 4-year synthetic data!
    df_master = build_master_dataset(
        "../data/processed/xauusd_m1_synthetic_4yrs.csv",
        "../data/processed/dxy_m1_clean.csv",
    )
    # Save the output to the master features file used by lstm_ppo.py
    df_master.to_csv("../data/processed/master_features_15m.csv")
    print("Successfully saved master_features_15m.csv")
