import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Engineers advanced features from a raw transaction dataframe.
    This version includes a robust strategy for handling missing values
    for the first transaction of a user.
    """
    # Ensure timestamp is in datetime format and sort for chronological processing
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["card_masked", "timestamp"]).reset_index(drop=True)

    # Time-based features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)

    # Time delta since the user's last transaction
    df["time_since_last_tx_seconds"] = (
        df.groupby("card_masked")["timestamp"].diff().dt.total_seconds()
    )

    # User's history with specific merchants and MCCs
    df["user_merchant_tx_count"] = df.groupby(["card_masked", "merchant"]).cumcount()
    df["user_has_used_merchant_before"] = (df["user_merchant_tx_count"] > 0).astype(int)

    df["user_mcc_tx_count"] = df.groupby(["card_masked", "mcc"]).cumcount()
    df["user_has_used_mcc_before"] = (df["user_mcc_tx_count"] > 0).astype(int)

    # Set timestamp as index to use time-based rolling windows efficiently
    df_dt_indexed = df.set_index("timestamp")

    # User's transaction counts over various rolling windows
    df["user_tx_count_1h"] = (
        df_dt_indexed.groupby("card_masked")["transaction_id"]
        .rolling("1h")
        .count()
        .reset_index(0, drop=True)
        .values
    )
    df["user_tx_count_6h"] = (
        df_dt_indexed.groupby("card_masked")["transaction_id"]
        .rolling("6h")
        .count()
        .reset_index(0, drop=True)
        .values
    )
    df["user_tx_count_24h"] = (
        df_dt_indexed.groupby("card_masked")["transaction_id"]
        .rolling("24h")
        .count()
        .reset_index(0, drop=True)
        .values
    )

    # User's unique merchant/location counts over various rolling windows
    df["user_unique_merchant_count_1h"] = (
        df_dt_indexed.groupby("card_masked")["merchant"]
        .rolling("1h")
        .apply(lambda x: x.nunique())
        .reset_index(0, drop=True)
        .values
    )
    df["user_unique_location_count_24h"] = (
        df_dt_indexed.groupby("card_masked")["location"]
        .rolling("24h")
        .apply(lambda x: x.nunique())
        .reset_index(0, drop=True)
        .values
    )

    # --- Start of Critical Change: Historical Amount Features ---

    # Calculate expanding average amount for each user
    expanding_avg = (
        df.groupby("card_masked")["amount"].expanding().mean().reset_index(0, drop=True)
    )

    # Use a temporary column to align the shifted average correctly
    df["temp_expanding_avg"] = expanding_avg
    df["user_avg_tx_amount_historical"] = df.groupby("card_masked")[
        "temp_expanding_avg"
    ].shift(1)
    df = df.drop(columns=["temp_expanding_avg"])

    # Fill historical average BEFORE calculating the ratio. Use 0 as a neutral start.
    df["user_avg_tx_amount_historical"] = df["user_avg_tx_amount_historical"].fillna(0)

    # Calculate the ratio of the current amount to the user's historical average
    df["amount_to_historical_avg_ratio"] = df["amount"] / (
        df["user_avg_tx_amount_historical"] + 1e-6  # Add epsilon to avoid division by zero
    )
    # Handle infinite values that occur on the first transaction (division by ~0)
    df['amount_to_historical_avg_ratio'] = df['amount_to_historical_avg_ratio'].replace([np.inf, -np.inf], np.nan)


    # --- End of Critical Change ---

    # Amount-based features
    df["is_round_amount"] = (df["amount"] % 1 == 0).astype(int)
    df["amount_cents"] = (df["amount"] * 100 % 100).astype(int)


    # --- Intelligent Filling of Missing Values ---
    
    # For features where "never happened before" is a distinct state, -1 is appropriate.
    # These are counts, time deltas, etc.
    cols_to_fill_negative_one = [
        "time_since_last_tx_seconds",
        "user_tx_count_1h",
        "user_tx_count_6h",
        "user_tx_count_24h",
        "user_unique_merchant_count_1h",
        "user_unique_location_count_24h",
    ]
    df[cols_to_fill_negative_one] = df[cols_to_fill_negative_one].fillna(-1)
    
    # For the ratio, the neutral value is 1.0 (current amount equals historical average).
    # This handles the NaN created from the first transaction's infinite ratio.
    df["amount_to_historical_avg_ratio"] = df["amount_to_historical_avg_ratio"].fillna(1.0)
    
    # Clean up intermediate helper columns that are not needed for the model
    df = df.drop(columns=["user_merchant_tx_count", "user_mcc_tx_count"])

    
    return df