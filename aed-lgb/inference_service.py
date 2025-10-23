import os
import json
import joblib
from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import deque

import torch
import torch.nn as nn
import lightgbm as lgb
import pandas as pd
import numpy as np

from ae import TransactionAutoencoderV2


MODEL_DIR = "saved_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TransactionInput:
    """
    Defines the structure for a single incoming transaction record.
    The fields must match the raw data columns used during training.
    """

    transaction_id: str
    timestamp: str
    amount: float
    merchant: str
    channel: str
    location: str
    city: str
    card_issuer: str
    card_masked: str


class FraudDetectionService:
    """
    A service to load all necessary artifacts and perform fraud detection inference.
    It manages user history internally for generating lagged features.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        print("--- Initializing Fraud Detection Service ---")
        self.model_dir = model_dir
        self.is_ready = False

        self.user_history = {}

        try:
            self._load_artifacts()
            self.is_ready = True
            print("--- Service Initialized Successfully ---")
        except Exception as e:
            print(f"Error initializing service: {e}")

    def _load_artifacts(self):
        """Loads all models, scalers, encoders, and metadata from disk."""
        print("Loading artifacts...")

        with open(os.path.join(self.model_dir, "model_metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.lgbm_booster = lgb.Booster(
            model_file=os.path.join(self.model_dir, "lightgbm_model.txt")
        )

        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
        self.label_encoders = joblib.load(
            os.path.join(self.model_dir, "label_encoders.joblib")
        )

        n_features = len(self.metadata["initial_features"])
        bottleneck_dim = self.metadata["bottleneck_dim"]
        self.autoencoder = TransactionAutoencoder(n_features, bottleneck_dim).to(DEVICE)
        self.autoencoder.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "autoencoder.pth"), map_location=DEVICE
            )
        )
        self.autoencoder.eval()

        self.max_history_len = self.metadata.get("ae_lags", 5)

        print("All artifacts loaded.")

    def _apply_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies loaded label encoders to the dataframe, handling unseen values."""
        for col, encoder in self.label_encoders.items():
            df[col] = (
                df[col]
                .astype(str)
                .map(lambda s: s if s in encoder.classes_ else "<unknown>")
            )

            # Add '<unknown>' to the encoder's classes if it's not there
            if "<unknown>" not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, "<unknown>")

            df[col] = encoder.transform(df[col])
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["card_masked", "timestamp"]).reset_index(drop=True)

        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(
            int
        )
        df["time_since_last_tx_seconds"] = (
            df.groupby("card_masked")["timestamp"].diff().dt.total_seconds()
        )

        df["user_merchant_tx_count"] = df.groupby(
            ["card_masked", "merchant"]
        ).cumcount()
        df["user_has_used_merchant_before"] = (df["user_merchant_tx_count"] > 0).astype(
            int
        )
        df["user_mcc_tx_count"] = df.groupby(["card_masked", "mcc"]).cumcount()
        df["user_has_used_mcc_before"] = (df["user_mcc_tx_count"] > 0).astype(int)
        df_dt_indexed = df.set_index("timestamp")
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
        expanding_avg = (
            df.groupby("card_masked")["amount"]
            .expanding()
            .mean()
            .reset_index(0, drop=True)
        )
        df["temp_expanding_avg"] = expanding_avg
        df["user_avg_tx_amount_historical"] = df.groupby("card_masked")[
            "temp_expanding_avg"
        ].shift(1)
        df = df.drop(columns=["temp_expanding_avg"])
        df["amount_to_historical_avg_ratio"] = df["amount"] / (
            df["user_avg_tx_amount_historical"] + 1e-6
        )
        df["is_round_amount"] = (df["amount"] % 1 == 0).astype(int)
        df["amount_cents"] = (df["amount"] * 100 % 100).astype(int)
        cols_to_fill = [
            "time_since_last_tx_seconds",
            "user_tx_count_1h",
            "user_tx_count_6h",
            "user_tx_count_24h",
            "user_unique_merchant_count_1h",
            "user_unique_location_count_24h",
            "user_avg_tx_amount_historical",
            "amount_to_historical_avg_ratio",
        ]
        df[cols_to_fill] = df[cols_to_fill].fillna(-1)
        df = df.drop(columns=["user_merchant_tx_count", "user_mcc_tx_count"])
        return df

    def predict(self, recent_transactions: List[TransactionInput]) -> Dict:
        """
        Performs fraud detection on the most recent transaction, using the provided history.

        Args:
            recent_transactions (List[TransactionInput]): A list of the most recent
                transactions for a single user, with the newest transaction at the end.
                The list length should be up to `max_history_len + 1`.

        Returns:
            A dictionary containing the prediction results.
        """
        if not self.is_ready:
            return {"error": "Service is not ready."}

        if not recent_transactions:
            return {"error": "Input list of transactions cannot be empty."}

        current_card = recent_transactions[-1].card_masked
        df = pd.DataFrame([asdict(tx) for tx in recent_transactions])

        df["mcc"] = df["merchant"].map(self.metadata["merchant_to_mcc"])
        df["mcc_risk_score"] = df["mcc"].map(self.metadata["mcc_risk_map"]).fillna(3)
        df = self._apply_label_encoders(df)

        processed_df = self._feature_engineering(df)

        final_row = processed_df.iloc[-1:]

        initial_features = final_row[self.metadata["initial_features"]]
        scaled_features = self.scaler.transform(initial_features)

        with torch.no_grad():
            latent_tensor = self.autoencoder.encode(
                torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
            )
            new_latent_vector = latent_tensor.cpu().numpy().flatten()

        # 6. Manage history and create lagged features
        if current_card not in self.user_history:
            self.user_history[current_card] = deque(
                [np.zeros_like(new_latent_vector)] * self.max_history_len,
                maxlen=self.max_history_len,
            )

        history = list(self.user_history[current_card])

        final_lgbm_features = [new_latent_vector] + history
        final_lgbm_vector = np.concatenate(final_lgbm_features).reshape(1, -1)

        prediction_proba = self.lgbm_booster.predict(final_lgbm_vector)[0]
        is_fraud = prediction_proba > self.metadata["best_threshold"]

        self.user_history[current_card].appendleft(new_latent_vector)

        return {
            "transaction_id": recent_transactions[-1].transaction_id,
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(prediction_proba),
            "threshold": self.metadata["best_threshold"],
        }


# --- TEST BLOCK ---
if __name__ == "__main__":
    from datetime import datetime, timedelta
    import uuid

    # Initialize the service
    # This will load all models and artifacts into memory
    service = FraudDetectionService()

    if not service.is_ready:
        print("\nService failed to initialize. Exiting.")
    else:
        print("\n--- Running Test Scenario ---")

        # --- Create dummy data for a single user to test history ---
        card_user_1 = "520000XXXXXX1234"
        base_time = datetime.utcnow()

        # Use merchants and channels the model has seen before
        dummy_transactions = [
            TransactionInput(
                transaction_id=str(uuid.uuid4()),
                timestamp=(base_time + timedelta(minutes=i * 10)).isoformat() + "Z",
                amount=25.50 + i * 5,
                merchant="Mercator",
                channel="pos",
                location="SI",
                city="Ljubljana",
                card_issuer="Mastercard",
                card_masked=card_user_1,
            )
            for i in range(6)
        ]
        # Make the last transaction look a bit suspicious
        dummy_transactions[-1].amount = 1250.75
        dummy_transactions[-1].merchant = "CryptoExchange"
        dummy_transactions[-1].channel = "online"
        dummy_transactions[-1].location = "EE"
        dummy_transactions[-1].city = "Tallinn"
        dummy_transactions[-1].timestamp = (
            base_time + timedelta(minutes=61)
        ).isoformat() + "Z"

        # Simulate a stream of transactions for this user
        print(f"\nSimulating transactions for user: {card_user_1}")
        for i in range(len(dummy_transactions)):
            # The input to the predict function is the current transaction + its history
            # In a real system, you'd fetch this history from a database or cache
            start_index = max(0, i - service.max_history_len)
            history_slice = dummy_transactions[start_index : i + 1]

            print(
                f"\nPredicting for transaction {i + 1}/{len(dummy_transactions)} (ID: {history_slice[-1].transaction_id})"
            )
            print(f"  - Using {len(history_slice)} transactions as context.")

            result = service.predict(history_slice)
            print(f"  - Prediction Result: {result}")

        # --- Test a new user with no history ---
        print("\n--- Testing a new user with no history ---")
        card_user_2 = "410000XXXXXX5678"
        new_user_tx = [
            TransactionInput(
                transaction_id=str(uuid.uuid4()),
                timestamp=(base_time + timedelta(minutes=1)).isoformat() + "Z",
                amount=15.00,
                merchant="Petrol",
                channel="pos",
                location="SI",
                city="Celje",
                card_issuer="Visa",
                card_masked=card_user_2,
            )
        ]
        result = service.predict(new_user_tx)
        print(f"Prediction Result for new user: {result}")
