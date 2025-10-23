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

from ae import TransactionAutoencoder
from feature import feature_engineering


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

        processed_df = feature_engineering(df)

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


# # --- TEST BLOCK ---
# if __name__ == "__main__":
#     from datetime import datetime, timedelta
#     import uuid

#     # Initialize the service
#     # This will load all models and artifacts into memory
#     service = FraudDetectionService()

#     if not service.is_ready:
#         print("\nService failed to initialize. Exiting.")
#     else:
#         print("\n--- Running Test Scenario ---")

#         # --- Create dummy data for a single user to test history ---
#         card_user_1 = "520000XXXXXX1234"
#         base_time = datetime.utcnow()

#         # flip for merchant, either "Mercator" or "Clothing Stores"
#         random_merchant = "Mercator" if np.random.rand() > 0.5 else "Clothing Stores"

#         # Use merchants and channels the model has seen before
#         dummy_transactions = [
#             TransactionInput(
#                 transaction_id=str(uuid.uuid4()),
#                 timestamp=(base_time + timedelta(minutes=i * 200)).isoformat() + "Z",
#                 amount=25.50 + i * 5,
#                 merchant=random_merchant,
#                 channel="pos",
#                 location="SI",
#                 city="Ljubljana",
#                 card_issuer="Mastercard",
#                 card_masked=card_user_1,
#             )
#             for i in range(8)
#         ]
#         # Make the last transaction look a bit suspicious
#         dummy_transactions[-1].amount = 1250.75
#         dummy_transactions[-1].merchant = "CryptoExchange"
#         dummy_transactions[-1].channel = "online"
#         dummy_transactions[-1].location = "EE"
#         dummy_transactions[-1].city = "Tallinn"
#         dummy_transactions[-1].timestamp = (
#             base_time + timedelta(minutes=61)
#         ).isoformat() + "Z"

#         # Simulate a stream of transactions for this user
#         print(f"\nSimulating transactions for user: {card_user_1}")
#         for i in range(len(dummy_transactions)):
#             # The input to the predict function is the current transaction + its history
#             # In a real system, you'd fetch this history from a database or cache
#             start_index = max(0, i - service.max_history_len)
#             history_slice = dummy_transactions[start_index : i + 1]

#             print(
#                 f"\nPredicting for transaction {i + 1}/{len(dummy_transactions)} (ID: {history_slice[-1].transaction_id})"
#             )
#             print(f"  - Using {len(history_slice)} transactions as context.")

#             result = service.predict(history_slice)
#             print(f"  - Prediction Result: {result}")

#         # --- Test a new user with no history ---
#         print("\n--- Testing a new user with no history ---")
#         card_user_2 = "410000XXXXXX5678"
#         new_user_tx = [
#             TransactionInput(
#                 transaction_id=str(uuid.uuid4()),
#                 timestamp=(base_time + timedelta(minutes=1)).isoformat() + "Z",
#                 amount=15.00,
#                 merchant="Petrol",
#                 channel="pos",
#                 location="SI",
#                 city="Celje",
#                 card_issuer="Visa",
#                 card_masked=card_user_2,
#             )
#         ]
#         result = service.predict(new_user_tx)
#         print(f"Prediction Result for new user: {result}")

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import uuid

    service = FraudDetectionService()

    if not service.is_ready:
        print("\nService failed to initialize. Exiting.")
    else:
        print("\n--- Running Test Scenario ---")

        # --- Create dummy data for a single user to test history ---
        card_user_1 = "520000XXXXXX1234"
        base_time = datetime.utcnow()

        dummy_transactions = [
            TransactionInput(
                transaction_id=str(uuid.uuid4()),
                timestamp=(base_time + timedelta(minutes=i * 100)).isoformat() + "Z",
                amount=25.50 + i * 5,
                merchant="Mercator", channel="pos", location="SI", city="Ljubljana",
                card_issuer="Mastercard", card_masked=card_user_1,
            ) for i in range(7) # 7 normal transactions
        ]
        # Make the last transaction look suspicious
        dummy_transactions.append(
            TransactionInput(
                transaction_id=str(uuid.uuid4()),
                timestamp=(base_time + timedelta(minutes=71)).isoformat() + "Z",
                amount=1250.75,
                merchant="CryptoExchange", channel="online", location="EE", city="Tallinn",
                card_issuer="Mastercard", card_masked=card_user_1,
            )
        )

        # Simulate a stream of transactions for this user
        print(f"\nSimulating transactions for user: {card_user_1}")
        
        # THIS IS THE KEY CHANGE: We accumulate the history
        transaction_history_for_user = []
        for new_tx in dummy_transactions:
            # Add the new transaction to the user's history
            transaction_history_for_user.append(new_tx)
            
            # The context passed to predict is the FULL history up to this point
            current_context = transaction_history_for_user

            print(
                f"\nPredicting for transaction (ID: {current_context[-1].transaction_id})"
            )
            print(f"  - Using {len(current_context)} transactions as context.")

            result = service.predict(current_context)
            print(f"  - Prediction Result: {result}")

        # --- Test a new user with no history ---
        # (This part remains the same and is correct)
        print("\n--- Testing a new user with no history ---")
        card_user_2 = "410000XXXXXX5678"
        new_user_tx = [
            TransactionInput(
                transaction_id=str(uuid.uuid4()),
                timestamp=(base_time + timedelta(minutes=1)).isoformat() + "Z",
                amount=15.00, merchant="Petrol", channel="pos",
                location="SI", city="Celje", card_issuer="Visa",
                card_masked=card_user_2,
            )
        ]
        result = service.predict(new_user_tx)
        print(f"Prediction Result for new user: {result}")