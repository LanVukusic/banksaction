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
from datetime import datetime, timedelta
import uuid
import psycopg2

import asyncio
import json
from nats.aio.client import Client as NATS


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


db_conn = None  # Global database connection


def get_user_history_df(num_latest: int, user_card_masked: str):
    global db_conn
    if db_conn is None:
        print("Database connection not initialized.")
        return

    cursor = db_conn.cursor()

    # Query to get the latest transactions
    query = "SELECT * FROM transactions WHERE card_masked = %s ORDER BY timestamp DESC LIMIT %s;"
    cursor.execute(query, (user_card_masked, num_latest))

    # Get all results
    rows = cursor.fetchall()

    # Get column names
    columns = [desc[0] for desc in cursor.description]

    # Create DataFrame manually
    df = pd.DataFrame(rows, columns=columns)
    return df

    cursor.close()


def store_fraud_prediction(transaction_id, fraud_probability):
    """Stores fraud prediction results in the database using the global connection."""
    global db_conn
    if db_conn is None:
        print("Database connection not initialized.")
        return

    try:
        cursor = db_conn.cursor()

        query = """
        INSERT INTO fraud_predictions 
        (transaction_id, fraud_probability, prediction_timestamp, source_model) 
        VALUES (%s, %s, NOW(), "LGBM_transformer")
        """
        cursor.execute(query, (transaction_id, fraud_probability))

        db_conn.commit()
        print(
            f"Stored fraud prediction for transaction '{transaction_id}' with probability {fraud_probability}."
        )
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error storing fraud prediction: {error}")
        db_conn.rollback()


def store_embedding(embedding, name, transaction_reference):
    """Stores an embedding in the database using the global connection."""
    global db_conn
    if db_conn is None:
        print("Database connection not initialized.")
        return

    try:
        cursor = db_conn.cursor()

        # Convert tensor to list for storage
        embedding_list = embedding.detach().numpy().tolist()

        query = (
            "INSERT INTO embeddings (embedding, name, transaction) VALUES (%s, %s, %s)"
        )
        cursor.execute(query, (embedding_list, name, transaction_reference))

        db_conn.commit()
        print(f"Stored embedding for '{name}'.")
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        db_conn.rollback()  # Rollback in case of error


async def run():
    global db_conn

    # Initialize database connection
    try:
        db_conn = psycopg2.connect(
            "dbname=transactions user=user password=password host=localhost port=5432 sslmode=disable"
        )
        print("Database connection established.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error connecting to database: {error}")
        return

    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    service = FraudDetectionService()

    async def message_handler(msg):
        subject = msg.subject
        data = json.loads(msg.data.decode())
        print(f"Received a message on '{subject}'")
        user = data["card_masked"]  # Fixed: access dictionary key
        print(f"received data for {user}")

        history = get_user_history_df(40, user)

        # If history_df is a DataFrame, convert it to TransactionInput objects
        history_transactions = []
        if not history.empty:
            for _, row in history.iterrows():
                history_transactions.append(
                    TransactionInput(
                        transaction_id=row["transaction_id"],
                        timestamp=row["timestamp"],
                        amount=row["amount"],
                        merchant=row["merchant"],
                        channel=row["channel"],
                        location=row["location"],
                        city=row["city"],
                        card_issuer=row["card_issuer"],
                        card_masked=row["card_masked"],
                    )
                )

        ids = [i.transaction_id for i in history_transactions]
        print(ids)

        # Convert the new transaction data to TransactionInput format
        new_transaction = TransactionInput(
            transaction_id=data["transaction_id"],
            timestamp=data["timestamp"],
            amount=data["amount"],
            merchant=data["merchant"],
            channel=data["channel"],
            location=data["location"],
            city=data["city"],
            card_issuer=data["card_issuer"],
            card_masked=data["card_masked"],
        )

        # Combine history with new transaction
        transaction_inputs = history_transactions + [new_transaction]

        print(f"Created {len(transaction_inputs)} transactions for processing")

        result = service.predict(transaction_inputs)
        store_fraud_prediction(
            new_transaction.transaction_id, result["fraud_probability"]
        )

        embedding = result["embedding"]
        store_embedding(
            embedding=embedding,
            name="Transaction embedding",
            transaction_reference=new_transaction.transaction_id,
        )

        print("\n\n")

    # Subscribe to the 'transactions' topic
    await nc.subscribe("transactions", cb=message_handler)
    print("Subscribed to 'transactions' topic.")

    # Keep the connection alive
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await nc.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Subscriber stopped.")
