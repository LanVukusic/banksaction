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

from transformer import TransactionTransformerAE
from feature import feature_engineering
from datetime import datetime, timedelta
import uuid
import psycopg2

import asyncio
import json
from nats.aio.client import Client as NATS


MODEL_DIR = "saved_models_transformer"
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

        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
        self.label_encoders = joblib.load(
            os.path.join(self.model_dir, "label_encoders.joblib")
        )

        n_features = len(self.metadata["initial_features"])
        self.transformer = TransactionTransformerAE(
            n_features=n_features,
            d_model=self.metadata["d_model"],
            n_head=self.metadata["n_head"],
            num_layers=self.metadata["num_layers"],
            dim_feedforward=self.metadata["dim_feedforward"],
            seq_length=self.metadata["sequence_length"],
        ).to(DEVICE)

        self.transformer.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "transformer_ae.pth"),
                map_location=DEVICE,
            )
        )
        self.transformer.eval()
        self.sequence_length = self.metadata["sequence_length"]

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
        if not self.is_ready:
            return {"error": "Service is not ready."}

        if len(recent_transactions) < self.sequence_length:
            return {
                "error": f"Not enough transactions to form a sequence. Need {self.sequence_length}, got {len(recent_transactions)}."
            }

        df = pd.DataFrame([asdict(tx) for tx in recent_transactions])
        df["mcc"] = df["merchant"].map(self.metadata["merchant_to_mcc"])
        df["mcc_risk_score"] = df["mcc"].map(self.metadata["mcc_risk_map"]).fillna(3)
        df = self._apply_label_encoders(df)

        processed_df = feature_engineering(df)

        # Ensure we only use the required sequence length from the end
        if len(processed_df) > self.sequence_length:
            processed_df = processed_df.iloc[-self.sequence_length :]

        sequence_features = processed_df[self.metadata["initial_features"]]
        scaled_features = self.scaler.transform(sequence_features)

        # The model expects a batch, so we add a dimension
        sequence_tensor = (
            torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        with torch.no_grad():
            embedding = self.transformer.encode(sequence_tensor)

        return {
            "transaction_id": recent_transactions[-1].transaction_id,
            "embedding": embedding.squeeze().cpu(),
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
        user = data["card_masked"]
        print(f"received data for {user}")

        # We need sequence_length - 1 transactions from history for the new one
        history = get_user_history_df(service.sequence_length - 1, user)

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

        # Combine history with new transaction, ensuring correct order (oldest to newest)
        transaction_inputs = history_transactions[::-1] + [new_transaction]

        print(f"Created {len(transaction_inputs)} transactions for processing")

        result = service.predict(transaction_inputs)

        if "embedding" in result:
            store_embedding(
                embedding=result["embedding"],
                name="Transaction embedding",
                transaction_reference=new_transaction.transaction_id,
            )
        else:
            print(f"Could not generate embedding: {result.get('error')}")

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
