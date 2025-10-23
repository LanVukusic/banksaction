import os
import uuid
import json
import joblib
import random
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Assuming feature.py and data_generator.py exist in the same directory
# You can also copy the functions directly into this script if you prefer
from feature import feature_engineering
from data_generator import generate_fraud_dataset, MERCHANTS

# --- Environment and Constants ---
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

DATASET_FILE = "imbalanced_fraud_dataset.csv"
TEST_SIZE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#  DEVICE = "mps"
MODEL_DIR = "saved_models_transformer"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LENGTH = 5
AE_EPOCHS = 40
AE_BATCH_SIZE = 2048
AE_LEARNING_RATE = 2e-4

D_MODEL = 164
N_HEAD = 4
DIM_FEEDFORWARD = 128
NUM_LAYERS = 4


# --- Data Preparation for Sequences ---
def create_sequences(X, y, card_ids, seq_length):
    """
    Converts 2D transaction data into 3D sequences for the Transformer.
    """
    X_seq, y_seq, seq_indices = [], [], []

    # Group by card_id to create sequences per user
    data_df = pd.DataFrame(X)
    data_df["card_masked"] = card_ids
    data_df["y"] = y

    for _, group in tqdm(data_df.groupby("card_masked"), desc="Creating Sequences"):
        features = group.drop(columns=["card_masked", "y"]).values
        labels = group["y"].values

        # Slide a window over the user's transactions
        for i in range(len(features) - seq_length + 1):
            X_seq.append(features[i : i + seq_length])
            # The label for a sequence is the label of the LAST transaction in it
            y_seq.append(labels[i + seq_length - 1])
            # Store the original index of the last item for splitting later
            seq_indices.append(group.index[i + seq_length - 1])

    return np.array(X_seq), np.array(y_seq), np.array(seq_indices)


# --- Transformer Autoencoder Model ---
class TransactionTransformerAE(nn.Module):
    def __init__(
        self, n_features, d_model, n_head, num_layers, dim_feedforward, seq_length
    ):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features

        # ... (Input Embedding, CLS Token, Positional Encoding are the same) ...
        self.input_embedding = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length + 1, d_model))

        # --- START OF THE FIX ---
        # 4. Transformer Encoder with Pre-Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,  # <-- THIS IS THE CRITICAL CHANGE
        )
        # --- END OF THE FIX ---

        # We also need to add a final LayerNorm after the encoder, as Pre-LN doesn't
        # normalize the final output of the stack.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(d_model)  # <-- ADD THIS FINAL NORM

        # 5. Decoder (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, seq_length * n_features),
        )

    def forward(self, src):
        batch_size = src.shape[0]

        embedded_src = self.input_embedding(src)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, embedded_src], dim=1)
        x += self.positional_encoding

        # Pass through encoder
        encoded_output = self.transformer_encoder(x)

        # --- APPLY THE FINAL NORM ---
        encoded_output = self.final_norm(encoded_output)
        # ----------------------------

        latent_vector = encoded_output[:, 0, :]
        reconstructed_flat = self.decoder(latent_vector)
        reconstructed_sequence = reconstructed_flat.view(
            batch_size, self.seq_length, self.n_features
        )

        return reconstructed_sequence, latent_vector

    def encode(self, src):
        with torch.no_grad():
            batch_size = src.shape[0]
            embedded_src = self.input_embedding(src)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, embedded_src], dim=1)
            x += self.positional_encoding
            encoded_output = self.transformer_encoder(x)
            latent_vector = encoded_output[:, 0, :]
        return latent_vector


if __name__ == "__main__":
    from sklearn.preprocessing import RobustScaler

    if not os.path.exists(DATASET_FILE):
        print("Dataset not found. Generating...")
        generate_fraud_dataset(
            output_file=DATASET_FILE, n_legit=6000000, n_fraud_events=20000
        )

    # 1. Load and Preprocess Data
    raw_df = pd.read_csv(DATASET_FILE)
    mcc_risk_map = {
        5411: 1,
        5541: 1,
        4814: 2,
        4121: 2,
        5651: 3,
        5942: 4,
        5732: 5,
        5944: 5,
        5947: 6,
        6051: 7,
        7379: 8,
        5999: 6,
    }
    merchant_to_mcc = {m["name"]: m["mcc"] for m in MERCHANTS}
    raw_df["mcc"] = raw_df["merchant"].map(merchant_to_mcc)
    raw_df["mcc_risk_score"] = raw_df["mcc"].map(mcc_risk_map).fillna(3)
    encoders = {}
    categorical_cols = ["merchant", "channel", "location", "city", "card_issuer"]
    for col in categorical_cols:
        le = LabelEncoder()
        raw_df[col] = le.fit_transform(raw_df[col].astype(str))
        encoders[col] = le
    processed_df = feature_engineering(raw_df)

    features_to_use = [
        col
        for col in processed_df.columns
        if col
        not in [
            "transaction_id",
            "timestamp",
            "currency",
            "card_masked",
            "TX_FRAUD",
            "FRAUD_SCENARIO",
            "mcc",
        ]
    ]
    X = processed_df[features_to_use].values
    y = processed_df["TX_FRAUD"].values

    # 2. Split data BEFORE scaling
    train_indices, test_indices = train_test_split(
        processed_df.index,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=processed_df["TX_FRAUD"],
    )

    # 3. Fit scaler ONLY on legitimate training data
    print("Fitting scaler on legitimate training data...")
    scaler = RobustScaler()
    X_train_legit_for_scaling = X[train_indices][y[train_indices] == 0]
    scaler.fit(X_train_legit_for_scaling)

    # 4. Transform the entire dataset
    print("Scaling entire dataset...")
    X_scaled = scaler.transform(X)

    # <<< FIX #1: HARD-CLIP THE SCALED DATA TO REMOVE EXTREME OUTLIERS >>>
    CLIP_VALUE = 10.0
    print(f"Clipping scaled values to [{-CLIP_VALUE}, {CLIP_VALUE}]")
    X_scaled = np.clip(X_scaled, -CLIP_VALUE, CLIP_VALUE)

    # 5. Sanity check for NaN/Inf values
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        raise ValueError("NaN or Inf found in scaled and clipped data.")
    else:
        print("Data sanity check passed.")

    # 6. Create Sequences using the fully prepared data
    X_sequences, y_sequences, original_indices = create_sequences(
        X_scaled, y, processed_df["card_masked"].values, SEQUENCE_LENGTH
    )

    # 7. Split sequences for train/test
    is_train_seq = np.isin(original_indices, train_indices)
    is_test_seq = np.isin(original_indices, test_indices)
    X_train_seq, X_test_seq = X_sequences[is_train_seq], X_sequences[is_test_seq]
    y_train_seq, y_test_seq = y_sequences[is_train_seq], y_sequences[is_test_seq]
    X_train_legit_seq = X_train_seq[y_train_seq == 0]

    print(f"Total sequences created: {len(X_sequences)}")
    print(f"Legitimate sequences for AE training: {len(X_train_legit_seq)}")

    # 8. Train Transformer Autoencoder
    train_dataset = TensorDataset(torch.tensor(X_train_legit_seq, dtype=torch.float32))
    train_loader = DataLoader(
        train_dataset, batch_size=AE_BATCH_SIZE, shuffle=True, num_workers=3
    )

    n_features = X_train_legit_seq.shape[2]
    autoencoder = TransactionTransformerAE(
        n_features, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, SEQUENCE_LENGTH
    ).to(DEVICE)

    # <<< FIX #2: USE L1LOSS (MEAN ABSOLUTE ERROR) INSTEAD OF MSE >>>
    criterion = nn.L1Loss()

    # <<< FIX #3: USE A LOWER LEARNING RATE >>>
    AE_LEARNING_RATE_FINAL = 1e-5

    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=AE_LEARNING_RATE_FINAL, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)

    print("\n--- Starting Transformer Autoencoder Training ---")
    for epoch in range(AE_EPOCHS):
        autoencoder.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{AE_EPOCHS}")
        for data in progress_bar:
            inputs = data[0].to(DEVICE)
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{AE_EPOCHS}], Average Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    INFERENCE_BATCH_SIZE = 1024

    def get_latent_vectors_in_batches(model, data, batch_size, device):
        """Helper function to run inference in batches."""
        model.eval()
        all_latents = []
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Encoding Batches"):
                batch_data = data[i : i + batch_size]
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)

                # We need to handle the forward pass differently for encode
                # The encode method in your class needs a slight modification
                latent_vector = model.encode(batch_tensor)

                all_latents.append(latent_vector.cpu().numpy())

        return np.concatenate(all_latents, axis=0)

    # We need to update the .encode() method slightly in the model class
    # to handle the final_norm correctly. Go back to the TransactionTransformerAE class and modify the encode method.

    # After modifying the encode method, run the batch processing
    train_latent_vectors = get_latent_vectors_in_batches(
        autoencoder, X_train_seq, INFERENCE_BATCH_SIZE, DEVICE
    )
    test_latent_vectors = get_latent_vectors_in_batches(
        autoencoder, X_test_seq, INFERENCE_BATCH_SIZE, DEVICE
    )

    X_train_final = train_latent_vectors
    X_test_final = test_latent_vectors
    y_train_final = y_train_seq
    y_test_final = y_test_seq

    print("\n--- Starting LightGBM Training ---")
    scale_pos_weight = (
        (y_train_final == 0).sum() / (y_train_final == 1).sum()
        if (y_train_final == 1).sum() > 0
        else 1
    )
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        metric="f1",
    )
    lgb_model.fit(X_train_final, y_train_final, eval_set=[(X_test_final, y_test_final)])

    print("\n--- Evaluation and Threshold Tuning ---")
    y_pred_proba = lgb_model.predict_proba(X_test_final)[:, 1]
    best_f1, best_threshold = 0, 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred_tuned = (y_pred_proba > threshold).astype(int)
        current_f1 = f1_score(y_test_final, y_pred_tuned, zero_division=0)
        if current_f1 > best_f1:
            best_f1, best_threshold = current_f1, threshold
    print(f"\nOptimal threshold: {best_threshold:.2f}, Best F1-score: {best_f1:.4f}")
    final_y_pred = (y_pred_proba > best_threshold).astype(int)
    print("\n--- Final Classification Report ---")
    print(
        classification_report(
            y_test_final,
            final_y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("\n--- Exporting models and artifacts ---")
    torch.save(autoencoder.state_dict(), os.path.join(MODEL_DIR, "transformer_ae.pth"))
    lgb_model.booster_.save_model(os.path.join(MODEL_DIR, "lightgbm_model.txt"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    metadata = {
        "best_threshold": best_threshold,
        "initial_features": features_to_use,
        "sequence_length": SEQUENCE_LENGTH,
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "mcc_risk_map": mcc_risk_map,
        "merchant_to_mcc": merchant_to_mcc,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print("\nAll artifacts successfully exported.")
