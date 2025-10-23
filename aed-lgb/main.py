import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from torch.optim.lr_scheduler import ReduceLROnPlateau
import uuid
import datetime
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import joblib
import json

from feature import feature_engineering
from ae import TransactionAutoencoder

from data_generator import MERCHANTS, generate_fraud_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATASET_FILE = "imbalanced_fraud_dataset.csv"
TEST_SIZE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "saved_models"


AE_EPOCHS = 40
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 2e-3
BOTTLENECK_DIM = 32


if __name__ == "__main__":
    import multiprocessing
    from sklearn.preprocessing import RobustScaler

    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file '{DATASET_FILE}' not found. Generating new dataset...")
        generate_fraud_dataset(
            output_file=DATASET_FILE, n_legit=2000000, n_fraud_events=8000
        )
    else:
        print(f"Found existing dataset: '{DATASET_FILE}'.")

    raw_df = pd.read_csv(DATASET_FILE)

    print("Performing pre-feature engineering mapping...")
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

    print("Label encoding categorical columns...")
    categorical_cols = ["merchant", "channel", "location", "city", "card_issuer"]
    for col in categorical_cols:
        raw_df[col] = LabelEncoder().fit_transform(raw_df[col].astype(str))

    encoders = {}
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

    X = processed_df[features_to_use]
    y = processed_df["TX_FRAUD"]

    users = processed_df["card_masked"].unique()
    train_users, test_users = train_test_split(
        users, test_size=TEST_SIZE, random_state=42
    )
    train_indices = processed_df["card_masked"].isin(train_users)
    test_indices = processed_df["card_masked"].isin(test_users)
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    print(f"Length of training set: {len(X_train)} and test set: {len(X_test)}")

    X_train_legit = X_train[y_train == 0]
    scaler = RobustScaler()
    X_train_legit_scaled = scaler.fit_transform(X_train_legit)

    print(
        f"\n--- Starting Autoencoder Training on {len(X_train_legit_scaled)} legitimate transactions ---"
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_legit_scaled, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=AE_BATCH_SIZE, shuffle=True, num_workers=0
    )

    n_features = X_train_legit_scaled.shape[1]
    autoencoder = TransactionAutoencoder(n_features, BOTTLENECK_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=AE_LEARNING_RATE, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    for epoch in range(AE_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{AE_EPOCHS}")
        epoch_loss = 0.0
        for data in progress_bar:
            inputs = data[0].to(DEVICE)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{AE_EPOCHS}], Average Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    print("Autoencoder training complete.")
    print("\n--- Generating and Lagging Latent Vectors ---")
    autoencoder.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X)
        all_latent_vectors = (
            autoencoder.encode(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE))
            .cpu()
            .numpy()
        )

    latent_df = pd.DataFrame(
        all_latent_vectors, columns=[f"ae_{i}" for i in range(BOTTLENECK_DIM)]
    )
    latent_df["card_masked"] = processed_df["card_masked"].values

    final_features_list = []
    ae_cols = [f"ae_{i}" for i in range(BOTTLENECK_DIM)]
    final_features_list.append(latent_df[ae_cols])
    lags = 5
    for lag in range(1, lags + 1):
        lagged_features = latent_df.groupby("card_masked")[ae_cols].shift(lag)
        lagged_features.columns = [f"ae_{i}_lag_{lag}" for i in range(BOTTLENECK_DIM)]
        final_features_list.append(lagged_features)

    final_features_df = pd.concat(final_features_list, axis=1)
    final_features_df.fillna(0, inplace=True)

    X_train_final = final_features_df.loc[train_indices]
    X_test_final = final_features_df.loc[test_indices]

    print(f"Final feature shape for LightGBM: {X_train_final.shape}")
    print("\n--- Starting LightGBM Training ---")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        metric="f1",
    )
    lgb_model.fit(X_train_final, y_train, eval_set=[(X_test_final, y_test)])

    print("\n--- Evaluation and Threshold Tuning ---")
    y_pred_proba = lgb_model.predict_proba(X_test_final)[:, 1]
    best_f1, best_threshold = 0, 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred_tuned = (y_pred_proba > threshold).astype(int)
        current_f1 = f1_score(y_test, y_pred_tuned, pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"\nOptimal threshold found at: {best_threshold:.2f}")
    print(f"Best F1-score for Fraud class: {best_f1:.4f}")

    final_y_pred = (y_pred_proba > best_threshold).astype(int)
    print("\n--- Final Classification Report ---")
    print(
        classification_report(
            y_test, final_y_pred, target_names=["Legitimate", "Fraud"], zero_division=0
        )
    )

    print("\n--- Exporting models and artifacts ---")

    ae_path = os.path.join(MODEL_DIR, "autoencoder.pth")
    torch.save(autoencoder.state_dict(), ae_path)
    print(f"Autoencoder state_dict saved to: {ae_path}")

    lgbm_path = os.path.join(MODEL_DIR, "lightgbm_model.txt")
    lgb_model.booster_.save_model(lgbm_path)
    print(f"LightGBM model saved to: {lgbm_path}")

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    encoders_path = os.path.join(MODEL_DIR, "label_encoders.joblib")
    joblib.dump(encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")

    metadata = {
        "best_threshold": best_threshold,
        "initial_features": features_to_use,
        "final_lgbm_features": X_train_final.columns.tolist(),
        "bottleneck_dim": BOTTLENECK_DIM,
        "ae_lags": lags,
        "mcc_risk_map": mcc_risk_map,
        "merchant_to_mcc": merchant_to_mcc,
    }
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to: {metadata_path}")

    print("\nAll artifacts successfully exported.")
