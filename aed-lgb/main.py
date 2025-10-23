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
from faker import Faker
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import joblib
import json

from ae import TransactionAutoencoderV2

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

MERCHANTS = [
    # Standard Merchants
    {
        "name": "Mercator",
        "mcc": 5411,
        "category": "Grocery Stores",
        "locations": [("SI", "Ljubljana"), ("SI", "Maribor")],
    },
    {
        "name": "Petrol",
        "mcc": 5541,
        "category": "Gas Stations",
        "locations": [("SI", "Celje"), ("HR", "Zagreb")],
    },
    {
        "name": "Amazon",
        "mcc": 5942,
        "category": "Online Retail",
        "locations": [("DE", "Berlin"), ("US", "Seattle"), ("UK", "London")],
    },
    {
        "name": "H&M",
        "mcc": 5651,
        "category": "Clothing Stores",
        "locations": [("SI", "Ljubljana"), ("AT", "Vienna")],
    },
    {
        "name": "Telekom Slovenije",
        "mcc": 4814,
        "category": "Telecommunication",
        "locations": [("SI", "Ljubljana")],
    },
    {
        "name": "Uber",
        "mcc": 4121,
        "category": "Transportation",
        "locations": [("DE", "Berlin"), ("US", "New York")],
    },
    # High-Risk Merchants for Fraud Scenarios
    {
        "name": "BigBox Electronics",
        "mcc": 5732,
        "category": "Electronics Stores",
        "locations": [("DE", "Berlin"), ("US", "New York")],
    },
    {
        "name": "Luxury Watches",
        "mcc": 5944,
        "category": "Jewelry Stores",
        "locations": [("AT", "Vienna"), ("US", "New York")],
    },
    {
        "name": "GiftCardHeaven",
        "mcc": 5947,
        "category": "Gift Card Stores",
        "locations": [("DE", "Berlin"), ("US", "Seattle")],
    },
    {
        "name": "CryptoExchange",
        "mcc": 6051,
        "category": "Financial Services",
        "locations": [("EE", "Tallinn"), ("MT", "Valletta")],
    },
    # Merchants for specific scenarios
    {
        "name": "ProTech Support",
        "mcc": 7379,
        "category": "Professional Services",
        "locations": [("IN", "Bangalore")],
    },  # For Social Engineering
    {
        "name": "Shady Gadgets Inc.",
        "mcc": 5999,
        "category": "Misc Retail",
        "locations": [("CY", "Nicosia")],
    },  # For Merchant Collusion
    {
        "name": "eBay",
        "mcc": 5999,
        "category": "Online Marketplace",
        "locations": [("US", "San Jose")],
    },  # For Triangulation
]

CHANNELS = ["pos", "online", "mobile", "moto"]
CARD_ISSUERS = ["Mastercard", "Visa"]
fake = Faker()


def create_card_details():
    card_bin = str(random.choice([520000, 410000, 550000]))
    last4 = f"{random.randint(1000, 9999)}"
    return card_bin, last4, f"{card_bin}XXXXXX{last4}", random.choice(CARD_ISSUERS)


def generate_legitimate_transactions(n, start_time):
    transactions = []
    current_time = start_time
    legit_merchants = [
        m
        for m in MERCHANTS
        if m["category"] not in ["Professional Services", "Misc Retail"]
    ]
    for _ in tqdm(range(n), desc="Generating Legitimate Transactions"):
        merchant_profile = random.choice(legit_merchants)
        location_code, city = random.choice(merchant_profile["locations"])
        amount = round(np.random.lognormal(mean=3.5, sigma=1.0) + 5, 2)
        _, _, masked, issuer = create_card_details()
        current_time += datetime.timedelta(seconds=random.randint(1, 300))
        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": current_time.isoformat(),
                "amount": amount,
                "currency": "EUR",
                "location": location_code,
                "city": city,
                "merchant": merchant_profile["name"],
                "channel": random.choices(
                    CHANNELS, weights=[0.6, 0.3, 0.09, 0.01], k=1
                )[0],
                "card_masked": masked,
                "card_issuer": issuer,
                "TX_FRAUD": 0,
                "FRAUD_SCENARIO": "None",
            }
        )
    return transactions


def get_night_time(base_time):
    return base_time.replace(
        hour=random.randint(1, 4),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
    )


def _generate_high_value(base_time, masked_card, issuer):
    merchant = random.choice(
        [
            m
            for m in MERCHANTS
            if m["category"] in ["Electronics Stores", "Jewelry Stores"]
        ]
    )
    location, city = random.choice(merchant["locations"])
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": get_night_time(base_time).isoformat(),
            "amount": round(random.uniform(1500.0, 7000.0), 2),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "online",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "High Value",
        }
    ]


def _generate_impossible_travel(base_time, masked_card, issuer):
    txs = []
    locations = [("SI", "Ljubljana"), ("US", "New York"), ("UK", "London")]
    random.shuffle(locations)
    for i in range(random.randint(2, 3)):
        location_code, city = locations[i]
        merchant = random.choice(
            [
                m
                for m in MERCHANTS
                if location_code in [loc[0] for loc in m["locations"]]
            ]
            or MERCHANTS
        )
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    base_time + datetime.timedelta(minutes=i * random.randint(5, 15))
                ).isoformat(),
                "amount": round(random.uniform(80.0, 500.0), 2),
                "currency": "EUR",
                "location": location_code,
                "city": city,
                "merchant": merchant["name"],
                "channel": random.choice(["pos", "online"]),
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Impossible Travel",
            }
        )
    return txs


def _generate_carding_attack(base_time, masked_card, issuer):
    txs = []
    merchant = random.choice([m for m in MERCHANTS if m["category"] == "Online Retail"])
    location, city = random.choice(merchant["locations"])
    for i in range(random.randint(4, 8)):
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    base_time + datetime.timedelta(seconds=i * random.randint(2, 10))
                ).isoformat(),
                "amount": round(random.uniform(0.5, 2.5), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": "online",
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Carding Attack",
            }
        )
    return txs


def _generate_local_shopping_spree(base_time, masked_card, issuer):
    txs = []
    city_locations = random.choice(
        [m["locations"] for m in MERCHANTS if len(m["locations"]) > 1]
    )
    location, city = city_locations[0]
    merchants_in_city = [
        m for m in MERCHANTS if any(loc[1] == city for loc in m["locations"])
    ]
    for i in range(random.randint(3, 5)):
        merchant = random.choice(merchants_in_city)
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    base_time + datetime.timedelta(minutes=i * random.randint(3, 8))
                ).isoformat(),
                "amount": round(random.uniform(50.0, 350.0), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": "pos",
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Local Shopping Spree",
            }
        )
    return txs


def _generate_account_takeover(base_time, masked_card, issuer):
    txs = []
    risky_merchants = [
        m for m in MERCHANTS if m["category"] in ["Gift Card Stores", "CryptoExchange"]
    ]
    for i in range(random.randint(2, 4)):
        merchant = random.choice(risky_merchants)
        location, city = random.choice(merchant["locations"])
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    get_night_time(base_time)
                    + datetime.timedelta(minutes=i * random.randint(2, 7))
                ).isoformat(),
                "amount": round(random.uniform(200.0, 900.0), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": "online",
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Account Takeover",
            }
        )
    return txs


def _generate_bust_out(base_time, masked_card, issuer):
    txs = []
    night_time = get_night_time(base_time)
    for i in range(random.randint(3, 5)):
        merchant = random.choice(
            [
                m
                for m in MERCHANTS
                if m["category"] not in ["Gas Stations", "Grocery Stores"]
            ]
        )
        location, city = random.choice(merchant["locations"])
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    night_time + datetime.timedelta(minutes=i * random.randint(5, 10))
                ).isoformat(),
                "amount": round(random.uniform(800.0, 2500.0), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": random.choice(["pos", "online"]),
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Bust-Out Fraud",
            }
        )
    return txs


def _generate_merchant_collusion(base_time, masked_card, issuer):
    merchant = next(m for m in MERCHANTS if m["name"] == "Shady Gadgets Inc.")
    location, city = merchant["locations"][0]
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_time.isoformat(),
            "amount": round(random.uniform(100, 1000), -2),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "online",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "Merchant Collusion",
        }
    ]


def _generate_moto_fraud(base_time, masked_card, issuer):
    merchant = random.choice(
        [m for m in MERCHANTS if m["category"] in ["Electronics Stores", "Misc Retail"]]
    )
    location, city = random.choice(merchant["locations"])
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_time.isoformat(),
            "amount": round(random.uniform(500.0, 1500.0), 2),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "moto",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "MOTO Fraud",
        }
    ]


def _generate_sim_swap(base_time, masked_card, issuer):
    txs = []
    night_time = get_night_time(base_time)
    city_locations = random.choice(
        [
            m["locations"]
            for m in MERCHANTS
            if len(m["locations"]) > 1 and m["category"] not in ["Gas Stations"]
        ]
    )
    location, city = city_locations[0]
    merchants_in_city = [
        m for m in MERCHANTS if any(loc[1] == city for loc in m["locations"])
    ]
    for i in range(random.randint(2, 4)):
        merchant = random.choice(merchants_in_city)
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (
                    night_time + datetime.timedelta(minutes=i * random.randint(2, 5))
                ).isoformat(),
                "amount": round(random.uniform(100.0, 800.0), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": "mobile",
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "SIM Swap",
            }
        )
    return txs


def _generate_triangulation(base_time, masked_card, issuer):
    merchant = random.choice(
        [m for m in MERCHANTS if m["name"] in ["Amazon", "eBay", "BigBox Electronics"]]
    )
    location, city = random.choice(merchant["locations"])
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_time.isoformat(),
            "amount": round(random.uniform(50.0, 450.0), 2),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "online",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "Triangulation Fraud",
        }
    ]


def _generate_friendly_fraud(base_time, masked_card, issuer):
    merchant = random.choice(
        [
            m
            for m in MERCHANTS
            if m["category"]
            in ["Clothing Stores", "Online Retail", "Electronics Stores"]
        ]
    )
    location, city = random.choice(merchant["locations"])
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_time.isoformat(),
            "amount": round(random.uniform(50.0, 300.0), 2),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "online",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "Friendly Fraud",
        }
    ]


def _generate_social_engineering(base_time, masked_card, issuer):
    merchant = next(m for m in MERCHANTS if m["name"] == "ProTech Support")
    location, city = merchant["locations"][0]
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_time.isoformat(),
            "amount": round(random.choice([299.99, 499.99, 999.95])),
            "currency": "EUR",
            "location": location,
            "city": city,
            "merchant": merchant["name"],
            "channel": "online",
            "card_masked": masked_card,
            "card_issuer": issuer,
            "TX_FRAUD": 1,
            "FRAUD_SCENARIO": "Social Engineering",
        }
    ]


def _generate_synthetic_identity(base_time, masked_card, issuer):
    txs = []
    merchants = [
        next(m for m in MERCHANTS if m["category"] == "Telecommunication"),
        next(m for m in MERCHANTS if m["category"] == "Gas Stations"),
        next(m for m in MERCHANTS if m["category"] == "Grocery Stores"),
    ]
    for i in range(random.randint(2, 4)):
        merchant = merchants[i % len(merchants)]
        location, city = random.choice(merchant["locations"])
        txs.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": (base_time - datetime.timedelta(days=i * 7)).isoformat(),
                "amount": round(random.uniform(25.0, 75.0), 2),
                "currency": "EUR",
                "location": location,
                "city": city,
                "merchant": merchant["name"],
                "channel": random.choice(["pos", "online"]),
                "card_masked": masked_card,
                "card_issuer": issuer,
                "TX_FRAUD": 1,
                "FRAUD_SCENARIO": "Synthetic Identity",
            }
        )
    return txs


def generate_fraud_scenarios(base_time):
    scenario_generators = {
        "impossible_travel": _generate_impossible_travel,
        "carding_attack": _generate_carding_attack,
        "high_value": _generate_high_value,
        "local_shopping_spree": _generate_local_shopping_spree,
        "account_takeover": _generate_account_takeover,
        "bust_out": _generate_bust_out,
        "sim_swap": _generate_sim_swap,
        "triangulation": _generate_triangulation,
        "moto_fraud": _generate_moto_fraud,
        "social_engineering": _generate_social_engineering,
        "friendly_fraud": _generate_friendly_fraud,
        "merchant_collusion": _generate_merchant_collusion,
        "synthetic_identity": _generate_synthetic_identity,
    }

    scenarios = list(scenario_generators.keys())
    weights = [
        0.15,  # impossible_travel
        0.15,  # carding_attack
        0.12,  # high_value
        0.10,  # local_shopping_spree
        0.10,  # account_takeover
        0.08,  # bust_out
        0.08,  # sim_swap
        0.05,  # triangulation
        0.05,  # moto_fraud
        0.04,  # social_engineering
        0.03,  # friendly_fraud
        0.03,  # merchant_collusion
        0.02,  # synthetic_identity
    ]

    chosen_scenario_name = random.choices(scenarios, weights=weights, k=1)[0]
    generator_func = scenario_generators[chosen_scenario_name]

    _, _, masked, issuer = create_card_details()
    return generator_func(base_time, masked, issuer)


def generate_fraud_dataset(output_file, n_legit=100000, n_fraud_events=200):
    start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=90
    )
    legit_transactions = generate_legitimate_transactions(n_legit, start_date)
    df_legit = pd.DataFrame(legit_transactions)
    df_legit["timestamp"] = pd.to_datetime(df_legit["timestamp"])

    all_fraud_transactions = []
    if len(df_legit) < n_fraud_events:
        raise ValueError("Not enough legitimate transactions to generate fraud events.")

    random_timestamps = df_legit.sample(n=n_fraud_events, random_state=42)["timestamp"]

    for ts in tqdm(random_timestamps, desc="Generating Fraud Events"):
        all_fraud_transactions.extend(generate_fraud_scenarios(ts))

    df_fraud = pd.DataFrame(all_fraud_transactions)
    df_fraud["timestamp"] = pd.to_datetime(df_fraud["timestamp"])

    df_final = pd.concat([df_legit, df_fraud], ignore_index=True)
    df_final["timestamp"] = df_final["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv(output_file, index=False)

    print("\n--- Dataset Generation Summary ---")
    print(f"Total Transactions: {len(df_final)}")
    print(f"Legitimate Transactions: {len(df_legit)}")
    print(f"Fraudulent Transactions: {len(df_fraud)}")
    print("\n--- Fraud Scenario Distribution ---")
    print(df_fraud["FRAUD_SCENARIO"].value_counts(normalize=True).round(3))


def feature_engineering(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["card_masked", "timestamp"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
    df["time_since_last_tx_seconds"] = (
        df.groupby("card_masked")["timestamp"].diff().dt.total_seconds()
    )

    df["user_merchant_tx_count"] = df.groupby(["card_masked", "merchant"]).cumcount()
    df["user_has_used_merchant_before"] = (df["user_merchant_tx_count"] > 0).astype(int)

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
        df.groupby("card_masked")["amount"].expanding().mean().reset_index(0, drop=True)
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

    print("Advanced feature engineering complete.")
    return df


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
