import datetime
import random
import pandas as pd
import numpy as np
import uuid
from faker import Faker
from tqdm import tqdm

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
