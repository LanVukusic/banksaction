CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ,
    amount FLOAT,
    currency VARCHAR(10),
    location VARCHAR(10),
    city VARCHAR(255),
    merchant VARCHAR(255),
    channel VARCHAR(50),
    card_masked VARCHAR(20),
    card_issuer VARCHAR(50),
    TX_FRAUD BOOLEAN,
    FRAUD_SCENARIO VARCHAR(255),
    fraud_probability FLOAT
);

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(128),
    name TEXT,
    transaction UUID
);

-- Optional: Create a table for fraud detection results if you want to store them separately
CREATE TABLE fraud_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(transaction_id),
    fraud_probability FLOAT,
    source_model VARCHAR(20),
    prediction_timestamp TIMESTAMPTZ DEFAULT NOW()
);



