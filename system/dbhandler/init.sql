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

CREATE TABLE known_frauds (
    id SERIAL PRIMARY KEY,
    embedding_id INT REFERENCES embeddings(id) ON DELETE CASCADE,
    fraud_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE fraud_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(transaction_id),
    fraud_probability FLOAT,
    source_model VARCHAR(20),
    prediction_timestamp TIMESTAMPTZ DEFAULT NOW()
);




CREATE OR REPLACE VIEW transaction_fraud_predictions AS
SELECT
    t.transaction_id,
    t.timestamp,
    t.amount,
    t.currency,
    t.city,
    t.merchant,
    t.channel,
    t.card_masked,
    t.card_issuer,
    fp.fraud_probability,
    fp.source_model,
    fp.prediction_timestamp
FROM transactions t
LEFT JOIN fraud_predictions fp
    ON t.transaction_id = fp.transaction_id;

CREATE OR REPLACE VIEW transaction_fraud_similarity AS
SELECT
    t.transaction_id,
    t.amount,
    t.city,
    t.merchant,
    kf.fraud_name,
    kf.description,
    e.embedding <#> fe.embedding AS distance -- cosine distance
FROM transactions t
JOIN embeddings e ON e.transaction = t.transaction_id
JOIN known_frauds kf ON kf.embedding_id IS NOT NULL
JOIN embeddings fe ON fe.id = kf.embedding_id
ORDER BY distance ASC;
