package server
package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

// HTTPServer wraps the DB and router
type HTTPServer struct {
	DB *sql.DB
}

// StartHTTPServer starts a simple HTTP API
func StartHTTPServer(db *sql.DB) {
	s := &HTTPServer{DB: db}

	r := chi.NewRouter()
	r.Get("/transactions", s.listTransactions)
	r.Get("/transactions/{id}", s.getTransaction)
	r.Post("/transactions/{id}/label", s.labelTransaction)
	r.Get("/fraud-similarity", s.listSimilarFrauds)

	log.Println("HTTP server listening on :8080")
	http.ListenAndServe(":8080", r)
}

// listTransactions returns recent transactions
func (s *HTTPServer) listTransactions(w http.ResponseWriter, r *http.Request) {
	rows, err := s.DB.Query(`
		SELECT transaction_id, timestamp, amount, currency, city, merchant, fraud_probability
		FROM transactions
		ORDER BY timestamp DESC
		LIMIT 50`)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	type tx struct {
		ID        string   `json:"transaction_id"`
		Timestamp string   `json:"timestamp"`
		Amount    float64  `json:"amount"`
		Currency  *string  `json:"currency"`
		City      *string  `json:"city"`
		Merchant  *string  `json:"merchant"`
		Prob      *float64 `json:"fraud_probability"`
	}
	var txs []tx
	for rows.Next() {
		var t tx
		rows.Scan(&t.ID, &t.Timestamp, &t.Amount, &t.Currency, &t.City, &t.Merchant, &t.Prob)
		txs = append(txs, t)
	}
	json.NewEncoder(w).Encode(txs)
}

// getTransaction returns full transaction details
func (s *HTTPServer) getTransaction(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	row := s.DB.QueryRow(`
		SELECT transaction_id, timestamp, amount, currency, location, city, merchant,
		       channel, card_masked, card_issuer, TX_FRAUD, FRAUD_SCENARIO, fraud_probability
		FROM transactions WHERE transaction_id = $1`, id)

	var t Transaction
	err := row.Scan(
		&t.TransactionID, &t.Timestamp, &t.Amount, &t.Currency, &t.Location,
		&t.City, &t.Merchant, &t.Channel, &t.CardMasked, &t.CardIssuer,
		&t.TxFraud, &t.FraudScenario, &t.TxFraud,
	)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}
	json.NewEncoder(w).Encode(t)
}
// labelTransaction marks the transaction (linked to embedding_id) as fraudulent
// and records it in known_frauds.
func (s *HTTPServer) labelTransaction(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id") // transaction_id in URL

	var req struct {
		EmbeddingID int    `json:"embedding_id"`
		FraudName   string `json:"fraud_name"`
		Description string `json:"description"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", 400)
		return
	}

	tx, err := s.DB.Begin()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	// Ensure embedding belongs to the specified transaction
	var count int
	err = tx.QueryRow(`SELECT COUNT(*) FROM embeddings WHERE id = $1 AND transaction = $2`,
		req.EmbeddingID, id).Scan(&count)
	if err != nil || count == 0 {
		tx.Rollback()
		http.Error(w, "embedding not found for this transaction", 400)
		return
	}

	// Mark transaction as fraudulent
	_, err = tx.Exec(`UPDATE transactions SET TX_FRAUD = TRUE WHERE transaction_id = $1`, id)
	if err != nil {
		tx.Rollback()
		http.Error(w, err.Error(), 500)
		return
	}

	// Add to known_frauds (linking to embedding)
	_, err = tx.Exec(`INSERT INTO known_frauds (embedding_id, fraud_name, description)
	                  VALUES ($1, $2, $3)`, req.EmbeddingID, req.FraudName, req.Description)
	if err != nil {
		tx.Rollback()
		http.Error(w, err.Error(), 500)
		return
	}

	tx.Commit()
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{
		"status":        "labeled",
		"transactionID": id,
	})
}

// listSimilarFrauds shows transactions similar to known frauds using pgvector,
// filtered by a distance threshold and optional limit.
func (s *HTTPServer) listSimilarFrauds(w http.ResponseWriter, r *http.Request) {
	limitStr := r.URL.Query().Get("limit")
	if limitStr == "" {
		limitStr = "50"
	}
	limit, _ := strconv.Atoi(limitStr)

	thresholdStr := r.URL.Query().Get("threshold")
	if thresholdStr == "" {
		thresholdStr = "0.3" // sensible default
	}
	threshold, _ := strconv.ParseFloat(thresholdStr, 64)

	rows, err := s.DB.Query(`
		SELECT
			t.transaction_id,
			kf.fraud_name,
			e.embedding <#> fe.embedding AS distance
		FROM transactions t
		JOIN embeddings e ON e.transaction = t.transaction_id
		JOIN known_frauds kf ON kf.embedding_id IS NOT NULL
		JOIN embeddings fe ON fe.id = kf.embedding_id
		WHERE (e.embedding <#> fe.embedding) <= $1
		ORDER BY distance ASC
		LIMIT $2`,
		threshold, limit)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	type sim struct {
		TransactionID string  `json:"transaction_id"`
		FraudName     string  `json:"fraud_name"`
		Distance      float64 `json:"distance"`
	}
	var sims []sim
	for rows.Next() {
		var s sim
		rows.Scan(&s.TransactionID, &s.FraudName, &s.Distance)
		sims = append(sims, s)
	}
	json.NewEncoder(w).Encode(sims)
}
