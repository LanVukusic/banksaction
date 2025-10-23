package server

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
)

// Transaction represents a record in the transactions table
type Transaction struct {
	TransactionID    string   `json:"transaction_id"`
	Timestamp        string   `json:"timestamp"`
	Amount           float64  `json:"amount"`
	Currency         *string  `json:"currency"`
	Location         *string  `json:"location"`
	City             *string  `json:"city"`
	Merchant         *string  `json:"merchant"`
	Channel          *string  `json:"channel"`
	CardMasked       *string  `json:"card_masked"`
	CardIssuer       *string  `json:"card_issuer"`
	TxFraud          int      `json:"TX_FRAUD"`
	FraudScenario    *int     `json:"FRAUD_SCENARIO"`
	FraudProbability *float64 `json:"fraud_probability"`
}

// HTTPServer wraps DB and handlers
type HTTPServer struct {
	DB *sql.DB
}

func StartHTTPServer(db *sql.DB) {
	s := &HTTPServer{DB: db}

	r := chi.NewRouter()

	r.Use(cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		AllowCredentials: true,
		MaxAge:           300,
	}).Handler)

	r.Get("/transactions", s.listTransactions)
	r.Get("/transactions/{id}", s.getTransaction)
	r.Post("/transactions/{id}/label", s.labelTransaction)
	r.Get("/fraud-similarity", s.listSimilarFrauds)
	r.Get("/transactions/stats", s.transactionStats)

	log.Println("HTTP server running on :8080")
	http.ListenAndServe(":8080", r)
}

func (s *HTTPServer) transactionStats(w http.ResponseWriter, r *http.Request) {
	probStr := r.URL.Query().Get("prob")
	prob := 0.5
	if probStr != "" {
		if p, err := strconv.ParseFloat(probStr, 64); err == nil {
			prob = p
		}
	}

	var stats struct {
		Total       int `json:"total"`
		Labeled     int `json:"labeled"`
		HandLabeled int `json:"hand_labeled"`
	}

	_ = s.DB.QueryRow(`SELECT COUNT(*) FROM transactions`).Scan(&stats.Total)
	_ = s.DB.QueryRow(`SELECT COUNT(*) FROM transactions WHERE fraud_probability > $1`, prob).Scan(&stats.Labeled)
	_ = s.DB.QueryRow(`SELECT COUNT(*) FROM known_frauds`).Scan(&stats.HandLabeled)

	json.NewEncoder(w).Encode(stats)
}

func (s *HTTPServer) listTransactions(w http.ResponseWriter, r *http.Request) {
	rows, err := s.DB.Query(`
		SELECT transaction_id, timestamp, amount, currency, city, merchant, fraud_probability
		FROM transactions ORDER BY timestamp DESC LIMIT 50`)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	type Tx struct {
		ID        string   `json:"transaction_id"`
		Timestamp string   `json:"timestamp"`
		Amount    float64  `json:"amount"`
		Currency  *string  `json:"currency"`
		City      *string  `json:"city"`
		Merchant  *string  `json:"merchant"`
		Prob      *float64 `json:"fraud_probability"`
	}
	var txs []Tx
	for rows.Next() {
		var t Tx
		rows.Scan(&t.ID, &t.Timestamp, &t.Amount, &t.Currency, &t.City, &t.Merchant, &t.Prob)
		txs = append(txs, t)
	}
	json.NewEncoder(w).Encode(txs)
}

func (s *HTTPServer) getTransaction(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	row := s.DB.QueryRow(`
		SELECT transaction_id, timestamp, amount, currency, location, city, merchant,
		       channel, card_masked, card_issuer, TX_FRAUD, FRAUD_SCENARIO, fraud_probability
		FROM transactions WHERE transaction_id = $1`, id)

	var t Transaction
	err := row.Scan(
		&t.TransactionID, &t.Timestamp, &t.Amount, &t.Currency, &t.Location, &t.City,
		&t.Merchant, &t.Channel, &t.CardMasked, &t.CardIssuer, &t.TxFraud,
		&t.FraudScenario, &t.FraudProbability,
	)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}
	json.NewEncoder(w).Encode(t)
}

func (s *HTTPServer) labelTransaction(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var req struct {
		FraudName   string  `json:"fraud_name"`
		Description string  `json:"description"`
		EmbeddingID int     `json:"embedding_id"`
		Probability float64 `json:"probability"`
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

	_, err = tx.Exec(`UPDATE transactions SET TX_FRAUD = TRUE, fraud_probability = $1 WHERE transaction_id = $2`,
		req.Probability, id)
	if err != nil {
		tx.Rollback()
		http.Error(w, err.Error(), 500)
		return
	}

	_, err = tx.Exec(`INSERT INTO known_frauds (embedding_id, fraud_name, description) VALUES ($1, $2, $3)`,
		req.EmbeddingID, req.FraudName, req.Description)
	if err != nil {
		tx.Rollback()
		http.Error(w, err.Error(), 500)
		return
	}

	tx.Commit()
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"status": "labeled"})
}

func (s *HTTPServer) listSimilarFrauds(w http.ResponseWriter, r *http.Request) {
	limitStr := r.URL.Query().Get("limit")
	if limitStr == "" {
		limitStr = "20"
	}
	limit, _ := strconv.Atoi(limitStr)

	rows, err := s.DB.Query(`
		SELECT t.transaction_id, kf.fraud_name, (e.embedding <#> fe.embedding) AS distance
		FROM transactions t
		JOIN embeddings e ON e.transaction = t.transaction_id
		JOIN known_frauds kf ON kf.embedding_id IS NOT NULL
		JOIN embeddings fe ON fe.id = kf.embedding_id
		ORDER BY distance ASC
		LIMIT $1`, limit)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	type Sim struct {
		TransactionID string  `json:"transaction_id"`
		FraudName     string  `json:"fraud_name"`
		Distance      float64 `json:"distance"`
	}
	var sims []Sim
	for rows.Next() {
		var s Sim
		rows.Scan(&s.TransactionID, &s.FraudName, &s.Distance)
		sims = append(sims, s)
	}
	json.NewEncoder(w).Encode(sims)
}
