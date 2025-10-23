package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	_ "github.com/lib/pq"
	"github.com/nats-io/nats.go"
)

// Transaction represents the cleaned transaction data for the new schema
type Transaction struct {
	TransactionID string  `json:"transaction_id"`
	Timestamp     string  `json:"timestamp"` // ISO format string
	Amount        float64 `json:"amount"`
	Currency      *string `json:"currency"` // Pointer for null
	Location      *string `json:"location"` // Pointer for null
	City          *string `json:"city"`     // Pointer for null
	Merchant      *string `json:"merchant"` // Pointer for null
	Channel       *string `json:"channel"`  // Pointer for null
	CardMasked    *string `json:"card_masked"`
	CardIssuer    *string `json:"card_issuer"`
	TxFraud       int     `json:"TX_FRAUD"`
	FraudScenario int     `json:"FRAUD_SCENARIO"`
}

// parseTime parses ISO 8601 format string to time.Time
func parseTime(timeStr string) (time.Time, error) {
	// Try different time formats that might come from Python
	formats := []string{
		"2006-01-02T15:04:05.999999Z",
		"2006-01-02T15:04:05Z07:00",
		time.RFC3339,
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
	}

	for _, format := range formats {
		t, err := time.Parse(format, timeStr)
		if err == nil {
			return t, nil
		}
	}

	return time.Time{}, &time.ParseError{
		Layout:     "multiple formats attempted",
		Value:      timeStr,
		LayoutElem: "",
		ValueElem:  "",
		Message:    "no suitable format found",
	}
}

// nullStringToSQL converts *string to sql.NullString
func nullStringToSQL(s *string) sql.NullString {
	if s == nil {
		return sql.NullString{Valid: false}
	}
	return sql.NullString{String: *s, Valid: true}
}

func initDb() (*sql.DB, error) {
	var db *sql.DB
	var err error

	// Connect directly to the default postgres database
	connStr := "host=localhost user=user password=password dbname=transactions sslmode=disable"
	for i := 0; i < 5; i++ {
		db, err = sql.Open("postgres", connStr)
		if err == nil {
			err = db.Ping()
			if err == nil {
				break
			}
		}
		log.Println("Waiting for PostgreSQL...")
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		return nil, err
	}

	// Read and execute the init.sql file directly on the postgres database
	c, err := os.ReadFile("init.sql")
	if err != nil {
		return nil, err
	}
	sql := string(c)
	_, err = db.Exec(sql)
	if err != nil {
		// If the table already exists, we can ignore the error
		if !strings.Contains(err.Error(), "already exists") {
			return nil, err
		}
	} else {
		log.Println("Table 'transactions' created.")
	}

	return db, nil
}

func main() {
	log.Printf("starting...")
	var nc *nats.Conn
	var err error

	// Retry connecting to NATS
	for range 5 {
		nc, err = nats.Connect("nats://localhost:4222")
		if err == nil {
			break
		}
		log.Println("Waiting for NATS...")
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		log.Fatal("Could not connect to NATS:", err)
	}
	log.Println("Connected to NATS")
	defer nc.Close()

	db, err := initDb()
	if err != nil {
		log.Fatal("Database initialization failed:", err)
	} else {
		log.Println("Database initialized")
	}
	defer db.Close()

	// Subscribe to the 'transactions' topic
	sub, err := nc.Subscribe("transactions", func(msg *nats.Msg) {
		var t Transaction
		err := json.Unmarshal(msg.Data, &t)
		if err != nil {
			log.Println("Error unmarshalling transaction:", err)
			return
		}

		// Parse the time string to time.Time
		parsedTime, err := parseTime(t.Timestamp)
		if err != nil {
			log.Println("Error parsing time:", err, "time string:", t.Timestamp)
			return
		}

		// Convert null values to SQL null types
		currency := nullStringToSQL(t.Currency)
		location := nullStringToSQL(t.Location)
		city := nullStringToSQL(t.City)
		merchant := nullStringToSQL(t.Merchant)
		channel := nullStringToSQL(t.Channel)
		cardMasked := nullStringToSQL(t.CardMasked)
		cardIssuer := nullStringToSQL(t.CardIssuer)

		// Insert the transaction into the database
		sqlStatement := `
			INSERT INTO transactions (transaction_id, timestamp, amount, currency, location, city, merchant, channel, card_masked, card_issuer, TX_FRAUD, FRAUD_SCENARIO)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`
		_, err = db.Exec(sqlStatement,
			t.TransactionID,
			parsedTime,
			t.Amount,
			currency,
			location,
			city,
			merchant,
			channel,
			cardMasked,
			cardIssuer,
			t.TxFraud,
			t.FraudScenario)
		if err != nil {
			log.Println("Error inserting transaction:", err)
			return
		}

		log.Printf("Stored transaction with ID: %s", t.TransactionID)
	})
	if err != nil {
		log.Fatal(err)
	}
	defer sub.Unsubscribe()

	log.Println("Subscribed to 'transactions' topic.")

	// Wait for a signal to exit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
}
