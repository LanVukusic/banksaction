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

// Transaction represents the cleaned transaction data
type Transaction struct {
	TransactionID         int      `json:"TRANSACTION_ID"`
	TxDatetime            string   `json:"TX_DATETIME"` // ISO format string
	CustomerID            *int     `json:"CUSTOMER_ID"` // Pointer for null
	TerminalID            *int     `json:"TERMINAL_ID"` // Pointer for null
	TxAmount              float64  `json:"TX_AMOUNT"`
	MerchantID            *int     `json:"MERCHANT_ID"` // Pointer for null
	IsOnline              bool     `json:"IS_ONLINE"`
	TxFraud               int      `json:"TX_FRAUD"`
	TxFraudScenario       int      `json:"TX_FRAUD_SCENARIO"`
	CustomerIsCompromised bool     `json:"CUSTOMER_IS_COMPROMISED"`
	MerchantIsCompromised bool     `json:"MERCHANT_IS_COMPROMISED"`
	TerminalIsCompromised bool     `json:"TERMINAL_IS_COMPROMISED"`
	TerminalX             *float64 `json:"TERMINAL_X"` // Pointer for null
	TerminalY             *float64 `json:"TERMINAL_Y"` // Pointer for null
}

// parseTime parses ISO 8601 format string to time.Time
func parseTime(timeStr string) (time.Time, error) {
	// Try different time formats that might come from Python
	formats := []string{
		"2006-01-02T15:04:05",
		"2006-01-02T15:04:05Z07:00",
		time.RFC3339,
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

// nullIntToSQL converts *int to sql.NullInt64
func nullIntToSQL(i *int) sql.NullInt64 {
	if i == nil {
		return sql.NullInt64{Valid: false}
	}
	return sql.NullInt64{Int64: int64(*i), Valid: true}
}

// nullFloatToSQL converts *float64 to sql.NullFloat64
func nullFloatToSQL(f *float64) sql.NullFloat64 {
	if f == nil {
		return sql.NullFloat64{Valid: false}
	}
	return sql.NullFloat64{Float64: *f, Valid: true}
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
		parsedTime, err := parseTime(t.TxDatetime)
		if err != nil {
			log.Println("Error parsing time:", err, "time string:", t.TxDatetime)
			return
		}

		// Convert null values to SQL null types
		customerID := nullIntToSQL(t.CustomerID)
		terminalID := nullIntToSQL(t.TerminalID)
		merchantID := nullIntToSQL(t.MerchantID)
		terminalX := nullFloatToSQL(t.TerminalX)
		terminalY := nullFloatToSQL(t.TerminalY)

		// Insert the transaction into the database
		sqlStatement := `
			INSERT INTO transactions (TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, MERCHANT_ID, IS_ONLINE, TX_FRAUD, TX_FRAUD_SCENARIO, CUSTOMER_IS_COMPROMISED, MERCHANT_IS_COMPROMISED, TERMINAL_IS_COMPROMISED, TERMINAL_X, TERMINAL_Y)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`
		_, err = db.Exec(sqlStatement,
			t.TransactionID,
			parsedTime,
			customerID,
			terminalID,
			t.TxAmount,
			merchantID,
			t.IsOnline,
			t.TxFraud,
			t.TxFraudScenario,
			t.CustomerIsCompromised,
			t.MerchantIsCompromised,
			t.TerminalIsCompromised,
			terminalX,
			terminalY)
		if err != nil {
			log.Println("Error inserting transaction:", err)
			return
		}

		log.Printf("Stored transaction with ID: %d", t.TransactionID)
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
