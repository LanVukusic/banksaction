package main

import (
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"database/sql"

	_ "github.com/lib/pq"
	"github.com/nats-io/nats.go"
)

// Transaction represents the structure of the transaction data
type Transaction struct {
	TransactionID         int       `json:"TRANSACTION_ID"`
	TxDatetime            time.Time `json:"TX_DATETIME"`
	CustomerID            int       `json:"CUSTOMER_ID"`
	TerminalID            int       `json:"TERMINAL_ID"`
	TxAmount              float64   `json:"TX_AMOUNT"`
	MerchantID            int       `json:"MERCHANT_ID"`
	IsOnline              bool      `json:"IS_ONLINE"`
	TxFraud               bool      `json:"TX_FRAUD"`
	TxFraudScenario       string    `json:"TX_FRAUD_SCENARIO"`
	CustomerIsCompromised bool      `json:"CUSTOMER_IS_COMPROMISED"`
	MerchantIsCompromised bool      `json:"MERCHANT_IS_COMPROMISED"`
	TerminalIsCompromised bool      `json:"TERMINAL_IS_COMPROMISED"`
	TerminalX             float64   `json:"TERMINAL_X"`
	TerminalY             float64   `json:"TERMINAL_Y"`
}

func main() {
	var nc *nats.Conn
	var err error

	// Retry connecting to NATS
	for i := 0; i < 5; i++ {
		nc, err = nats.Connect("nats://nats:4222")
		if err == nil {
			break
		}
		log.Println("Waiting for NATS...")
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		log.Fatal("Could not connect to NATS:", err)
	}
	defer nc.Close()

	var db *sql.DB

	// Retry connecting to PostgreSQL
	connStr := "host=postgres user=user password=password dbname=transactions sslmode=disable"
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
		log.Fatal("Could not connect to PostgreSQL:", err)
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

		// Insert the transaction into the database
		sqlStatement := `
			INSERT INTO transactions (TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, MERCHANT_ID, IS_ONLINE, TX_FRAUD, TX_FRAUD_SCENARIO, CUSTOMER_IS_COMPROMISED, MERCHANT_IS_COMPROMISED, TERMINAL_IS_COMPROMISED, TERMINAL_X, TERMINAL_Y)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`
		_, err = db.Exec(sqlStatement, t.TransactionID, t.TxDatetime, t.CustomerID, t.TerminalID, t.TxAmount, t.MerchantID, t.IsOnline, t.TxFraud, t.TxFraudScenario, t.CustomerIsCompromised, t.MerchantIsCompromised, t.TerminalIsCompromised, t.TerminalX, t.TerminalY)
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
