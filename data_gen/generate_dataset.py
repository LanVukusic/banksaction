# Necessary imports for this notebook
import os
import numpy as np
import pandas as pd
import datetime
import time
import random
from pandarallel import pandarallel
pandarallel.initialize()

def generate_customer_profiles_table(n_customers, random_state=0):

    np.random.seed(random_state)

    customer_id_properties = []

    # Generate customer properties from random distributions
    for customer_id in range(n_customers):

        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        mean_amount = np.random.uniform(5, 100)  # Arbitrary (but sensible) value
        std_amount = mean_amount / 2  # Arbitrary (but sensible) value

        mean_nb_tx_per_day = np.random.uniform(0, 4)  # Arbitrary (but sensible) value
        
        # Add fraud propensity - some customers are more likely to commit fraud
        fraud_probability = np.random.uniform(0, 0.1)  # 0-10% chance of fraud per transaction

        customer_id_properties.append(
            [
                customer_id,
                x_customer_id,
                y_customer_id,
                mean_amount,
                std_amount,
                mean_nb_tx_per_day,
                fraud_probability
            ]
        )

    customer_profiles_table = pd.DataFrame(
        customer_id_properties,
        columns=[
            "CUSTOMER_ID",
            "x_customer_id",
            "y_customer_id",
            "mean_amount",
            "std_amount",
            "mean_nb_tx_per_day",
            "fraud_probability"
        ],
    )

    return customer_profiles_table


def generate_merchants_profiles_table(
    n_merchants, online_chance=20, store_chance=80, random_state=0
):

    np.random.seed(random_state)

    merchant_id_properties = []

    # Generate merchant properties from random distributions
    for merchant_id in range(n_merchants):

        is_online = 1 if np.random.uniform(0, 100) < online_chance else 0
        is_store = 1 if np.random.uniform(0, 100) < store_chance else 0

        merchant_id_properties.append([merchant_id, is_online, is_store])

    merchant_profiles_table = pd.DataFrame(
        merchant_id_properties, columns=["MERCHANT_ID", "ONLINE", "STORES"]
    )

    return merchant_profiles_table


def generate_terminal_profiles_table(n_terminals, merchants, random_state=0):

    np.random.seed(random_state)

    # Find merchants with STORES set to 1
    store_merchants = merchants[merchants['STORES'] == 1]
    
    terminal_id_properties = []

    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        
        # Associate terminal with a random merchant that has STORES=1
        if len(store_merchants) > 0:
            associated_merchant = store_merchants.sample(n=1, random_state=terminal_id).iloc[0]['MERCHANT_ID']
        else:
            # If no merchants with STORES=1, assign a random merchant
            associated_merchant = merchants.sample(n=1, random_state=terminal_id).iloc[0]['MERCHANT_ID']

        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id, associated_merchant])

    terminal_profiles_table = pd.DataFrame(
        terminal_id_properties,
        columns=["TERMINAL_ID", "x_terminal_id", "y_terminal_id", "MERCHANT_ID"],
    )

    return terminal_profiles_table


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):

    # Use numpy arrays in the following to speed up computations

    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[["x_customer_id", "y_customer_id"]].values.astype(
        float
    )

    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals


def generate_transactions_table(customer_profile, start_date="2024-04-01", nb_days=10, merchants_df=None, terminals_df=None):
    """
    Generate transactions for a customer with compromised entities system.
    Each day, some entities become compromised and some become uncompromised.
    Fraudulent transactions are generated only when involving compromised entities.
    """
    customer_transactions = []

    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    # Initialize sets to track compromised entities
    compromised_customers = set()
    compromised_merchants = set()
    compromised_terminals = set()

    # For all days
    for day in range(nb_days):
        # Each day, some entities become compromised and some become uncompromised
        
        # Randomly select entities to become compromised today
        if random.random() < 0.02:  # 2% chance a customer becomes compromised (reduced from 10%)
            if random.random() < customer_profile.fraud_probability:  # Use original fraud probability
                compromised_customers.add(customer_profile.CUSTOMER_ID)
        
        # Randomly select merchants to become compromised
        if random.random() < 0.01:  # 1% chance a merchant becomes compromised (reduced from 5%)
            # Select a random merchant to compromise
            random_merchant = merchants_df.sample(n=1).iloc[0]['MERCHANT_ID']
            compromised_merchants.add(random_merchant)
        
        # Randomly select terminals to become compromised
        if random.random() < 0.005:  # 0.5% chance a terminal becomes compromised (reduced from 2%)
            if len(customer_profile.available_terminals) > 0:
                # Select a random terminal to compromise
                random_terminal = random.choice(customer_profile.available_terminals)
                compromised_terminals.add(random_terminal)

        # Randomly select entities to become uncompromised (recover)
        if random.random() < 0.3:  # 30% chance a compromised customer becomes uncompromised (increased from 15%)
            if customer_profile.CUSTOMER_ID in compromised_customers:
                compromised_customers.discard(customer_profile.CUSTOMER_ID)
        
        if random.random() < 0.25:  # 25% chance a compromised merchant becomes uncompromised (increased from 10%)
            if len(compromised_merchants) > 0:
                # Select a random compromised merchant to uncompromise
                random_merchant = random.choice(list(compromised_merchants))
                compromised_merchants.discard(random_merchant)
        
        if random.random() < 0.2: # 20% chance a compromised terminal becomes uncompromised (increased from 8%)
            if len(compromised_terminals) > 0:
                # Select a random compromised terminal to uncompromised
                random_terminal = random.choice(list(compromised_terminals))
                compromised_terminals.discard(random_terminal)

        # Random number of transactions for that day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

        # If nb_tx positive, let us generate transactions
        if nb_tx > 0:

            for tx in range(nb_tx):
                # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that
                # most transactions occur during the day.
                time_tx = int(np.random.normal(86400 / 2, 20000))

                # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                if (time_tx > 0) and (time_tx < 86400):

                    # Decide if transaction is online (20% chance) or terminal-based (80% chance)
                    is_online = np.random.uniform(0, 100) < 20 # 20% chance of online transaction

                    # Determine if this transaction should be fraudulent
                    # A transaction is fraudulent if the customer, merchant, or terminal is compromised
                    is_fraud = False
                    fraud_scenario = 0
                    
                    # Check if any involved entity is compromised
                    customer_is_compromised = customer_profile.CUSTOMER_ID in compromised_customers
                    
                    # For this transaction, we'll decide if it should be fraudulent based on compromised entities
                    # If the customer is compromised, any transaction they make could be fraudulent
                    if customer_is_compromised:
                        is_fraud = True
                        fraud_scenario = np.random.choice([1, 3, 5])  # Customer-related fraud scenarios

                    if is_online:
                        # Select a random online merchant
                        online_merchants = merchants_df[merchants_df['ONLINE'] == 1]
                        if len(online_merchants) > 0:
                            selected_merchant = online_merchants.sample(n=1).iloc[0]['MERCHANT_ID']
                            # Check if merchant is compromised
                            if selected_merchant in compromised_merchants:
                                is_fraud = True
                                fraud_scenario = 3  # Card not present fraud
                            # For online transactions, we don't use a terminal
                            terminal_id = None
                        else:
                            # Fallback: if no online merchants, use any merchant and a terminal
                            selected_merchant = merchants_df.sample(n=1).iloc[0]['MERCHANT_ID']
                            # Check if merchant is compromised
                            if selected_merchant in compromised_merchants:
                                is_fraud = True
                                fraud_scenario = 3  # Card not present fraud
                            if len(customer_profile.available_terminals) > 0:
                                terminal_id = random.choice(customer_profile.available_terminals)
                                # Check if terminal is compromised
                                if terminal_id in compromised_terminals:
                                    is_fraud = True
                                    fraud_scenario = 2 # Geographic fraud
                            else:
                                terminal_id = None
                    else:
                        # Terminal-based transaction
                        if len(customer_profile.available_terminals) > 0:
                            terminal_id = random.choice(customer_profile.available_terminals)
                            
                            # Check if terminal is compromised
                            terminal_is_compromised = terminal_id in compromised_terminals
                            if terminal_is_compromised:
                                is_fraud = True
                                fraud_scenario = 2  # Geographic fraud
                            
                            # Get the merchant associated with this terminal
                            selected_merchant = terminals_df[terminals_df['TERMINAL_ID'] == terminal_id].iloc[0]['MERCHANT_ID']
                            
                            # Check if merchant is compromised
                            if selected_merchant in compromised_merchants:
                                is_fraud = True
                                fraud_scenario = np.random.choice([2, 4])  # Terminal/merchant-related fraud
                        else:
                            # If no terminals available, we can't create a terminal-based transaction
                            continue

                    # Amount is drawn from a normal distribution
                    amount = np.random.normal(
                        customer_profile.mean_amount, customer_profile.std_amount
                    )

                    # If amount negative, draw from a uniform distribution
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)

                    amount = np.round(amount, decimals=2)

                    # If this is a fraud transaction, potentially modify the amount or other properties
                    if is_fraud:
                        if fraud_scenario == 1: # Unusual amount fraud
                            # Make the amount significantly higher than normal
                            amount = np.random.normal(customer_profile.mean_amount * 3, customer_profile.std_amount * 2)
                            if amount < customer_profile.mean_amount * 2:
                                amount = customer_profile.mean_amount * 2 + np.random.uniform(0, 50)
                        elif fraud_scenario == 5:  # Testing fraud
                            # Make the amount small
                            amount = np.random.uniform(0.5, 5.0)
                        # Other scenarios might modify other properties

                    # Add transaction to list
                    # Check if entities were compromised for this transaction
                    customer_compromised = customer_profile.CUSTOMER_ID in compromised_customers
                    merchant_compromised = selected_merchant in compromised_merchants
                    terminal_compromised = terminal_id in compromised_terminals if terminal_id is not None else False
                    
                    # Get terminal coordinates if not online
                    terminal_x = None
                    terminal_y = None
                    if not is_online and terminal_id is not None:
                        terminal_row = terminals_df[terminals_df['TERMINAL_ID'] == terminal_id].iloc[0]
                        terminal_x = terminal_row['x_terminal_id']
                        terminal_y = terminal_row['y_terminal_id']
                    
                    customer_transactions.append(
                        [
                            customer_profile.CUSTOMER_ID,
                            terminal_id,
                            amount,
                            selected_merchant,
                            is_online,
                            1 if is_fraud else 0,  # TX_FRAUD
                            fraud_scenario,  # TX_FRAUD_SCENARIO
                            customer_compromised,  # CUSTOMER_IS_COMPROMISED
                            merchant_compromised,  # MERCHANT_IS_COMPROMISED
                            terminal_compromised,  # TERMINAL_IS_COMPROMISED
                            terminal_x,  # TERMINAL_X
                            terminal_y,  # TERMINAL_Y
                            time_tx + day * 86400,  # TX_TIME_SECONDS (needed for datetime conversion)
                            day  # TX_TIME_DAYS (needed for datetime conversion)
                        ]
                    )

    customer_transactions = pd.DataFrame(
        customer_transactions,
        columns=[
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "MERCHANT_ID",
            "IS_ONLINE",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
            "CUSTOMER_IS_COMPROMISED",
            "MERCHANT_IS_COMPROMISED",
            "TERMINAL_IS_COMPROMISED",
            "TERMINAL_X",
            "TERMINAL_Y",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS"
        ],
    )

    if len(customer_transactions) > 0:
        customer_transactions["TX_DATETIME"] = pd.to_datetime(
            customer_transactions["TX_TIME_SECONDS"], unit="s", origin=start_date
        )
        # Remove TX_TIME_SECONDS and TX_TIME_DAYS from the final output
        customer_transactions = customer_transactions[
            [
                "TX_DATETIME",
                "CUSTOMER_ID",
                "TERMINAL_ID",
                "TX_AMOUNT",
                "MERCHANT_ID",
                "IS_ONLINE",
                "TX_FRAUD",
                "TX_FRAUD_SCENARIO",
                "CUSTOMER_IS_COMPROMISED",
                "MERCHANT_IS_COMPROMISED",
                "TERMINAL_IS_COMPROMISED",
                "TERMINAL_X",
                "TERMINAL_Y"
            ]
        ]

    return customer_transactions


def generate_dataset(
    n_customers=10000, n_terminals=10000, n_merchants=1000, nb_days=180, start_date="2024-04-01", r=5
):

    start_time = time.time()
    customer_profiles_table = generate_customer_profiles_table(
        n_customers, random_state=0
    )
    print(
        "Time to generate customer profiles table: {0:.2}s".format(
            time.time() - start_time
        )
    )

    start_time = time.time()
    merchant_profiles_table = generate_merchants_profiles_table(
        n_merchants, random_state=0
    )
    print(
        "Time to generate merchant profiles table: {0:.2}s".format(
            time.time() - start_time
        )
    )

    start_time = time.time()
    terminal_profiles_table = generate_terminal_profiles_table(
        n_terminals, merchant_profiles_table, random_state=1
    )
    print(
        "Time to generate terminal profiles table: {0:.2}s".format(
            time.time() - start_time
        )
    )

    start_time = time.time()
    x_y_terminals = terminal_profiles_table[
        ["x_terminal_id", "y_terminal_id"]
    ].values.astype(float)
    # customer_profiles_table["available_terminals"] = customer_profiles_table.apply(
    #     lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r),
    #     axis=1,
    # )
    # With Pandarallel
    customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table["nb_terminals"] = (
        customer_profiles_table.available_terminals.apply(len)
    )
    print(
        "Time to associate terminals to customers: {0:.2}s".format(
            time.time() - start_time
        )
    )

    start_time = time.time()
    # transactions_df = (
    #     customer_profiles_table.groupby("CUSTOMER_ID")
    #     .apply(lambda x: generate_transactions_table(x.iloc[0], nb_days=nb_days))
    #     .reset_index(drop=True)
    # )
    # With Pandarallel
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days, merchants_df=merchant_profiles_table, terminals_df=terminal_profiles_table)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(time.time() - start_time))

    # Sort transactions chronologically
    transactions_df = transactions_df.sort_values("TX_DATETIME")
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True, drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns={"index": "TRANSACTION_ID"}, inplace=True)

    return (customer_profiles_table, terminal_profiles_table, transactions_df)


# Generate the dataset with fraud scenarios integrated
print("Generating dataset with fraud scenarios integrated...")
(customer_profiles_table, terminal_profiles_table, transactions_df) = generate_dataset(
    n_customers=5000, n_terminals=10000, nb_days=180, start_date="2018-04-01", r=5
)

# Save the tables as parquet files
print("Saving tables as parquet files...")
customer_profiles_table.to_parquet('customer_profiles.parquet', index=False)
terminal_profiles_table.to_parquet('terminal_profiles.parquet', index=False)
transactions_df.to_parquet('transactions.parquet', index=False)

print("Dataset generation complete!")
print(f"Total transactions: {len(transactions_df)}")
print(f"Fraudulent transactions: {transactions_df['TX_FRAUD'].sum()}")
print(f"Fraud rate: {transactions_df['TX_FRAUD'].mean()*100:.2f}%")
