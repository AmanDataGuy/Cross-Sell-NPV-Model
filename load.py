"""
Load: bank-full.csv -> SQLite (customers, campaign_history).

One customer-contact per row in the source CSV; split into a customer
profile table and a this-campaign-contact table so there's a real join,
then sanity-check the joined modelling table (via db.py) before anything
downstream touches it.
"""
from pathlib import Path

import pandas as pd

import sqlite3

from db import DB_PATH, get_modelling_table, get_subscription_rate_by_job

RAW_CSV = Path("data/raw/bank-full.csv")

CUSTOMER_COLS = ["customer_id", "age", "job", "marital", "education", "default", "balance", "housing", "loan"]
CAMPAIGN_COLS = ["customer_id", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]


def main() -> None:
    df = pd.read_csv(RAW_CSV, sep=";")
    df.insert(0, "customer_id", range(1, len(df) + 1))

    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df[CUSTOMER_COLS].to_sql("customers", conn, if_exists="replace", index=False)
        df[CAMPAIGN_COLS].to_sql("campaign_history", conn, if_exists="replace", index=False)

    modelling = get_modelling_table()
    assert "duration" not in modelling.columns, "leakage column duration reached the modelling table"
    print(f"rows={len(modelling)}  positive_rate={modelling['y'].mean():.3%}")
    print("\nSubscription rate by job (SQL GROUP BY, first look before modelling):")
    print(get_subscription_rate_by_job().to_string(index=False))


if __name__ == "__main__":
    main()
