"""
Shared SQLite access for the modelling table. `load.py` populates the DB;
every other script (model.py, hypothesis.py, segment.py, backtest.py) reads
through `get_modelling_table()` so the join/leakage-exclusion logic lives
in exactly one place.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("db/crosssell.db")

# `duration` (call length) is excluded here on purpose: it's only known
# after the call happens, so it leaks the outcome. It stays in
# campaign_history for anyone who wants to look, just not in this query.
MODELLING_QUERY = """
SELECT
    c.customer_id, c.age, c.job, c.marital, c.education, c."default",
    c.balance, c.housing, c.loan,
    h.contact, h.day, h.month, h.campaign, h.pdays, h.previous, h.poutcome,
    CASE WHEN h.y = 'yes' THEN 1 ELSE 0 END AS y
FROM customers c
JOIN campaign_history h USING (customer_id)
"""


def get_modelling_table() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(MODELLING_QUERY, conn)


# A pure-SQL aggregation (GROUP BY + computed rate), separate from the join
# above, so there's a real first-look-at-the-data query alongside the
# feature-assembly one - not everything has to go through pandas.
SUBSCRIPTION_RATE_BY_JOB_QUERY = """
SELECT c.job,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(CASE WHEN h.y = 'yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS subscription_rate_pct
FROM customers c
JOIN campaign_history h USING (customer_id)
GROUP BY c.job
ORDER BY subscription_rate_pct DESC
"""


def get_subscription_rate_by_job() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(SUBSCRIPTION_RATE_BY_JOB_QUERY, conn)
