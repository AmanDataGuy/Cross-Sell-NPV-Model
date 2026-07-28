# Data dictionary

Source: UCI Bank Marketing, `bank-full.csv` (see [README.md](README.md)). Types are the actual
pandas dtypes returned by `db.get_modelling_table()`.

## Modelling table (`db.py`)

| Column | Type | Source | Description |
|---|---|---|---|
| `customer_id` | int | derived | Sequential ID assigned by `load.py` in original row order (not present in the raw CSV) — also recovers chronological contact order for `backtest.py`. |
| `age` | int | raw | Customer age in years. |
| `job` | str (categorical) | raw | Job type — 12 categories (e.g. `management`, `blue-collar`, `student`, `retired`). |
| `marital` | str (categorical) | raw | `married`, `single`, or `divorced` (includes widowed). |
| `education` | str (categorical) | raw | `primary`, `secondary`, `tertiary`, or `unknown`. |
| `default` | str (`yes`/`no`) | raw | Has credit in default. |
| `balance` | int | raw | Average yearly account balance, in euros. Can be negative (overdraft). Used as the revenue proxy in `assumptions.py`. |
| `housing` | str (`yes`/`no`) | raw | Has a housing loan. |
| `loan` | str (`yes`/`no`) | raw | Has a personal loan. |
| `contact` | str (categorical) | raw | Contact channel for the last call: `cellular`, `telephone`, or `unknown`. |
| `day` | int | raw | Day of the month of the last contact (1-31). |
| `month` | str (categorical) | raw | Month of the last contact (`jan`-`dec`). |
| `campaign` | int | raw | Number of contacts made to this customer during this campaign (includes the last one). |
| `pdays` | int | raw | Days since the customer was last contacted in a *previous* campaign; `-1` means never contacted before. |
| `previous` | int | raw | Number of contacts made to this customer before this campaign. |
| `poutcome` | str (categorical) | raw | Outcome of the previous campaign: `success`, `failure`, `other`, or `unknown`. |
| `y` | int (0/1) | derived | Target — subscribed to a term deposit. Recoded from the raw `yes`/`no` string. |

**Excluded on purpose**: `duration` (last-contact call length, seconds) exists in `campaign_history`
but is dropped from the modelling table — it's only known *after* the call happens, so including
it would leak the outcome. See `db.py`'s `MODELLING_QUERY`.

## Pipeline outputs (not in the raw data — derived by this repo)

| Column | Type | Written by | Description |
|---|---|---|---|
| `propensity` | float (0-1) | `model.py` | Calibrated probability of subscribing, from the recommended model (GBM in the reference run). |
| `tier` | str (`High`/`Medium`/`Low`) | `segment.py` | Decision-tree segment, ordered by average propensity. |
| `expected_npv` | float ($) | `segment.py` | `propensity x expected_revenue(balance) - campaign_cost`, per customer (see `assumptions.py`). |
