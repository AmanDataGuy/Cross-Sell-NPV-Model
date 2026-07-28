# Data

Source: UCI Bank Marketing — https://archive.ics.uci.edu/dataset/222/bank+marketing

`data/raw/` (not committed to git — regenerate by re-downloading if missing):
- `bank-full.csv` — 45,211 customer contacts, 17 columns. Used over the `bank-additional-full.csv`
  variant in the same archive because it includes `balance` (account balance), the only usable
  revenue proxy for the NPV model. The `bank-additional-*` files, the 10%-sample CSVs, and
  macOS/R clutter from the original download were removed as unused.
- `bank-names.txt` — UCI's data dictionary for this file (column definitions).

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the modelling table's own columns (types,
raw vs. derived, and what's deliberately excluded).

`duration` (last-contact call length) is a known leakage column — it's only known after the
outcome, so `db.py`'s modelling-table query excludes it.
