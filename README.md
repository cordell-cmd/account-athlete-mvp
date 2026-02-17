# Account Athlete MVP

A local Streamlit prototype that reframes bank accounts like athletes:
baseline • load • stress • recovery • trajectory (no ML).

## What it does
Uploads (or uses sample CSVs) to compute per-account metrics and a trajectory label:
- Baseline balance (30d median)
- Volatility (30d std)
- Cash buffer days
- Drawdown events (>=30% drop in 7 days)
- Recovery speed (avg days to recover baseline)
- Overdraft/NSF-like events (90d)
- Trajectory + top reasons

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Generate sample data

```bash
python scripts/make_sample_data.py
```

## Run

```bash
streamlit run app.py
```

## CSV formats

accounts.csv (minimum):
- account_id

transactions.csv (minimum):
- account_id
- post_date (YYYY-MM-DD)
- amount (credits positive, debits negative)

daily_balances.csv (optional but recommended):
- account_id
- date (YYYY-MM-DD)
- balance

## Notes
- If daily_balances.csv is not provided, balances are approximated from transactions (prototype-only).
- This is concept validation tooling. Not intended for automated decisions.

## VS Code AI master prompt (copy/paste)

Paste this as one message to the VS Code AI:

```text
Build a working Streamlit prototype called “Account Athlete MVP” in the existing scaffold.

Goal:
- A local Streamlit app that lets a user upload accounts.csv, transactions.csv, daily_balances.csv (optional).
- It computes and displays athlete-style account metrics and a trajectory label with top reasons.
- It must run out of the box with the included sample data generator (scripts/make_sample_data.py).

Requirements:
- Use only streamlit, pandas, numpy, matplotlib (no seaborn).
- No ML; explainable rules only.
- Handle missing columns gracefully and show helpful errors in the UI.
- Do not invent fields. If a field is missing, fallback or mark metric as unavailable.

Data contracts:
accounts.csv columns (minimum):
- account_id (string)
- customer_type (consumer|business) [optional but recommended]
- open_date (YYYY-MM-DD) [optional]

transactions.csv columns (minimum):
- account_id (string)
- post_date (YYYY-MM-DD)
- amount (float; credits positive, debits negative)
Optional:
- txn_code (string) or type (string) for fee/overdraft detection
- description (string)

daily_balances.csv columns (minimum):
- account_id (string)
- date (YYYY-MM-DD)
- balance (float)

Metrics (per account):
1) baseline_balance_30d: median(daily balance over last 30 days)
2) balance_volatility_30d: std(daily balance over last 30 days)
3) cash_buffer_days: latest_balance / avg_daily_outflow_30d (outflow = abs(sum of negative txns per day))
4) drawdown_count_90d: count of events where balance drops >= 30% within any 7-day window (in last 90 days)
5) avg_recovery_days_90d: average days from drawdown start until balance >= baseline_balance_30d (cap search at 60 days; if never recovers, mark None)
6) overdraft_nsf_90d: count of txns with txn_code/type/description matching ['OD','OVERDRAFT','NSF','RETURNED ITEM'] in last 90 days (case-insensitive)
7) trajectory: Improving/Stable/Declining using transparent rules and computed deltas (compare 30d vs prior 30d when possible).
Also output trajectory_reasons: top 3 human-readable reasons.

UI:
- Sidebar: upload CSVs OR use bundled sample data button.
- Main: account selector dropdown.
- Show player card metrics, balance trend chart (last 90 days), recent transaction summary table.
- Show portfolio table sorted by trajectory severity and risk flags.

Deliver:
- Ensure app.py uses engine/io.py, engine/metrics.py, engine/trajectory.py, engine/explain.py.
- Ensure scripts/make_sample_data.py generates realistic sample CSVs into /data.

Make the code clean and well-commented.
Even if you use the AI, everything below already works; the AI would only refine.
```

---

## Quick start

From the project folder:

```bash
python scripts/make_sample_data.py
streamlit run app.py
```
