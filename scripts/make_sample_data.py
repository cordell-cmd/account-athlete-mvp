from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
np.random.seed(7)

# ----------------------------
# CONFIG
# ----------------------------
N_TOTAL_ACCOUNTS = 1000
DAYS = 200  # number of simulated days for balances + transactions

LOAN_PRODUCTS = ["Consumer Auto", "Commercial CI", "Commercial CRE"]

# Interest rates from 4.00% to 10.00% in 10 bp (0.10%) increments
RATE_VALUES = np.round(np.arange(4.0, 10.0 + 0.1, 0.1), 2)  # 4.0, 4.1, ... 10.0


def sample_loan_balance(product: str, customer_type: str) -> float:
    """Prototype loan outstanding principal by product.

    This is intentionally *not* tied to the deposit balance simulation; it represents loan exposure.
    """
    if product == "Consumer Auto":
        # typical auto exposure
        return float(np.random.uniform(5_000, 45_000))
    if product == "Commercial CI":
        # working capital / equipment
        return float(np.random.uniform(25_000, 600_000))
    # Commercial CRE
    # keep within a demo-friendly range
    base = float(np.random.uniform(200_000, 2_500_000))
    # small bump for business accounts (most CRE is business)
    if customer_type == "business":
        base *= float(np.random.uniform(1.0, 1.2))
    return float(base)


def daterange(start: datetime, days: int):
    for i in range(days):
        yield start + timedelta(days=i)


def sample_days_past_due(n: int) -> np.ndarray:
    """
    Weighted delinquency distribution:
    - Most current/near-current (0–10)
    - Some moderate (11–30)
    - Few serious (31–60)
    - Very few severe (61–90)
    """
    buckets = np.random.choice(
        ["0_10", "11_30", "31_60", "61_90"],
        size=n,
        p=[0.78, 0.14, 0.06, 0.02],
    )
    out = np.zeros(n, dtype=int)
    for i, b in enumerate(buckets):
        if b == "0_10":
            out[i] = int(np.random.randint(0, 11))
        elif b == "11_30":
            out[i] = int(np.random.randint(11, 31))
        elif b == "31_60":
            out[i] = int(np.random.randint(31, 61))
        else:
            out[i] = int(np.random.randint(61, 91))
    return out


def sample_risk_rating_for_product(product: str) -> int:
    """
    Light bias by product (prototype):
    - Consumer Auto: generally lower risk
    - Commercial CI: middle
    - Commercial CRE: higher tail risk
    """
    if product == "Consumer Auto":
        return int(
            np.random.choice(
                range(1, 10),
                p=[0.18, 0.18, 0.16, 0.14, 0.10, 0.08, 0.07, 0.06, 0.03],
            )
        )
    if product == "Commercial CI":
        return int(
            np.random.choice(
                range(1, 10),
                p=[0.10, 0.12, 0.14, 0.14, 0.13, 0.12, 0.10, 0.09, 0.06],
            )
        )
    # Commercial CRE
    return int(
        np.random.choice(
            range(1, 10),
            p=[0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.13, 0.12, 0.11],
        )
    )


def make_account_id(prefix: str, n: int) -> str:
    # prefix "A" or "B"
    return f"{prefix}-{n:04d}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = datetime.today().date()
    start_date = today - timedelta(days=DAYS)

    # ---------------------------------------------------------
    # Keep your original "named" demo accounts
    # ---------------------------------------------------------
    base_accounts = [
        {"account_id": "A-1001", "customer_type": "consumer", "open_date": "2022-04-15"},
        {"account_id": "A-1002", "customer_type": "consumer", "open_date": "2023-01-10"},
        {"account_id": "B-2001", "customer_type": "business",  "open_date": "2020-08-01"},
        {"account_id": "B-2002", "customer_type": "business",  "open_date": "2021-06-20"},
        {"account_id": "B-2003", "customer_type": "business",  "open_date": "2024-03-05"},
    ]

    # Distinct, recognizable trajectories for the base accounts
    base_trend_map = {
        "A-1001": 0.20,   # improving
        "A-1002": -0.30,  # declining
        "B-2001": 0.10,   # mild improving
        "B-2002": -0.20,  # mild declining
        "B-2003": 0.00,   # stable
    }

    # ---------------------------------------------------------
    # Generate remaining accounts to reach N_TOTAL_ACCOUNTS
    # Mix consumer/business, mix has_loan True/False
    # ---------------------------------------------------------
    accounts = list(base_accounts)

    used_ids = set(a["account_id"] for a in accounts)

    remaining = max(0, N_TOTAL_ACCOUNTS - len(accounts))

    # Choose customer types for the new accounts
    cust_types = np.random.choice(["consumer", "business"], size=remaining, p=[0.70, 0.30])

    # Create new accounts with unique IDs that won't collide with base IDs
    # We'll start numeric sequence at 3000 to avoid 1001/2001 overlap.
    next_consumer_num = 3000
    next_business_num = 3000

    for i in range(remaining):
        cust = str(cust_types[i])
        prefix = "A" if cust == "consumer" else "B"

        if prefix == "A":
            while True:
                candidate = make_account_id("A", next_consumer_num)
                next_consumer_num += 1
                if candidate not in used_ids:
                    break
        else:
            while True:
                candidate = make_account_id("B", next_business_num)
                next_business_num += 1
                if candidate not in used_ids:
                    break

        used_ids.add(candidate)

        # random open date in last ~7 years
        open_date = today - timedelta(days=int(np.random.randint(30, 2500)))

        accounts.append(
            {
                "account_id": candidate,
                "customer_type": cust,
                "open_date": str(open_date),
            }
        )

    # ---------------------------------------------------------
    # Assign loan fields with a realistic-ish mix:
    # - Consumers: lower loan incidence
    # - Business: higher loan incidence
    # ---------------------------------------------------------
    for a in accounts:
        cust = a["customer_type"]

        # Keep the base demo accounts loan-enabled so your loan panels have examples
        if a["account_id"] in base_trend_map:
            has_loan = True
        else:
            # adjustable mix:
            # consumers ~35% have loans, business ~55%
            has_loan = bool(np.random.choice([True, False], p=[0.35, 0.65])) if cust == "consumer" else bool(np.random.choice([True, False], p=[0.55, 0.45]))

        a["has_loan"] = has_loan

        if has_loan:
            a["product"] = str(np.random.choice(LOAN_PRODUCTS))
            a["interest_rate"] = float(np.random.choice(RATE_VALUES))  # percent number like 6.7
            a["loan_balance"] = round(sample_loan_balance(a["product"], cust), 2)
            a["days_past_due"] = int(sample_days_past_due(1)[0])
            a["risk_rating"] = int(sample_risk_rating_for_product(a["product"]))

            # mild coupling: worse delinquency nudges risk higher
            if a["days_past_due"] >= 60:
                a["risk_rating"] = int(min(9, a["risk_rating"] + np.random.choice([1, 2], p=[0.6, 0.4])))
            elif a["days_past_due"] >= 30:
                a["risk_rating"] = int(min(9, a["risk_rating"] + np.random.choice([0, 1], p=[0.5, 0.5])))
        else:
            a["product"] = None
            a["interest_rate"] = None
            a["loan_balance"] = None
            a["days_past_due"] = 0
            a["risk_rating"] = None

    accounts_df = pd.DataFrame(accounts)

    # ---------------------------------------------------------
    # Generate transactions + daily balances (same style as before)
    # ---------------------------------------------------------
    txns = []
    balances = []

    # Some accounts get a "shock" drawdown event, especially among declining trends
    shock_prob_consumer = 0.06
    shock_prob_business = 0.08

    for acc in accounts:
        acc_id = acc["account_id"]
        cust_type = acc["customer_type"]

        # starting balance differs
        bal = float(np.random.uniform(800, 8000) if cust_type == "consumer" else np.random.uniform(5000, 45000))

        # Trend:
        # - base accounts use fixed mapping
        # - others: small random drift centered near 0
        if acc_id in base_trend_map:
            trend = float(base_trend_map[acc_id])
        else:
            # Most are stable, some improving/declining
            trend = float(np.clip(np.random.normal(loc=0.0, scale=0.12), -0.35, 0.35))

        # Decide if this account will get a shock day
        shock_day = None
        if acc_id in ["A-1002", "B-2002"]:
            # keep the classic demo shock
            shock_day = start_date + timedelta(days=150)
        else:
            p = shock_prob_consumer if cust_type == "consumer" else shock_prob_business
            # Slightly more likely if trend is negative
            if trend < -0.10:
                p *= 1.6
            if np.random.rand() < p:
                shock_day = start_date + timedelta(days=int(np.random.randint(60, DAYS - 10)))

        for d in daterange(datetime.combine(start_date, datetime.min.time()), DAYS):
            date = d.date()

            # inflow cadence
            inflow = 0.0
            if cust_type == "consumer":
                # biweekly payroll-ish
                if (date - start_date).days % 14 == 0:
                    inflow = float(np.random.normal(1600, 120))
            else:
                # business inflows several days/week
                if np.random.rand() < 0.45:
                    inflow = float(max(0, np.random.normal(2500, 900)))

            # outflow cadence
            outflow = 0.0
            if cust_type == "consumer":
                if np.random.rand() < 0.55:
                    outflow = float(max(0, np.random.normal(120, 55)))
            else:
                if np.random.rand() < 0.65:
                    outflow = float(max(0, np.random.normal(1800, 800)))

            # apply trend by slowly shifting expense/income balance
            # (same behavior as before, just generalized)
            inflow *= (1.0 + max(0.0, trend) * ((date - start_date).days / DAYS))
            outflow *= (1.0 + max(0.0, -trend) * ((date - start_date).days / DAYS))

            # shock: big outflow
            if shock_day and date == shock_day:
                outflow += 0.35 * max(bal, 0) + float(np.random.uniform(600, 1800))

            # record transactions (skip tiny zeros)
            if inflow > 5:
                txns.append(
                    {
                        "account_id": acc_id,
                        "post_date": str(date),
                        "amount": round(inflow, 2),
                        "type": "INFLOW",
                        "description": "Deposit",
                    }
                )
                bal += inflow

            if outflow > 5:
                txns.append(
                    {
                        "account_id": acc_id,
                        "post_date": str(date),
                        "amount": -round(outflow, 2),
                        "type": "OUTFLOW",
                        "description": "Spend/Vendor",
                    }
                )
                bal -= outflow

            # occasional fees/overdraft if negative
            if bal < 0 and np.random.rand() < 0.25:
                txns.append(
                    {
                        "account_id": acc_id,
                        "post_date": str(date),
                        "amount": -35.0,
                        "type": "NSF",
                        "description": "NSF FEE",
                    }
                )
                bal -= 35.0

            # Keep business balances from running too far positive
            if bal > 120000 and cust_type == "business":
                sweep = bal - 90000
                txns.append(
                    {
                        "account_id": acc_id,
                        "post_date": str(date),
                        "amount": -round(sweep, 2),
                        "type": "TRANSFER",
                        "description": "Sweep/Transfer",
                    }
                )
                bal -= sweep

            balances.append({"account_id": acc_id, "date": str(date), "balance": round(bal, 2)})

    txns_df = pd.DataFrame(txns).sort_values(["account_id", "post_date"])
    bal_df = pd.DataFrame(balances).sort_values(["account_id", "date"])

    accounts_path = os.path.join(OUTPUT_DIR, "sample_accounts.csv")
    txns_path = os.path.join(OUTPUT_DIR, "sample_transactions.csv")
    bal_path = os.path.join(OUTPUT_DIR, "sample_daily_balances.csv")

    accounts_df.to_csv(accounts_path, index=False)
    txns_df.to_csv(txns_path, index=False)
    bal_df.to_csv(bal_path, index=False)

    print("Wrote:")
    print(accounts_path)
    print(txns_path)
    print(bal_path)
    print(f"Accounts: {len(accounts_df):,} | Txns: {len(txns_df):,} | Daily balances: {len(bal_df):,}")


if __name__ == "__main__":
    main()
