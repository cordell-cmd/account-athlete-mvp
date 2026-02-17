from __future__ import annotations

import pandas as pd

REQUIRED_ACCOUNTS = ["account_id"]
REQUIRED_TXNS = ["account_id", "post_date", "amount"]
REQUIRED_BAL = ["account_id", "date", "balance"]


def _ensure_cols(df: pd.DataFrame, required: list[str], name: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"{name} is missing required columns: {missing}")


def load_accounts(csv_bytes) -> pd.DataFrame:
	df = pd.read_csv(csv_bytes)
	_ensure_cols(df, REQUIRED_ACCOUNTS, "accounts.csv")
	df["account_id"] = df["account_id"].astype(str)
	# Optional columns
	if "customer_type" in df.columns:
		df["customer_type"] = df["customer_type"].astype(str).str.lower()
	return df


def load_transactions(csv_bytes) -> pd.DataFrame:
	df = pd.read_csv(csv_bytes)
	_ensure_cols(df, REQUIRED_TXNS, "transactions.csv")
	df["account_id"] = df["account_id"].astype(str)
	df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
	df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
	df = df.dropna(subset=["post_date", "amount"])
	# Optional text fields
	for col in ["type", "txn_code", "description"]:
		if col in df.columns:
			df[col] = df[col].astype(str)
	return df


def load_balances(csv_bytes) -> pd.DataFrame:
	df = pd.read_csv(csv_bytes)
	_ensure_cols(df, REQUIRED_BAL, "daily_balances.csv")
	df["account_id"] = df["account_id"].astype(str)
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
	df = df.dropna(subset=["date", "balance"])
	return df


def derive_balances_from_transactions(txns: pd.DataFrame, start_balance: float = 0.0) -> pd.DataFrame:
	"""
	If daily balances aren’t provided, approximate them by cumulative sum of transactions per day.
	This is ONLY for prototyping.
	"""
	if txns.empty:
		return pd.DataFrame(columns=["account_id", "date", "balance"])

	df = txns.copy()
	df["date"] = df["post_date"].dt.floor("D")
	daily = (
		df.groupby(["account_id", "date"], as_index=False)["amount"]
		.sum()
		.sort_values(["account_id", "date"])
	)
	out_rows = []
	for acc_id, g in daily.groupby("account_id"):
		bal = start_balance
		for _, row in g.iterrows():
			bal += float(row["amount"])
			out_rows.append({"account_id": acc_id, "date": row["date"], "balance": bal})
	return pd.DataFrame(out_rows)
