from __future__ import annotations

import pandas as pd


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
	"""Fetch daily OHLCV data from Stooq.

	Returns columns: date, open, high, low, close, volume (when available).
	Raises on network/parse errors.
	"""
	url = STOOQ_DAILY_URL.format(symbol=symbol)
	df = pd.read_csv(url)
	# Stooq uses capitalized headers
	df = df.rename(columns={
		"Date": "date",
		"Open": "open",
		"High": "high",
		"Low": "low",
		"Close": "close",
		"Volume": "volume",
	})
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")
	# Ensure numeric
	for c in ["open", "high", "low", "close", "volume"]:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")
	return df.dropna(subset=["date"]).sort_values("date")


def market_snapshot(symbol: str) -> dict:
	"""Compute a latest quote snapshot from Stooq daily data."""
	df = fetch_stooq_daily(symbol)
	if df.empty or "close" not in df.columns:
		return {"symbol": symbol, "ok": False, "error": "No data."}
	last = df.iloc[-1]
	prev = df.iloc[-2] if len(df) >= 2 else None
	close = float(last["close"]) if pd.notna(last["close"]) else None
	prev_close = float(prev["close"]) if prev is not None and pd.notna(prev["close"]) else None
	change = (close - prev_close) if (close is not None and prev_close is not None) else None
	pct = (change / prev_close * 100.0) if (change is not None and prev_close not in (None, 0.0)) else None
	return {
		"symbol": symbol,
		"ok": True,
		"as_of": last["date"].date().isoformat() if pd.notna(last["date"]) else None,
		"close": close,
		"change": change,
		"pct": pct,
		"series": df[["date", "close"]].copy(),
	}
