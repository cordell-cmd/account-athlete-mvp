from __future__ import annotations

import io
import os
import ssl
import urllib.request
from urllib.parse import quote

import pandas as pd


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def _encode_stooq_symbol(symbol: str) -> str:
	# Stooq uses symbols like "^spx" and "spy.us". We URL-encode the caret.
	return quote(symbol.strip(), safe=".-").lower()


def _fetch_url_bytes(url: str) -> bytes:
	try:
		with urllib.request.urlopen(url, timeout=20) as resp:
			return resp.read()
	except Exception:
		# Some local networks (e.g., SSL interception) can break cert validation.
		# Allow an explicit opt-in fallback for local dev.
		if os.getenv("STOOQ_INSECURE_SSL") == "1":
			ctx = ssl._create_unverified_context()
			with urllib.request.urlopen(url, context=ctx, timeout=20) as resp:
				return resp.read()
		raise


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
	"""Fetch daily OHLCV data from Stooq.

	Returns columns: date, open, high, low, close, volume (when available).
	Raises on network/parse errors.
	"""
	url = STOOQ_DAILY_URL.format(symbol=_encode_stooq_symbol(symbol))
	try:
		df = pd.read_csv(url)
	except Exception:
		data = _fetch_url_bytes(url)
		df = pd.read_csv(io.BytesIO(data))

	# If Stooq returns an error page/HTML, pandas may still return a "df".
	# Fail closed instead of showing bogus index levels.
	expected = {"Date", "Open", "High", "Low", "Close"}
	if not expected.issubset(set(df.columns)):
		raise ValueError(f"Unexpected Stooq response for {symbol} (columns={list(df.columns)[:8]}).")
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
