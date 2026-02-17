from __future__ import annotations

import io
import html as html_lib
import os
import re
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


def _strip_tags(s: str) -> str:
	s = re.sub(r"<[^>]+>", "", s)
	s = html_lib.unescape(s)
	s = s.replace("\xa0", " ")
	return " ".join(s.split()).strip()


def _parse_market_stat_rows(html_fragment: str, limit: int) -> list[dict]:
	rows: list[dict] = []
	for row_html in re.findall(r"<tr>(.*?)</tr>", html_fragment, flags=re.I | re.S):
		if "href=q/?s=" not in row_html.lower():
			continue
		tds = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.I | re.S)
		if len(tds) < 5:
			continue
		# Column layout: [symbol], [name], [market], [price], [change%], ...
		sym_match = re.search(r">\s*([A-Za-z0-9._^:-]+)\s*</a>", tds[0], flags=re.I)
		symbol = (sym_match.group(1).strip().upper() if sym_match else _strip_tags(tds[0]).upper())
		price_text = _strip_tags(tds[3])
		pct_text = _strip_tags(tds[4])
		m_price = re.search(r"([+-]?\d+(?:\.\d+)?)", price_text)
		m_pct = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", pct_text)
		price = float(m_price.group(1)) if m_price else None
		pct = float(m_pct.group(1)) if m_pct else None
		rows.append({"symbol": symbol, "price": price, "pct": pct})
		if len(rows) >= limit:
			break
	return rows


def _extract_between(html: str, start_pat: str, end_pat: str | None = None) -> str:
	start = re.search(start_pat, html, flags=re.I)
	if not start:
		return ""
	sub = html[start.end():]
	if end_pat:
		end = re.search(end_pat, sub, flags=re.I)
		if end:
			sub = sub[: end.start()]
	return sub


def us_market_movers(limit: int = 8) -> dict:
	"""Return US market lists (Most Active / Top Gainers / Top Losers) from Stooq.

	Each list item: {symbol, price, pct}
	"""
	# Gainers/losers are on the default Market Stat: U.S. page.
	base_url = "https://stooq.com/t/s/?m=us"
	base_html = _fetch_url_bytes(base_url).decode("utf-8", errors="ignore")

	adv_fragment = _extract_between(base_html, r"<b>\s*Advancers\s*</b>", r"<b>\s*Decliners\s*</b>")
	dec_fragment = _extract_between(base_html, r"<b>\s*Decliners\s*</b>", r"</table>")

	# Most active (by turnover) is on t=1.
	active_url = "https://stooq.com/t/s/?m=us&t=1"
	active_html = _fetch_url_bytes(active_url).decode("utf-8", errors="ignore")
	active_fragment = _extract_between(active_html, r"<b>\s*Most\s+Active\s*-\s*Turnover\s*</b>", r"<b>\s*Most\s+Active\s*-\s*No\.\s*Trades\s*</b>")

	return {
		"most_active": _parse_market_stat_rows(active_fragment, limit=limit),
		"gainers": _parse_market_stat_rows(adv_fragment, limit=limit),
		"losers": _parse_market_stat_rows(dec_fragment, limit=limit),
	}


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
