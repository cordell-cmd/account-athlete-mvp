from __future__ import annotations

import io
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from engine.io import load_accounts, load_transactions, load_balances, derive_balances_from_transactions
from engine.metrics import (
	compute_portfolio_metrics,
	weighted_avg_interest_rate_overall,
	weighted_avg_interest_rate_by_product
)
from engine.trajectory import add_trajectory
from engine.explain import player_card_text

st.set_page_config(page_title="Relationship Signals Prototype", layout="wide")

# -----------------------------
# Newspaper-style masthead
# -----------------------------
st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&display=swap');

	.cnb-masthead {
		text-align: center;
		margin: 0.25rem 0 0.75rem 0;
	}

	.cnb-title {
		font-family: 'Playfair Display', 'Georgia', 'Times New Roman', serif;
		font-size: 56px;
		font-weight: 800;
		letter-spacing: 0.5px;
		line-height: 1.0;
		margin: 0;
		padding: 0;
	}

	.cnb-deck {
		font-family: 'Georgia', 'Times New Roman', serif;
		font-size: 14px;
		margin-top: 0.25rem;
		color: rgba(0,0,0,0.70);
	}

	.cnb-nav {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 18px;
		border-top: 1px solid rgba(0,0,0,0.65);
		border-bottom: 1px solid rgba(0,0,0,0.65);
		padding: 8px 0;
		margin: 0.75rem 0 0.25rem 0;
		font-family: 'Georgia', 'Times New Roman', serif;
		font-size: 12px;
		letter-spacing: 1px;
		text-transform: uppercase;
	}

	.cnb-nav span {
		color: rgba(0,0,0,0.70);
	}

	.cnb-nav .active {
		color: rgba(0,0,0,0.95);
		font-weight: 700;
		text-decoration: underline;
		text-underline-offset: 4px;
	}

	/* Streamlit metric sizing (avoid clipped values on smaller widths) */
	div[data-testid="stMetricValue"] {
		font-size: 1.35rem;
		line-height: 1.1;
		white-space: nowrap;
	}
	div[data-testid="stMetricLabel"] {
		font-size: 0.9rem;
		line-height: 1.1;
	}
	@media (max-width: 1100px) {
		div[data-testid="stMetricValue"] { font-size: 1.15rem; }
		div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
	}
	@media (max-width: 800px) {
		div[data-testid="stMetricValue"] { font-size: 1.0rem; }
		div[data-testid="stMetricLabel"] { font-size: 0.8rem; }
	}
	</style>

	<div class="cnb-masthead">
		<div class="cnb-title">The CNB Times</div>
	</div>
	<div class="cnb-nav">
		<span class="active">Front Page</span>
		<span>Lending</span>
		<span>Risk</span>
		<span>Wealth</span>
		<span>Markets</span>
		<span>Treasury</span>
		<span>Operations</span>
		<span>Compliance</span>
	</div>
	""",
	unsafe_allow_html=True,
)

st.caption("Front Page • Portfolio Desk")


def _to_bytes(f) -> bytes:
	"""Return stable bytes for caching (works for both local files and Streamlit uploads)."""
	if f is None:
		return b""
	# Streamlit UploadedFile
	if hasattr(f, "getvalue"):
		return bytes(f.getvalue())
	# file-like
	try:
		f.seek(0)
	except Exception:
		pass
	return bytes(f.read())


@st.cache_data(show_spinner=False)
def _compute_portfolio_from_csv_bytes(accounts_bytes: bytes, txns_bytes: bytes, balances_bytes: bytes | None):
	accounts_df = load_accounts(io.BytesIO(accounts_bytes))
	txns_df = load_transactions(io.BytesIO(txns_bytes))
	if balances_bytes:
		balances_df = load_balances(io.BytesIO(balances_bytes))
	else:
		balances_df = derive_balances_from_transactions(txns_df)
	portfolio_df = compute_portfolio_metrics(accounts_df, txns_df, balances_df)
	portfolio_df = add_trajectory(portfolio_df)
	return portfolio_df, balances_df

# Sidebar controls
st.sidebar.header("Data input")
use_sample = st.sidebar.toggle("Use bundled sample data", value=True)

accounts_file = None
txns_file = None
bal_file = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_ACCTS = os.path.join(DATA_DIR, "sample_accounts.csv")
SAMPLE_TXNS = os.path.join(DATA_DIR, "sample_transactions.csv")
SAMPLE_BALS = os.path.join(DATA_DIR, "sample_daily_balances.csv")

if use_sample:
	if not (os.path.exists(SAMPLE_ACCTS) and os.path.exists(SAMPLE_TXNS) and os.path.exists(SAMPLE_BALS)):
		st.sidebar.warning("Sample files not found. Run: python scripts/make_sample_data.py")
	else:
		accounts_file = open(SAMPLE_ACCTS, "rb")
		txns_file = open(SAMPLE_TXNS, "rb")
		bal_file = open(SAMPLE_BALS, "rb")
else:
	accounts_file = st.sidebar.file_uploader("Upload accounts.csv", type=["csv"])
	txns_file = st.sidebar.file_uploader("Upload transactions.csv", type=["csv"])
	bal_file = st.sidebar.file_uploader("Upload daily_balances.csv (optional)", type=["csv"])

st.sidebar.divider()
st.sidebar.markdown("### How to test")
st.sidebar.markdown("1) Use sample data toggle, or upload CSVs.\n2) Pick an account.\n3) Review player card + trajectory + chart.")

if not accounts_file or not txns_file:
	st.info("Provide accounts.csv and transactions.csv (or use sample data) to begin.")
	st.stop()

# Load + compute (cached) with safe error handling
try:
	acct_bytes = _to_bytes(accounts_file)
	txn_bytes = _to_bytes(txns_file)
	bal_bytes = _to_bytes(bal_file) if bal_file else None
	portfolio_raw, balances = _compute_portfolio_from_csv_bytes(acct_bytes, txn_bytes, bal_bytes)
except Exception as e:
	st.error(f"Data load/compute error: {e}")
	st.stop()

if balances.empty:
	st.warning("No balance data available. Some metrics will be unavailable unless daily_balances.csv is provided.")


def _truthy_mask(series: pd.Series) -> pd.Series:
	return (
		series.astype(str)
		.str.strip()
		.str.lower()
		.isin(["true", "1", "yes", "y", "t"])
	)


def _numeric_range(series: pd.Series, fallback_min: float, fallback_max: float) -> tuple[float, float]:
	s = pd.to_numeric(series, errors="coerce").dropna()
	if s.empty:
		return float(fallback_min), float(fallback_max)
	return float(s.min()), float(s.max())


def _between_or_na(series: pd.Series, lo: float, hi: float) -> pd.Series:
	"""Range filter that does not exclude missing values.

	This prevents 'full range' sliders from accidentally dropping accounts whose
	metric is unavailable (NaN).
	"""
	s = pd.to_numeric(series, errors="coerce")
	return s.isna() | s.between(lo, hi)


# -----------------------------
# Sidebar: Segment filters (scales to 1,000+ accounts)
# -----------------------------
st.sidebar.header("Segment & Search")

portfolio = portfolio_raw.copy()

# normalize numeric columns for filtering
_numeric_cols = [
	"loan_balance",
	"interest_rate",
	"risk_rating",
	"days_past_due",
	"latest_balance",
	"baseline_balance_30d",
	"balance_volatility_30d",
	"cash_buffer_days",
	"drawdown_count_90d",
	"avg_recovery_days_90d",
	"overdraft_nsf_90d",
]
for _c in _numeric_cols:
	if _c in portfolio.columns:
		portfolio[_c] = pd.to_numeric(portfolio[_c], errors="coerce")

only_loans = st.sidebar.checkbox("Loans only", value=False)

traj_options = sorted([t for t in portfolio.get("trajectory", pd.Series(dtype=str)).dropna().unique().tolist()])
traj_selected = st.sidebar.multiselect("Trajectory", options=traj_options, default=traj_options)

prod_series = portfolio.get("product", pd.Series(dtype=object))
prod_options = sorted([p for p in prod_series.dropna().astype(str).unique().tolist()])
prod_selected = st.sidebar.multiselect("Product", options=prod_options, default=prod_options)

st.sidebar.markdown("**Loan risk filters**")
risk_min, risk_max = (1, 9)
risk_sel = st.sidebar.slider("Risk rating (filter range)", min_value=risk_min, max_value=risk_max, value=(risk_min, risk_max))
dpd_sel = st.sidebar.slider("Days past due (filter range)", min_value=0, max_value=90, value=(0, 90))

rate_lo, rate_hi = _numeric_range(portfolio.get("interest_rate", pd.Series(dtype=float)), 4.0, 10.0)
rate_sel = st.sidebar.slider(
	"Interest rate (%)",
	min_value=float(min(0.0, rate_lo)),
	max_value=float(max(10.0, rate_hi)),
	value=(float(rate_lo), float(rate_hi)),
)

loan_bal_sel = None
if "loan_balance" in portfolio.columns:
	loan_lo, loan_hi = _numeric_range(portfolio.get("loan_balance", pd.Series(dtype=float)), 0.0, 1.0)
	loan_bal_sel = st.sidebar.slider(
		"Loan balance (outstanding)",
		min_value=float(max(0.0, loan_lo)),
		max_value=float(max(1.0, loan_hi)),
		value=(float(max(0.0, loan_lo)), float(max(1.0, loan_hi))),
	)

with st.sidebar.expander("Advanced metric filters", expanded=False):
	bal_lo, bal_hi = _numeric_range(portfolio.get("latest_balance", pd.Series(dtype=float)), 0.0, 1.0)
	baseline_lo, baseline_hi = _numeric_range(portfolio.get("baseline_balance_30d", pd.Series(dtype=float)), 0.0, 1.0)
	vol_lo, vol_hi = _numeric_range(portfolio.get("balance_volatility_30d", pd.Series(dtype=float)), 0.0, 1.0)
	buf_lo, buf_hi = _numeric_range(portfolio.get("cash_buffer_days", pd.Series(dtype=float)), 0.0, 1.0)
	dd_lo, dd_hi = _numeric_range(portfolio.get("drawdown_count_90d", pd.Series(dtype=float)), 0.0, 1.0)
	rec_lo, rec_hi = _numeric_range(portfolio.get("avg_recovery_days_90d", pd.Series(dtype=float)), 0.0, 1.0)
	od_lo, od_hi = _numeric_range(portfolio.get("overdraft_nsf_90d", pd.Series(dtype=float)), 0.0, 1.0)

	bal_sel = st.slider("Latest balance", min_value=float(bal_lo), max_value=float(bal_hi), value=(float(bal_lo), float(bal_hi)))
	baseline_sel = st.slider("Baseline (30d median)", min_value=float(baseline_lo), max_value=float(baseline_hi), value=(float(baseline_lo), float(baseline_hi)))
	vol_sel = st.slider("Volatility (30d std)", min_value=float(vol_lo), max_value=float(vol_hi), value=(float(vol_lo), float(vol_hi)))
	buf_sel = st.slider("Cash buffer (days)", min_value=float(buf_lo), max_value=float(buf_hi), value=(float(buf_lo), float(buf_hi)))
	dd_sel = st.slider("Drawdowns (90d)", min_value=int(dd_lo), max_value=int(dd_hi), value=(int(dd_lo), int(dd_hi)))
	rec_sel = st.slider("Avg recovery (days)", min_value=float(rec_lo), max_value=float(rec_hi), value=(float(rec_lo), float(rec_hi)))
	od_sel = st.slider("Overdraft/NSF (90d)", min_value=int(od_lo), max_value=int(od_hi), value=(int(od_lo), int(od_hi)))

# Apply filters
mask = pd.Series(True, index=portfolio.index)

if only_loans and "has_loan" in portfolio.columns:
	mask &= _truthy_mask(portfolio["has_loan"])

if "trajectory" in portfolio.columns and traj_selected:
	# If all trajectory options are selected, treat as no filter.
	if set(traj_selected) != set(traj_options):
		mask &= portfolio["trajectory"].isin(traj_selected)

if "product" in portfolio.columns and prod_selected:
	# If all product options are selected, treat as no filter.
	if set(prod_selected) != set(prod_options):
		mask &= portfolio["product"].astype(str).isin(prod_selected)

if "risk_rating" in portfolio.columns:
	mask &= _between_or_na(portfolio["risk_rating"], risk_sel[0], risk_sel[1])

if "days_past_due" in portfolio.columns:
	mask &= _between_or_na(portfolio["days_past_due"], dpd_sel[0], dpd_sel[1])

if "interest_rate" in portfolio.columns:
	mask &= _between_or_na(portfolio["interest_rate"], rate_sel[0], rate_sel[1])

if loan_bal_sel is not None and "loan_balance" in portfolio.columns:
	mask &= _between_or_na(portfolio["loan_balance"], loan_bal_sel[0], loan_bal_sel[1])

if "latest_balance" in portfolio.columns:
	mask &= _between_or_na(portfolio["latest_balance"], bal_sel[0], bal_sel[1])
if "baseline_balance_30d" in portfolio.columns:
	mask &= _between_or_na(portfolio["baseline_balance_30d"], baseline_sel[0], baseline_sel[1])
if "balance_volatility_30d" in portfolio.columns:
	mask &= _between_or_na(portfolio["balance_volatility_30d"], vol_sel[0], vol_sel[1])
if "cash_buffer_days" in portfolio.columns:
	mask &= _between_or_na(portfolio["cash_buffer_days"], buf_sel[0], buf_sel[1])
if "drawdown_count_90d" in portfolio.columns:
	mask &= _between_or_na(portfolio["drawdown_count_90d"], dd_sel[0], dd_sel[1])
if "avg_recovery_days_90d" in portfolio.columns:
	mask &= _between_or_na(portfolio["avg_recovery_days_90d"], rec_sel[0], rec_sel[1])
if "overdraft_nsf_90d" in portfolio.columns:
	mask &= _between_or_na(portfolio["overdraft_nsf_90d"], od_sel[0], od_sel[1])

portfolio = portfolio[mask].copy()

st.sidebar.caption(f"Accounts after filters: {len(portfolio):,} / {len(portfolio_raw):,}")

# -----------------------------
# Portfolio headline metrics
# -----------------------------
st.subheader("Portfolio headline")

# Loan-only view for loan metrics (WAIR / risk). Deposit balances remain portfolio-wide.
if "has_loan" in portfolio.columns:
	loans_view = portfolio[_truthy_mask(portfolio["has_loan"])].copy()
else:
	loans_view = portfolio.copy()

wair_all = weighted_avg_interest_rate_overall(loans_view)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
	st.metric("Accounts in view (all)", str(len(portfolio)))
with c2:
	total_bal = portfolio["latest_balance"].dropna().sum() if "latest_balance" in portfolio.columns else 0
	st.metric("Total latest balance (deposit)", f"${total_bal:,.0f}")
	st.caption("Sum of deposit balances as-of latest date.")
with c3:
	st.metric("Loans in view", str(len(loans_view)))

with c4:
	loan_exposure = loans_view["loan_balance"].dropna().sum() if "loan_balance" in loans_view.columns else 0
	st.metric("Total loan balance", f"${loan_exposure:,.0f}" if loan_exposure else "—")
	st.caption("Outstanding principal (loan exposure).")

with c5:
	if wair_all is None:
		st.metric("WAIR (loans only)", "—")
		st.caption("Weighted by loan balance.")
	else:
		st.metric("WAIR (loans only)", f"{wair_all:.2f}%")
		st.caption("Weighted by loan balance.")

with c6:
	# Simple loan-risk headline: count of high risk or past due (loan-only)
	if {"risk_rating", "days_past_due"}.issubset(set(loans_view.columns)):
		high_risk_count = int((loans_view["risk_rating"].fillna(0).astype(int) >= 7).sum())
		past_due_count = int((loans_view["days_past_due"].fillna(0).astype(int) > 0).sum())
		st.metric("High risk loans (≥7)", str(high_risk_count))
		st.caption(f"Past due loans (>0 days): {past_due_count}")
	else:
		st.metric("High risk loans (≥7)", "—")

# WAIR by Product table (loan-only)
wair_by_prod = weighted_avg_interest_rate_by_product(loans_view)
st.markdown("**WAIR by Product (loans only; weighted by loan balance)**")
show_unweighted = st.checkbox(
	"Show simple average rate (unweighted)",
	value=False,
	help="Adds the plain (unweighted) mean interest rate per product to contrast against balance-weighted WAIR.",
)
if wair_by_prod.empty:
	st.write("— (missing product/interest_rate/loan_balance)")
else:
	wair_disp = wair_by_prod.copy()
	# rename for clarity (this is loan exposure, not deposit balance)
	if "total_balance" in wair_disp.columns:
		wair_disp = wair_disp.rename(columns={"total_balance": "total_loan_balance"})

	if show_unweighted:
		df_rates = loans_view.dropna(subset=["product", "interest_rate", "loan_balance"]).copy()
		df_rates["interest_rate"] = pd.to_numeric(df_rates["interest_rate"], errors="coerce")
		df_rates["loan_balance"] = pd.to_numeric(df_rates["loan_balance"], errors="coerce")
		df_rates = df_rates.dropna(subset=["interest_rate", "loan_balance"]) 
		df_rates = df_rates[df_rates["loan_balance"] > 0]
		avg_by_prod = (
			df_rates.groupby("product", dropna=False)["interest_rate"]
			.mean()
			.reset_index()
			.rename(columns={"interest_rate": "avg_rate_pct"})
		)
		wair_disp = wair_disp.merge(avg_by_prod, on="product", how="left")

	wair_disp["wair_pct"] = wair_disp["wair_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
	if "avg_rate_pct" in wair_disp.columns:
		wair_disp["avg_rate_pct"] = wair_disp["avg_rate_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
	if "total_loan_balance" in wair_disp.columns:
		wair_disp["total_loan_balance"] = wair_disp["total_loan_balance"].map(lambda x: f"${x:,.0f}")
	st.dataframe(wair_disp, use_container_width=True)

# -----------------------------
# Segment summary (aggregate view)
# -----------------------------
st.subheader("Segment summary")
st.caption("Aggregates the filtered accounts into buckets so you can browse segments before drilling into a specific account.")

seg = portfolio.copy()
if seg.empty:
	st.info("No accounts match the current filters.")
else:
	# Buckets (lightweight, exec-friendly)
	if "risk_rating" in seg.columns:
		seg["risk_bucket"] = pd.cut(seg["risk_rating"], bins=[0, 3, 6, 9], labels=["1–3", "4–6", "7–9"], include_lowest=True)
	if "days_past_due" in seg.columns:
		seg["dpd_bucket"] = pd.cut(seg["days_past_due"], bins=[-0.1, 0, 10, 30, 60, 90], labels=["0", "1–10", "11–30", "31–60", "61–90"], include_lowest=True)
	if "interest_rate" in seg.columns:
		seg["rate_bucket"] = pd.cut(seg["interest_rate"], bins=[0, 5, 6, 7, 8, 9, 10, 99], labels=["<5", "5–6", "6–7", "7–8", "8–9", "9–10", "10+"], include_lowest=True)
	if "latest_balance" in seg.columns:
		_bal = pd.to_numeric(seg["latest_balance"], errors="coerce").dropna()
		if _bal.empty or _bal.nunique(dropna=True) < 2:
			seg["balance_bucket"] = "All"
		else:
			qs = _bal.quantile([0.25, 0.5, 0.75]).tolist()
			raw_bins = [-float("inf"), float(qs[0]), float(qs[1]), float(qs[2]), float("inf")]
			# De-duplicate bins (must be strictly increasing for pd.cut)
			bins = []
			for b in raw_bins:
				if not bins:
					bins.append(b)
					continue
				if b > bins[-1]:
					bins.append(b)
			# If we couldn't form at least 2 intervals, fall back to a single bucket
			if len(bins) < 3:
				seg["balance_bucket"] = "All"
			else:
				labels_pool = ["Low", "Mid", "High", "Top"]
				labels = labels_pool[: (len(bins) - 1)]
				seg["balance_bucket"] = pd.cut(
					pd.to_numeric(seg["latest_balance"], errors="coerce"),
					bins=bins,
					labels=labels,
					include_lowest=True,
				)

	group_choices = [c for c in ["trajectory", "product", "risk_bucket", "dpd_bucket", "rate_bucket", "balance_bucket"] if c in seg.columns]
	group_by = st.multiselect("Group by", options=group_choices, default=[c for c in ["trajectory", "product"] if c in group_choices])
	if not group_by:
		st.info("Pick at least one grouping field.")
	else:
		g = seg.groupby(group_by, dropna=False)
		rows = []
		for keys, df in g:
			if not isinstance(keys, tuple):
				keys = (keys,)
			row = {group_by[i]: keys[i] for i in range(len(group_by))}
			row["account_count"] = int(len(df))
			if "latest_balance" in df.columns:
				row["total_deposit_balance"] = float(df["latest_balance"].fillna(0).sum())
				row["avg_deposit_balance"] = float(df["latest_balance"].dropna().mean()) if not df["latest_balance"].dropna().empty else None
			if "loan_balance" in df.columns:
				row["total_loan_balance"] = float(df["loan_balance"].fillna(0).sum())
				row["avg_loan_balance"] = float(df["loan_balance"].dropna().mean()) if not df["loan_balance"].dropna().empty else None
			# weighted WAIR within segment
			row["wair_pct"] = weighted_avg_interest_rate_overall(df)
			# simple means for the relationship metrics
			for c in [
				"baseline_balance_30d",
				"balance_volatility_30d",
				"cash_buffer_days",
				"drawdown_count_90d",
				"avg_recovery_days_90d",
				"overdraft_nsf_90d",
				"risk_rating",
				"days_past_due",
				"interest_rate",
			]:
				if c in df.columns:
					val = df[c].dropna().mean() if not df[c].dropna().empty else None
					row[f"avg_{c}"] = float(val) if val is not None else None
			rows.append(row)

		seg_table = pd.DataFrame(rows)
		if "total_loan_balance" in seg_table.columns:
			seg_table = seg_table.sort_values(["total_loan_balance", "account_count"], ascending=[False, False])
		elif "total_deposit_balance" in seg_table.columns:
			seg_table = seg_table.sort_values(["total_deposit_balance", "account_count"], ascending=[False, False])
		else:
			seg_table = seg_table.sort_values(["account_count"], ascending=[False])

		# Display formatting
		disp = seg_table.copy()
		def _coerce_float(x):
			if pd.isna(x):
				return None
			if isinstance(x, str):
				x = x.replace("$", "").replace(",", "").strip()
				if x in {"", "—"}:
					return None
			try:
				return float(x)
			except Exception:
				return None

		def _fmt_pct(x):
			v = _coerce_float(x)
			return "—" if v is None else f"{v:.2f}%"

		def _fmt_currency0(x):
			v = _coerce_float(x)
			return "—" if v is None else f"${v:,.0f}"

		def _fmt_float2(x):
			v = _coerce_float(x)
			return "—" if v is None else f"{v:.2f}"

		if "wair_pct" in disp.columns:
			disp["wair_pct"] = disp["wair_pct"].map(_fmt_pct)
		if "total_deposit_balance" in disp.columns:
			disp["total_deposit_balance"] = disp["total_deposit_balance"].map(_fmt_currency0)
		if "avg_deposit_balance" in disp.columns:
			disp["avg_deposit_balance"] = disp["avg_deposit_balance"].map(_fmt_currency0)
		if "total_loan_balance" in disp.columns:
			disp["total_loan_balance"] = disp["total_loan_balance"].map(_fmt_currency0)
		if "avg_loan_balance" in disp.columns:
			disp["avg_loan_balance"] = disp["avg_loan_balance"].map(_fmt_currency0)
		for c in [c for c in disp.columns if c.startswith("avg_")]:
			if c in {"avg_deposit_balance", "avg_latest_balance", "avg_baseline_balance_30d", "avg_loan_balance"}:
				disp[c] = disp[c].map(_fmt_currency0)
			else:
				disp[c] = disp[c].map(_fmt_float2)

		st.dataframe(disp, use_container_width=True)

# -----------------------------
# NEW: High Risk / Past Due Today drill-down
# -----------------------------
st.subheader("High Risk / Past Due Loans (Today)")

colA, colB, colC = st.columns([1, 1, 2])
with colA:
	risk_cut = st.slider("Risk rating threshold (hot list)", min_value=1, max_value=9, value=7)
with colB:
	dpd_cut = st.slider("Days past due threshold (hot list)", min_value=0, max_value=90, value=30)
with colC:
	st.caption("This panel surfaces the small set of loans that most warrant attention today (high risk and/or meaningfully past due), within the current left-sidebar filters.")

# Only include loan accounts in this drill-down
if "has_loan" in portfolio.columns:
	has_loan_mask = (
		portfolio["has_loan"]
		.astype(str)
		.str.strip()
		.str.lower()
		.isin(["true", "1", "yes", "y", "t"])
	)
	drill = portfolio[has_loan_mask].copy()
else:
	drill = portfolio.copy()
if "risk_rating" in drill.columns:
	drill["risk_rating"] = pd.to_numeric(drill["risk_rating"], errors="coerce")
if "days_past_due" in drill.columns:
	drill["days_past_due"] = pd.to_numeric(drill["days_past_due"], errors="coerce")

has_risk = "risk_rating" in drill.columns
has_dpd = "days_past_due" in drill.columns

if not (has_risk and has_dpd):
	st.info("Risk fields not present. Ensure accounts.csv includes risk_rating and days_past_due.")
else:
	# Flag accounts meeting either condition
	drill["risk_flag"] = drill["risk_rating"].fillna(0) >= risk_cut
	drill["dpd_flag"] = drill["days_past_due"].fillna(0) >= dpd_cut

	hot = drill[(drill["risk_flag"]) | (drill["dpd_flag"])].copy()

	# -----------------------------
	# NEW: Exec narrative ("CNB Times")
	# -----------------------------
	loan_count = int(len(drill))
	hot_count = int(len(hot))
	hot_pct = (hot_count / loan_count * 100.0) if loan_count else 0.0

	loan_bal_sum = float(pd.to_numeric(drill.get("loan_balance"), errors="coerce").fillna(0).sum()) if "loan_balance" in drill.columns else 0.0
	loan_wair = weighted_avg_interest_rate_overall(drill)
	severe_dpd = int((drill["days_past_due"].fillna(0) >= 60).sum())
	very_high_risk = int((drill["risk_rating"].fillna(0) >= 8).sum())

	top_prod_txt = "—"
	if "product" in drill.columns and "loan_balance" in drill.columns:
		_prod = drill.dropna(subset=["product"]).copy()
		if not _prod.empty:
			_prod["loan_balance"] = pd.to_numeric(_prod["loan_balance"], errors="coerce")
			g = _prod.groupby("product")["loan_balance"].sum().sort_values(ascending=False)
			if not g.empty:
				top_prod = str(g.index[0])
				top_prod_bal = float(g.iloc[0])
				top_prod_share = (top_prod_bal / loan_bal_sum * 100.0) if loan_bal_sum else 0.0
				top_prod_txt = f"{top_prod} (${top_prod_bal:,.0f}, {top_prod_share:.0f}% of loan exposure)"

	as_of_txt = str(drill["as_of"].dropna().iloc[0]) if "as_of" in drill.columns and not drill["as_of"].dropna().empty else "today"
	wair_txt = "—" if loan_wair is None else f"{loan_wair:.2f}%"

	narrative = (
		f"CNB Times — As of {as_of_txt}, the loan book in view includes {loan_count:,} loans totaling ${loan_bal_sum:,.0f} "
		f"with WAIR {wair_txt}. {hot_count:,} loans ({hot_pct:.1f}%) meet today’s attention thresholds "
		f"(risk ≥ {risk_cut} and/or days past due ≥ {dpd_cut}). "
		f"Severe delinquency (60+ DPD): {severe_dpd:,}; very high risk (8–9): {very_high_risk:,}. "
		f"Largest product concentration: {top_prod_txt}."
	)
	st.markdown("**CNB Times (Auto-Summary)**")
	st.write(narrative)

	# Rank: highest days past due first, then highest risk, then largest loan exposure
	sort_cols = [c for c in ["days_past_due", "risk_rating", "loan_balance"] if c in hot.columns]
	if sort_cols:
		hot = hot.sort_values(sort_cols, ascending=[False] * len(sort_cols))

	if hot.empty:
		st.success("No accounts meet the current thresholds.")
	else:
		# -----------------------------
		# NEW: Wealth Team export (CSV)
		# -----------------------------
		export_cols = [
			"account_id",
			"customer_type",
			"has_loan",
			"product",
			"loan_balance",
			"interest_rate",
			"risk_rating",
			"days_past_due",
			"trajectory",
			"trajectory_reasons",
			"risk_flag",
			"dpd_flag",
			"as_of",
		]
		export_cols = [c for c in export_cols if c in hot.columns]
		export_df = hot[export_cols].copy()
		csv_bytes = export_df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="Send to Wealth Team (CSV)",
			data=csv_bytes,
			file_name=f"wealth_team_hot_loans_{as_of_txt}.csv",
			mime="text/csv",
		)

		show_hot_cols = [
			"account_id",
			"customer_type",
			"product",
			"risk_rating",
			"days_past_due",
			"interest_rate",
			"loan_balance",
			"trajectory",
			"trajectory_reasons",
		]
		show_hot_cols = [c for c in show_hot_cols if c in hot.columns]
		show_hot = hot[show_hot_cols].copy()

		# display formatting
		if "interest_rate" in show_hot.columns:
			show_hot["interest_rate"] = show_hot["interest_rate"].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}%")
		if "loan_balance" in show_hot.columns:
			show_hot["loan_balance"] = show_hot["loan_balance"].map(lambda x: "—" if pd.isna(x) else f"${float(x):,.0f}")

		st.dataframe(show_hot, use_container_width=True)

st.divider()

# Account selector (search-first)
st.subheader("Account lookup")
st.caption("Pick a specific account from the currently filtered set to view its player card + trend.")

if portfolio.empty:
	st.info("No accounts available after filters.")
	st.stop()

acc_ids_all = portfolio["account_id"].astype(str).tolist()

# Streamlit selectboxes can get sluggish with extremely large option lists.
# For demo-sized portfolios, show all accounts by default.
MAX_SELECTBOX_OPTIONS = 2000

query = st.text_input("Search account_id", value="", placeholder="Type e.g., A-1001 or B-30…")

if query.strip():
	q = query.strip().lower()
	acc_ids = [a for a in acc_ids_all if q in a.lower()]
	if not acc_ids:
		st.warning("No account_ids match your search in the current filtered set.")
		acc_ids = acc_ids_all
	else:
		if len(acc_ids) > MAX_SELECTBOX_OPTIONS:
			st.caption(
				f"Showing first {MAX_SELECTBOX_OPTIONS:,} of {len(acc_ids):,} matches. Refine your search to narrow further."
			)
			acc_ids = acc_ids[:MAX_SELECTBOX_OPTIONS]
else:
	if len(acc_ids_all) <= MAX_SELECTBOX_OPTIONS:
		acc_ids = acc_ids_all
		st.caption(f"Showing all {len(acc_ids_all):,} accounts in the current filtered set.")
	else:
		acc_ids = acc_ids_all[:MAX_SELECTBOX_OPTIONS]
		st.caption(f"Showing first {MAX_SELECTBOX_OPTIONS:,} of {len(acc_ids_all):,} accounts. Type to search.")

selected = st.selectbox("Select account", acc_ids)

row = portfolio[portfolio["account_id"].astype(str) == str(selected)].iloc[0].to_dict()

# Layout columns
left, right = st.columns([1.15, 1.0])

with left:
	st.subheader("Player Card")
	st.markdown(player_card_text(row))

	st.write(
		f"**Product:** {row.get('product', '—')}  \n"
		f"**Interest rate:** {('—' if row.get('interest_rate') is None else str(row.get('interest_rate')) + '%')}  \n"
		f"**Risk rating:** {row.get('risk_rating', '—')}  \n"
		f"**Days past due:** {row.get('days_past_due', '—')}"
	)

	st.subheader("Trajectory")
	st.metric("Trajectory", row.get("trajectory", "—"))
	reasons = row.get("trajectory_reasons", "")
	if reasons:
		st.write("Top reasons:")
		for r in reasons.split("; "):
			if r.strip():
				st.write(f"- {r.strip()}")

with right:
	st.subheader("Balance trend (last 90 days)")
	bal_acc = balances[balances["account_id"].astype(str) == str(selected)].copy()
	bal_acc = bal_acc.sort_values("date")
	if not bal_acc.empty:
		end_date = pd.to_datetime(bal_acc["date"]).max()
		start_date = end_date - pd.Timedelta(days=90)
		bal_90 = bal_acc[(bal_acc["date"] >= start_date) & (bal_acc["date"] <= end_date)].copy()

		fig = plt.figure()
		plt.plot(bal_90["date"], bal_90["balance"])
		plt.xlabel("Date")
		plt.ylabel("Balance")
		plt.xticks(rotation=30)
		st.pyplot(fig, clear_figure=True)
	else:
		st.info("No balance series available for chart.")

st.divider()

with st.expander("Portfolio rows (filtered) — raw table", expanded=False):
	st.caption("This is the underlying row-level dataset for the current filters. Use Segment summary above for browsing patterns.")
	portfolio_view = portfolio.sort_values(["trajectory_severity", "cash_buffer_days"], ascending=[False, True])

	show_cols = [
		"account_id", "customer_type", "product", "interest_rate",
		"risk_rating", "days_past_due",
		"as_of", "trajectory", "trajectory_reasons",
		"loan_balance",
		"latest_balance", "baseline_balance_30d", "balance_volatility_30d", "cash_buffer_days",
		"drawdown_count_90d", "avg_recovery_days_90d", "overdraft_nsf_90d",
	]
	existing = [c for c in show_cols if c in portfolio_view.columns]
	st.dataframe(portfolio_view[existing], use_container_width=True)

st.caption("Note: Prototype uses explainable rules + sample data. It is intended for concept validation, not production decisions.")
