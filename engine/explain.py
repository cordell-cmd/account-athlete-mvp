from __future__ import annotations


def fmt_money(x):
	if x is None:
		return "—"
	try:
		return f"${x:,.0f}"
	except Exception:
		return str(x)


def fmt_num(x, nd=2):
	if x is None:
		return "—"
	try:
		if x == float("inf"):
			return "∞"
		return f"{x:.{nd}f}"
	except Exception:
		return str(x)


def fmt_days(x):
	if x is None:
		return "—"
	try:
		if x == float("inf"):
			return "∞ days"
		return f"{x:.0f} days"
	except Exception:
		return str(x)


def player_card_text(metrics_row: dict) -> str:
	return (
		f"As of **{metrics_row.get('as_of', '—')}**\n\n"
		f"- Latest balance: **{fmt_money(metrics_row.get('latest_balance'))}**\n"
		f"- Baseline (30d median): **{fmt_money(metrics_row.get('baseline_balance_30d'))}**\n"
		f"- Volatility (30d std): **{fmt_money(metrics_row.get('balance_volatility_30d'))}**\n"
		f"- Cash buffer: **{fmt_days(metrics_row.get('cash_buffer_days'))}**\n"
		f"- Drawdowns (90d): **{fmt_num(metrics_row.get('drawdown_count_90d'), 0)}**\n"
		f"- Avg recovery time: **{fmt_days(metrics_row.get('avg_recovery_days_90d'))}**\n"
		f"- Overdraft/NSF-like events (90d): **{fmt_num(metrics_row.get('overdraft_nsf_90d'), 0)}**\n"
	)
