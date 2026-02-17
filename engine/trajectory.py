from __future__ import annotations

import numpy as np


def classify_trajectory(metrics_row: dict) -> tuple[str, list[str]]:
	"""
	Transparent rules (no ML): combine stress signals and trend deltas.

	Returns (trajectory_label, reasons[])
	"""
	reasons: list[str] = []

	base = metrics_row.get("baseline_balance_30d")
	vol = metrics_row.get("balance_volatility_30d")
	buf = metrics_row.get("cash_buffer_days")
	dd = metrics_row.get("drawdown_count_90d")
	rec = metrics_row.get("avg_recovery_days_90d")
	od = metrics_row.get("overdraft_nsf_90d")
	lb = metrics_row.get("latest_balance")
	base_delta = metrics_row.get("baseline_delta_30d")
	out_delta = metrics_row.get("avg_daily_outflow_delta_30d")

	# Normalize missing
	dd = dd if dd is not None else 0
	od = od if od is not None else 0

	stress_points = 0
	improve_points = 0

	# Buffer is the cleanest early warning
	if buf is not None:
		if buf < 3:
			stress_points += 2
			reasons.append("Low cash buffer (<3 days).")
		elif buf < 7:
			stress_points += 1
			reasons.append("Moderate cash buffer (<7 days).")
		elif buf >= 14:
			improve_points += 2
			reasons.append("Strong cash buffer (≥14 days).")
		elif buf >= 10:
			improve_points += 1
			reasons.append("Healthy cash buffer (≥10 days).")

	# Drawdowns
	if dd is not None and dd >= 2:
		stress_points += 2
		reasons.append("Multiple drawdown events in last 90 days.")
	elif dd is not None and dd == 1:
		stress_points += 1
		reasons.append("Drawdown event detected in last 90 days.")

	# Recovery
	if rec is not None:
		if rec > 30:
			stress_points += 2
			reasons.append("Slow recovery after drawdowns (>30 days).")
		elif rec > 14:
			stress_points += 1
			reasons.append("Moderate recovery time after drawdowns (>14 days).")
		elif rec <= 7:
			improve_points += 1
			reasons.append("Fast recovery after drawdowns (≤7 days).")

	# Overdraft/NSF
	if od is not None:
		if od >= 2:
			stress_points += 2
			reasons.append("Multiple overdraft/NSF-like events (90d).")
		elif od == 1:
			stress_points += 1
			reasons.append("Overdraft/NSF-like event detected (90d).")
		elif od == 0:
			improve_points += 1
			reasons.append("No overdraft/NSF-like events (90d).")

	# Volatility relative to baseline
	if vol is not None and base is not None:
		rel = abs(vol) / (abs(base) + 1e-6)
		if rel > 0.75:
			stress_points += 2
			reasons.append("Very high volatility relative to baseline.")
		elif rel > 0.40:
			stress_points += 1
			reasons.append("High volatility relative to baseline.")
		elif rel < 0.15:
			improve_points += 1
			reasons.append("Low volatility relative to baseline.")

	# Latest balance vs baseline
	if lb is not None and base is not None and base != 0:
		drop = (base - lb) / abs(base)
		if drop > 0.35:
			stress_points += 2
			reasons.append("Latest balance significantly below baseline.")
		elif drop > 0.20:
			stress_points += 1
			reasons.append("Latest balance below baseline.")

	# Trend deltas: compare last 30d vs prior 30d when available
	if base_delta is not None and base is not None:
		# Scale by baseline size so business/consumer both work
		rel_delta = base_delta / (abs(base) + 1e-6)
		if rel_delta <= -0.10:
			stress_points += 2
			reasons.append("Baseline (30d median) is down vs prior 30 days.")
		elif rel_delta <= -0.03:
			stress_points += 1
			reasons.append("Baseline is slightly down vs prior 30 days.")
		elif rel_delta >= 0.10:
			improve_points += 2
			reasons.append("Baseline (30d median) is up vs prior 30 days.")
		elif rel_delta >= 0.03:
			improve_points += 1
			reasons.append("Baseline is slightly up vs prior 30 days.")

	if out_delta is not None and (metrics_row.get("avg_daily_outflow_30d") is not None):
		out = float(metrics_row.get("avg_daily_outflow_30d"))
		rel_out = out_delta / (abs(out) + 1e-6)
		if rel_out >= 0.15:
			stress_points += 1
			reasons.append("Average daily outflow is up vs prior 30 days.")
		elif rel_out <= -0.15:
			improve_points += 1
			reasons.append("Average daily outflow is down vs prior 30 days.")

	# Decide label
	if stress_points >= 4 and stress_points > improve_points:
		label = "Declining"
	elif improve_points >= 4 and improve_points > stress_points:
		label = "Improving"
	else:
		label = "Stable"

	# Keep top 3 reasons (order is severity-ish)
	top_reasons = reasons[:3]
	return label, top_reasons


def add_trajectory(portfolio) -> object:
	labels = []
	reasons = []
	severity = []
	for _, row in portfolio.iterrows():
		label, r = classify_trajectory(row.to_dict())
		labels.append(label)
		reasons.append("; ".join(r) if r else "")
		severity.append({"Declining": 2, "Stable": 1, "Improving": 0}.get(label, 1))
	out = portfolio.copy()
	out["trajectory"] = labels
	out["trajectory_reasons"] = reasons
	out["trajectory_severity"] = severity
	return out
