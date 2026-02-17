from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MetricResult:
    value: float | None
    detail: str = ""


def _window(df: pd.DataFrame, date_col: str, days: int, end_date: pd.Timestamp) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=days)
    return df[(df[date_col] > start) & (df[date_col] <= end_date)].copy()


def baseline_balance_30d(bal: pd.DataFrame, end_date: pd.Timestamp) -> MetricResult:
    w = _window(bal, "date", 30, end_date)
    if w.empty:
        return MetricResult(None, "No balance data in last 30 days.")
    return MetricResult(float(w["balance"].median()), "Median daily balance (30d).")


def volatility_balance_30d(bal: pd.DataFrame, end_date: pd.Timestamp) -> MetricResult:
    w = _window(bal, "date", 30, end_date)
    if len(w) < 5:
        return MetricResult(None, "Insufficient balance points in last 30 days.")
    return MetricResult(float(w["balance"].std(ddof=0)), "Std dev of daily balance (30d).")


def latest_balance(bal: pd.DataFrame) -> MetricResult:
    if bal.empty:
        return MetricResult(None, "No balance data.")
    last = bal.sort_values("date").iloc[-1]
    return MetricResult(float(last["balance"]), f"Latest balance on {last['date'].date()}.")


def avg_daily_outflow_30d(txns: pd.DataFrame, end_date: pd.Timestamp) -> MetricResult:
    w = _window(txns, "post_date", 30, end_date)
    if w.empty:
        return MetricResult(None, "No transactions in last 30 days.")
    w["date"] = w["post_date"].dt.floor("D")
    daily_out = w[w["amount"] < 0].groupby("date")["amount"].sum().abs()
    if daily_out.empty:
        return MetricResult(0.0, "No outflows in last 30 days.")
    return MetricResult(float(daily_out.mean()), "Average daily outflow (30d).")


def cash_buffer_days(bal: pd.DataFrame, txns: pd.DataFrame, end_date: pd.Timestamp) -> MetricResult:
    lb = latest_balance(bal).value
    out = avg_daily_outflow_30d(txns, end_date).value
    if lb is None or out is None:
        return MetricResult(None, "Need balances and transactions for buffer days.")
    if out == 0:
        return MetricResult(float("inf"), "No recent outflows; buffer is effectively infinite.")
    return MetricResult(float(lb / out), "Latest balance ÷ avg daily outflow (30d).")


def overdraft_nsf_90d(txns: pd.DataFrame, end_date: pd.Timestamp) -> MetricResult:
    w = _window(txns, "post_date", 90, end_date)
    if w.empty:
        return MetricResult(0.0, "No transactions in last 90 days.")
    cols = [c for c in ["type", "txn_code", "description"] if c in w.columns]
    if not cols:
        return MetricResult(None, "No type/txn_code/description column available to detect overdraft/NSF.")
    blob = w[cols].astype(str).agg(" ".join, axis=1).str.upper()
    keywords = ["OD", "OVERDRAFT", "NSF", "RETURNED ITEM", "RETURN ITEM"]
    mask = np.zeros(len(blob), dtype=bool)
    for k in keywords:
        mask |= blob.str.contains(k, na=False)
    return MetricResult(float(mask.sum()), "Count of OD/NSF-like events (90d) using code/type/description match.")


def drawdown_events_90d(
    bal: pd.DataFrame,
    end_date: pd.Timestamp,
    drop_pct: float = 0.30,
    window_days: int = 7,
) -> MetricResult:
    w = _window(bal, "date", 90, end_date).sort_values("date")
    if len(w) < 10:
        return MetricResult(None, "Insufficient balance history in last 90 days.")
    w = w.reset_index(drop=True)
    count = 0
    i = 0
    while i < len(w):
        start_bal = w.loc[i, "balance"]
        start_date = w.loc[i, "date"]
        j_end = i
        while j_end < len(w) and (w.loc[j_end, "date"] - start_date) <= pd.Timedelta(days=window_days):
            j_end += 1
        if j_end <= i + 1:
            i += 1
            continue
        min_bal = w.loc[i:j_end - 1, "balance"].min()
        if start_bal > 0 and (start_bal - min_bal) / start_bal >= drop_pct:
            count += 1
            i = j_end
        else:
            i += 1
    return MetricResult(float(count), f"Count of ≥{int(drop_pct * 100)}% balance drops within {window_days} days (90d).")


def avg_recovery_days_90d(
    bal: pd.DataFrame,
    end_date: pd.Timestamp,
    baseline_30d: float | None,
    drop_pct: float = 0.30,
    window_days: int = 7,
    max_search_days: int = 60,
) -> MetricResult:
    if baseline_30d is None:
        return MetricResult(None, "Need baseline balance to compute recovery.")
    w = _window(bal, "date", 150, end_date).sort_values("date").reset_index(drop=True)
    if len(w) < 20:
        return MetricResult(None, "Insufficient balance history to compute recovery.")
    recoveries = []
    i = 0
    while i < len(w):
        start_bal = w.loc[i, "balance"]
        start_date = w.loc[i, "date"]
        j_end = i
        while j_end < len(w) and (w.loc[j_end, "date"] - start_date) <= pd.Timedelta(days=window_days):
            j_end += 1
        if j_end <= i + 1:
            i += 1
            continue
        min_bal = w.loc[i:j_end - 1, "balance"].min()
        if start_bal > 0 and (start_bal - min_bal) / start_bal >= drop_pct:
            k = j_end
            recovered = None
            while k < len(w) and (w.loc[k, "date"] - start_date) <= pd.Timedelta(days=max_search_days):
                if w.loc[k, "balance"] >= baseline_30d:
                    recovered = (w.loc[k, "date"] - start_date).days
                    break
                k += 1
            if recovered is not None:
                recoveries.append(recovered)
            i = j_end
        else:
            i += 1
    if not recoveries:
        return MetricResult(None, "No recoverable drawdowns found (or no recovery within window).")
    return MetricResult(float(np.mean(recoveries)), "Average days to recover to baseline after drawdown (cap 60d).")


def compute_all_metrics_for_account(accounts_row: dict, txns_acc: pd.DataFrame, bal_acc: pd.DataFrame) -> dict:
    if not bal_acc.empty:
        end_date = pd.to_datetime(bal_acc["date"]).max()
    elif not txns_acc.empty:
        end_date = pd.to_datetime(txns_acc["post_date"]).max()
    else:
        end_date = pd.Timestamp.today().normalize()

    base = baseline_balance_30d(bal_acc, end_date).value
    vol = volatility_balance_30d(bal_acc, end_date).value
    buf = cash_buffer_days(bal_acc, txns_acc, end_date).value
    dd = drawdown_events_90d(bal_acc, end_date).value
    od = overdraft_nsf_90d(txns_acc, end_date).value
    rec = avg_recovery_days_90d(bal_acc, end_date, base).value
    lb = latest_balance(bal_acc).value

    return {
        "account_id": accounts_row.get("account_id"),
        "customer_type": accounts_row.get("customer_type", None),

        # Loan presence (used for loan-only drill-down filtering)
        "has_loan": accounts_row.get("has_loan", None),

        # Loan exposure (outstanding principal)
        "loan_balance": accounts_row.get("loan_balance", None),

        # Deposit / account info
        "product": accounts_row.get("product", None),
        "interest_rate": accounts_row.get("interest_rate", None),

        # Loan-side risk info (NEW)
        "risk_rating": accounts_row.get("risk_rating", None),
        "days_past_due": accounts_row.get("days_past_due", None),

        "as_of": str(end_date.date()),
        "latest_balance": lb,
        "baseline_balance_30d": base,
        "balance_volatility_30d": vol,
        "cash_buffer_days": buf,
        "drawdown_count_90d": dd,
        "avg_recovery_days_90d": rec,
        "overdraft_nsf_90d": od,
    }


def compute_portfolio_metrics(accounts: pd.DataFrame, txns: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in accounts.iterrows():
        acc_id = str(a["account_id"])
        txns_acc = txns[txns["account_id"] == acc_id].copy()
        bal_acc = bal[bal["account_id"] == acc_id].copy()
        rows.append(compute_all_metrics_for_account(a.to_dict(), txns_acc, bal_acc))
    return pd.DataFrame(rows)


# -------------------------
# WAIR METRICS
# -------------------------


def weighted_avg_interest_rate_overall(portfolio: pd.DataFrame) -> float | None:
    if portfolio.empty:
        return None
    if "interest_rate" not in portfolio.columns or "loan_balance" not in portfolio.columns:
        return None

    df = portfolio.dropna(subset=["interest_rate", "loan_balance"]).copy()
    if df.empty:
        return None
    df = df[df["loan_balance"] > 0]
    if df.empty:
        return None

    weights = df["loan_balance"].astype(float)
    rates = df["interest_rate"].astype(float)
    return float((rates * weights).sum() / weights.sum())


def weighted_avg_interest_rate_by_product(portfolio: pd.DataFrame) -> pd.DataFrame:
    needed = {"product", "interest_rate", "loan_balance"}
    if portfolio.empty or not needed.issubset(set(portfolio.columns)):
        return pd.DataFrame(columns=["product", "wair_pct", "total_balance", "account_count"])

    df = portfolio.dropna(subset=["product", "interest_rate", "loan_balance"]).copy()
    df = df[df["loan_balance"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["product", "wair_pct", "total_balance", "account_count"])

    rows = []
    for product, g in df.groupby("product"):
        weights = g["loan_balance"].astype(float)
        rates = g["interest_rate"].astype(float)
        wair = float((rates * weights).sum() / weights.sum()) if weights.sum() != 0 else None
        rows.append({
            "product": product,
            "wair_pct": wair,
            "total_balance": float(weights.sum()),
            "account_count": int(len(g)),
        })

    out = pd.DataFrame(rows).sort_values("total_balance", ascending=False)
    return out


# -------------------------
# DERIVED RISK RATING (RULES)
# -------------------------


def derive_risk_rating(metrics_row: dict) -> tuple[int | None, list[str]]:
    """Derive an explainable 1–9 risk rating from existing metrics.

    This is meant for prototypes and demos where a risk band is helpful even when
    no risk_rating is supplied in the input accounts.csv.

    Higher number = higher risk.
    """
    reasons: list[str] = []
    points = 0

    buf = metrics_row.get("cash_buffer_days")
    dd = metrics_row.get("drawdown_count_90d")
    rec = metrics_row.get("avg_recovery_days_90d")
    od = metrics_row.get("overdraft_nsf_90d")
    base = metrics_row.get("baseline_balance_30d")
    vol = metrics_row.get("balance_volatility_30d")
    lb = metrics_row.get("latest_balance")
    dpd = metrics_row.get("days_past_due")

    # Delinquency is the strongest loan signal when present.
    if dpd is not None:
        try:
            dpd_v = float(dpd)
            if dpd_v >= 60:
                points += 4
                reasons.append("60+ days past due.")
            elif dpd_v >= 30:
                points += 3
                reasons.append("30+ days past due.")
            elif dpd_v >= 10:
                points += 1
                reasons.append("10+ days past due.")
            elif dpd_v > 0:
                points += 1
                reasons.append("Past due (>0 days).")
        except Exception:
            pass

    # Cash buffer
    if buf is not None:
        try:
            buf_v = float(buf)
            if buf_v < 3:
                points += 3
                reasons.append("Low cash buffer (<3 days).")
            elif buf_v < 7:
                points += 2
                reasons.append("Moderate cash buffer (<7 days).")
            elif buf_v < 10:
                points += 1
                reasons.append("Thin cash buffer (<10 days).")
            elif buf_v >= 14:
                points -= 1
                reasons.append("Strong cash buffer (≥14 days).")
        except Exception:
            pass

    # Drawdowns
    if dd is not None:
        try:
            dd_v = float(dd)
            if dd_v >= 2:
                points += 2
                reasons.append("Multiple drawdowns (90d).")
            elif dd_v == 1:
                points += 1
                reasons.append("Drawdown detected (90d).")
        except Exception:
            pass

    # Recovery
    if rec is not None:
        try:
            rec_v = float(rec)
            if rec_v > 30:
                points += 2
                reasons.append("Slow recovery after drawdowns (>30 days).")
            elif rec_v > 14:
                points += 1
                reasons.append("Moderate recovery time (>14 days).")
        except Exception:
            pass

    # Overdraft/NSF
    if od is not None:
        try:
            od_v = float(od)
            if od_v >= 2:
                points += 2
                reasons.append("Multiple overdraft/NSF-like events (90d).")
            elif od_v == 1:
                points += 1
                reasons.append("Overdraft/NSF-like event (90d).")
        except Exception:
            pass

    # Volatility relative to baseline
    if vol is not None and base is not None:
        try:
            rel = abs(float(vol)) / (abs(float(base)) + 1e-6)
            if rel > 0.75:
                points += 2
                reasons.append("Very high volatility vs baseline.")
            elif rel > 0.40:
                points += 1
                reasons.append("High volatility vs baseline.")
        except Exception:
            pass

    # Latest balance vs baseline
    if lb is not None and base is not None:
        try:
            base_v = float(base)
            lb_v = float(lb)
            if base_v != 0:
                drop = (base_v - lb_v) / abs(base_v)
                if drop > 0.35:
                    points += 2
                    reasons.append("Latest balance far below baseline.")
                elif drop > 0.20:
                    points += 1
                    reasons.append("Latest balance below baseline.")
                elif drop < -0.10:
                    points -= 1
                    reasons.append("Latest balance above baseline.")
        except Exception:
            pass

    # If we have essentially no usable inputs, return None.
    if not reasons:
        return None, []

    # Map points into 1..9.
    rating = int(max(1, min(9, 5 + points)))
    return rating, reasons[:4]


def add_derived_risk_rating(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Add `risk_rating_derived` and `risk_rating_reasons` columns."""
    if portfolio.empty:
        out = portfolio.copy()
        out["risk_rating_derived"] = []
        out["risk_rating_reasons"] = []
        return out

    ratings: list[int | None] = []
    reasons: list[str] = []
    for _, row in portfolio.iterrows():
        r, why = derive_risk_rating(row.to_dict())
        ratings.append(r)
        reasons.append("; ".join(why) if why else "")
    out = portfolio.copy()
    out["risk_rating_derived"] = ratings
    out["risk_rating_reasons"] = reasons
    return out
