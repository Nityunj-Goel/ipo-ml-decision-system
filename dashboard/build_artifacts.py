"""
Build all Streamlit dashboard artifacts in one offline pass.

Outputs to ``artifacts/dashboard/``:
  - ``trades.csv``  per-(date, IPO) walk-forward trade ledger.
  - ``meta.json``   config snapshot, holdout KPIs, example trading day.

Run:
    python -m dashboard.build_artifacts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio.backtester import run_detailed_backtest
from src.utils.utils import get_project_root, load_config


def main(model_type: str = "logistic_regression") -> None:
    root = get_project_root()
    cfg = load_config()
    out_dir = root / cfg["paths"]["dashboard_artifacts"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running detailed walk-forward backtest ({model_type})...")
    result = run_detailed_backtest(model_type=model_type, **cfg[model_type])
    trades: pd.DataFrame = result["trades"]
    meta: dict = result["meta"]

    earliest_year = cfg["dashboard"]["earliest_reporting_year"]
    trades = trades[trades["date"].dt.year >= earliest_year].reset_index(drop=True)
    meta["num_ipos_backtested"] = int(len(trades))
    meta["data_range"] = {
        "from": str(trades["date"].min().date()),
        "to": str(trades["date"].max().date()),
    }

    trades_path = out_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)
    print(f"Wrote {trades_path}  ({len(trades)} rows)")

    # Holdout KPIs from the held-out slice of the ledger
    holdout = trades[trades["is_holdout"]]
    meta["holdout"] = _compute_kpis(holdout)

    # Example trading day: pick a day with 3-5 IPOs and mixed outcomes
    sample_src = holdout if len(holdout) else trades
    meta["example_day"] = _pick_example_day(sample_src)

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote {meta_path}")


def _compute_kpis(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    daily = trades.groupby("date").agg(
        portfolio_return=("contribution_pct", "sum"),
        total_allocation=("weight", "sum"),
    ).reset_index()

    returns = daily["portfolio_return"].values
    allocations = daily["total_allocation"].values
    num_days = len(returns)
    mean_r = float(np.mean(returns)) if num_days else 0.0
    vol = float(np.std(returns)) if num_days else 0.0
    calendar_days = (
        int((daily["date"].max() - daily["date"].min()).days) + 1
        if num_days else 0
    )

    return {
        "num_days": num_days,          # IPO trading days (days with >=1 IPO)
        "calendar_days": calendar_days,
        "num_ipos": int(len(trades)),
        "mean_daily_return": mean_r,
        "win_rate": float(np.mean(returns > 0)) if num_days else 0.0,
        "avg_allocation": float(np.mean(allocations)) if num_days else 0.0,
        "pct_days_traded": float(np.mean(allocations > 0)) if num_days else 0.0,
        "volatility": vol,
        "sharpe_like": float(mean_r / vol) if vol > 0 else 0.0,
    }


def _pick_example_day(trades: pd.DataFrame) -> dict | None:
    """A day with 3-5 IPOs, preferring mixed winners/losers for illustration."""
    if trades.empty:
        return None

    counts = trades.groupby("date").size()
    candidates = counts[(counts >= 3) & (counts <= 5)].index.tolist()
    if not candidates:
        candidates = counts.sort_values(ascending=False).head(5).index.tolist()

    chosen = None
    for day in candidates:
        day_df = trades[trades["date"] == day]
        has_win = (day_df["actual_return_pct"] > 0).any()
        has_loss = (day_df["actual_return_pct"] <= 0).any()
        if has_win and has_loss:
            chosen = day
            break
    if chosen is None:
        chosen = candidates[0]

    day_df = trades[trades["date"] == chosen].sort_values("prob", ascending=False)
    return {
        "date": str(pd.Timestamp(chosen).date()),
        "ipos": [
            {
                "company": str(row["company"]),
                "prob": round(float(row["prob"]), 4),
                "weight": round(float(row["weight"]), 4),
                "actual_return_pct": round(float(row["actual_return_pct"]), 2),
            }
            for _, row in day_df.iterrows()
        ],
    }


if __name__ == "__main__":
    main()
