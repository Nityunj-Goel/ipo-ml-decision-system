import math

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict

from configs.feature_config import RAW_FEATURES, TARGET
from src.models.trainer import train
from src.portfolio.allocator import compute_allocation
from src.utils.utils import load_config, load_raw_dataset


def run_backtest(
    model_type: str,
    t_min: float | None = None,
    alpha: float | None = None,
    n_splits: int | None = None,
    gap: int | None = None,
    holdout_fraction: float | None = None,
    listing_gain_threshold_perc: float | None = None,
    verbose: bool = False,
    from_year: int | None = None,
    **model_kwargs,
) -> dict:
    """Backtest the full trading strategy across time-series folds.

    For each fold the classifier is trained on the training window, then
    probabilities are predicted on the test window.  Test IPOs are grouped
    by listing date (the decision day).  For each day the allocator
    computes portfolio weights and the weighted return is recorded.  The
    mean daily return per fold is aggregated across all folds.

    Args:
        model_type: Pipeline key (e.g. ``"logistic_regression"``).
        t_min: Minimum probability threshold passed to the allocator
            (config default if None).
        alpha: Concentration exponent passed to the allocator
            (config default if None).
        n_splits: Number of TimeSeriesSplit folds (config default if None).
        gap: Gap between train and test windows (config default if None).
        holdout_fraction: Fraction of recent rows held out (config default
            if None).
        listing_gain_threshold_perc: Binarization threshold for the
            classifier target (config default if None).
        verbose: Print per-fold and per-day details.
        **model_kwargs: Forwarded to the model trainer.

    Returns:
        Dictionary with per-fold details, overall mean return, and the
        holdout split for further analysis.
    """
    df = load_raw_dataset(from_year)
    config = load_config()
    listing_date_col = RAW_FEATURES["listing_date"]
    cv_cfg = config["cv"]
    portfolio_cfg = config["portfolio"]

    t_min = t_min if t_min is not None else portfolio_cfg["trade_threshold"]
    alpha = alpha if alpha is not None else portfolio_cfg["alpha"]
    n_splits = n_splits if n_splits is not None else cv_cfg["n_splits"]
    gap = gap if gap is not None else cv_cfg["gap"]
    holdout_fraction = holdout_fraction if holdout_fraction is not None else cv_cfg["holdout_fraction"]
    threshold = listing_gain_threshold_perc if listing_gain_threshold_perc is not None else config["target"]["listing_gain_threshold_perc"]

    df[listing_date_col] = pd.to_datetime(df[listing_date_col]).dt.tz_localize(None)
    end_date_col = RAW_FEATURES["ipo_end_date"]
    df[end_date_col] = pd.to_datetime(df[end_date_col], errors="coerce").dt.tz_localize(None)

    # impute missing end dates using median gap from listing date
    known = df.dropna(subset=[end_date_col])
    known = known[known[listing_date_col] >= known[end_date_col]]
    median_gap = (known[listing_date_col] - known[end_date_col]).median()
    mask = df[end_date_col].isna()
    df.loc[mask, end_date_col] = df.loc[mask, listing_date_col] - median_gap

    df = df.sort_values(end_date_col).reset_index(drop=True)

    # hold out most-recent IPOs
    holdout_start = len(df) - math.ceil(len(df) * holdout_fraction)
    X_cv = df.iloc[:holdout_start]
    X_holdout = df.iloc[holdout_start:]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_cv), start=1):
        X_train = X_cv.iloc[tr_idx]
        X_val = X_cv.iloc[val_idx]

        # 1. train classifier
        pipeline = train(
            X_train, model_type,
            listing_gain_threshold_perc=threshold,
            **model_kwargs,
        )

        # 2. predict probabilities on test set
        pos_idx = list(pipeline.classes_).index(1)
        val_probs = pipeline.predict_proba(X_val)[:, pos_idx]
        actual_returns = _compute_listing_gain_pct(X_val)

        val_df = pd.DataFrame({
            "prob": val_probs,
            "actual_return": actual_returns.values,
            "date": X_val[end_date_col].values,
        })

        # 3. group by decision day (listing date)
        day_returns, daily_allocations = [], []
        for day, day_df in val_df.groupby("date", sort=False):
            probs = day_df["prob"].values
            returns = day_df["actual_return"].values

            # 4. compute allocation
            weights = compute_allocation(probs, t_min=t_min, alpha=alpha, normalize=True)

            # 5. portfolio return for this day
            portfolio_return_day = float(np.sum(weights * returns))
            day_returns.append(portfolio_return_day)
            daily_allocations.append(weights.sum())

            # if verbose:
            #     print(
            #         f"  Fold {fold} | {day.date()} | "
            #         f"#IPOs={len(day_df)} | "
            #         f"allocated={weights.sum():.2f} | "
            #         f"return={portfolio_return_day:+.2f}%"
            #     )

        # 6. Financial metrics for this fold
        metrics = compute_portfolio_metrics(day_returns, daily_allocations)

        fold_results.append({
            "fold": fold,
            **metrics
        })

    fold_df = pd.DataFrame(fold_results)

    # train final pipeline on full CV window
    final_pipeline = train(X_cv, model_type, listing_gain_threshold_perc=threshold, **model_kwargs)

    if verbose:
        print("=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Model:       {model_type}")
        print(f"  t_min:       {t_min}")
        print(f"  alpha:       {alpha}")
        print(f"  Folds:       {n_splits}")
        for _, row in fold_df.iterrows():
            print(
                f"  Fold {int(row['fold'])}: "
                f"{int(row['num_days'])} days | "
                f"mean_return = {row['mean_daily_return']:+.2f}% | "
                f"win_rate = {row['win_rate']:.2%} | "
                f"avg_alloc = {row['avg_allocation']:.2f} | "
                f"traded_days = {row['pct_days_traded']:.2%} | "
                f"volatility = {row['volatility']:.2f} | "
                f"sharpe = {row['sharpe_like']:.2f}"
            )
        print(f"\nOverall mean daily return: {fold_df['mean_daily_return'].mean():+.2f}%")
        print(f"Overall mean win rate: {fold_df['win_rate'].mean():.2%}")
        print(f"Overall average allocation: {fold_df['avg_allocation'].mean():+.2f}")
        print(f"Overall mean % days traded: {fold_df['pct_days_traded'].mean():.2%}")
        print(f"Overall mean volatility: {fold_df['volatility'].mean():+.2f}")
        print(f"Overall mean sharpe like: {fold_df['sharpe_like'].mean():+.2f}")
        print("=" * 60)

    return {
        "fold_results": fold_df,
        "final_pipeline": final_pipeline,
        "holdout": X_holdout,
    }

def compute_portfolio_metrics(day_returns: List[float], daily_allocations: List[float]) -> Dict[str, float]:
    """
    Compute key business metrics for the IPO allocation strategy.

    Parameters
    ----------
    day_returns : list or np.ndarray
        Portfolio return (%) for each decision day, computed as:
            Σ (w_i × actual_return_i)

    daily_allocations : list or np.ndarray
        Total capital allocated each day, computed as:
            Σ w_i
        (values in [0, 1], where 1 means fully deployed capital)

    Returns
    -------
    dict
        Dictionary containing:

        - mean_daily_return : float
            Average return per decision day (%)

        - cumulative_return : float
            Compounded return across all days (%)

        - num_days : int
            Number of decision days

        - pct_days_traded : float
            Fraction of days where some capital was deployed

        - avg_allocation : float
            Average capital deployed per day

        - win_rate : float
            Fraction of days with positive portfolio return

        - volatility : float
            Standard deviation of daily returns (%)

        - sharpe_like : float
            Risk-adjusted return (mean / std)
    """
    day_returns = np.array(day_returns)
    daily_allocations = np.array(daily_allocations)

    num_days = len(day_returns)

    mean_daily_return = np.mean(day_returns) if num_days else 0.0

    pct_days_traded = np.mean(daily_allocations > 0) if num_days else 0.0

    avg_allocation = np.mean(daily_allocations) if num_days else 0.0

    win_rate = np.mean(day_returns > 0) if num_days else 0.0

    volatility = np.std(day_returns) if num_days else 0.0

    sharpe_like = (
        (mean_daily_return / volatility) if volatility > 0 else 0.0
    )

    return {
        "mean_daily_return": mean_daily_return,
        "num_days": num_days,
        "pct_days_traded": pct_days_traded,
        "avg_allocation": avg_allocation,
        "win_rate": win_rate,
        "volatility": volatility,
        "sharpe_like": sharpe_like,
    }

def _compute_listing_gain_pct(df: pd.DataFrame) -> pd.Series:
    return (
        (df[TARGET["listing_price"]] - df[TARGET["issue_price"]])
        / df[TARGET["issue_price"]]
        * 100
    )


def run_detailed_backtest(
    model_type: str,
    t_min: float | None = None,
    alpha: float | None = None,
    n_splits: int | None = None,
    gap: int | None = None,
    holdout_fraction: float | None = None,
    listing_gain_threshold_perc: float | None = None,
    from_year: int | None = None,
    **model_kwargs,
) -> dict:
    """Walk-forward backtest producing a flat per-(date, IPO) trade ledger.

    Each fold trains on the fold's train window and scores its val window,
    giving every CV date a prediction from a model that never saw it.
    The final pipeline is fit on the full CV window and used to score
    the held-out tail. Holdout rows are tagged with ``is_holdout=True``.

    Returns:
        dict with:
          - ``trades`` (pd.DataFrame): columns
            ``[date, company, prob, weight, actual_return_pct,
              contribution_pct, allocated, is_holdout]``.
          - ``final_pipeline`` (sklearn.Pipeline): fit on full CV window.
          - ``meta`` (dict): params + data range summary.
    """
    df = load_raw_dataset(from_year)
    config = load_config()
    listing_date_col = RAW_FEATURES["listing_date"]
    end_date_col = RAW_FEATURES["ipo_end_date"]
    company_col = RAW_FEATURES["company"]
    cv_cfg = config["cv"]
    portfolio_cfg = config["portfolio"]

    t_min = t_min if t_min is not None else portfolio_cfg["trade_threshold"]
    alpha = alpha if alpha is not None else portfolio_cfg["alpha"]
    n_splits = n_splits if n_splits is not None else cv_cfg["n_splits"]
    gap = gap if gap is not None else cv_cfg["gap"]
    holdout_fraction = holdout_fraction if holdout_fraction is not None else cv_cfg["holdout_fraction"]
    threshold = listing_gain_threshold_perc if listing_gain_threshold_perc is not None else config["target"]["listing_gain_threshold_perc"]

    df[listing_date_col] = pd.to_datetime(df[listing_date_col]).dt.tz_localize(None)
    df[end_date_col] = pd.to_datetime(df[end_date_col], errors="coerce").dt.tz_localize(None)

    # impute missing end dates using median gap from listing date
    known = df.dropna(subset=[end_date_col])
    known = known[known[listing_date_col] >= known[end_date_col]]
    median_gap = (known[listing_date_col] - known[end_date_col]).median()
    mask = df[end_date_col].isna()
    df.loc[mask, end_date_col] = df.loc[mask, listing_date_col] - median_gap

    df = df.sort_values(end_date_col).reset_index(drop=True)

    holdout_start = len(df) - math.ceil(len(df) * holdout_fraction)
    X_cv = df.iloc[:holdout_start]
    X_holdout = df.iloc[holdout_start:]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    all_trades = []

    # Walk-forward CV folds
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_cv), start=1):
        X_train = X_cv.iloc[tr_idx]
        X_val = X_cv.iloc[val_idx]
        pipeline = train(X_train, model_type, listing_gain_threshold_perc=threshold, **model_kwargs)
        fold_trades = _score_and_allocate(
            pipeline, X_val, t_min, alpha,
            end_date_col, company_col, is_holdout=False,
        )
        all_trades.append(fold_trades)

    # Final pipeline: train on full CV window, score the holdout
    final_pipeline = train(X_cv, model_type, listing_gain_threshold_perc=threshold, **model_kwargs)
    holdout_trades = _score_and_allocate(
        final_pipeline, X_holdout, t_min, alpha,
        end_date_col, company_col, is_holdout=True,
    )
    all_trades.append(holdout_trades)

    trades_df = pd.concat(all_trades, ignore_index=True).sort_values("date").reset_index(drop=True)

    meta = {
        "alpha": alpha,
        "t_min": t_min,
        "listing_gain_threshold_perc": threshold,
        "model_type": model_type,
        "n_splits": n_splits,
        "gap": gap,
        "holdout_fraction": holdout_fraction,
        "data_range": {
            "from": str(trades_df["date"].min().date()),
            "to": str(trades_df["date"].max().date()),
        },
        "last_training_date": str(X_cv[end_date_col].max().date()),
        "num_ipos_backtested": int(len(trades_df)),
    }

    return {"trades": trades_df, "final_pipeline": final_pipeline, "meta": meta}


def _score_and_allocate(
    pipeline,
    X: pd.DataFrame,
    t_min: float,
    alpha: float,
    end_date_col: str,
    company_col: str,
    is_holdout: bool,
) -> pd.DataFrame:
    """Score X, allocate per decision day, return flat per-IPO rows."""
    pos_idx = list(pipeline.classes_).index(1)
    probs = pipeline.predict_proba(X)[:, pos_idx]
    actual_returns = _compute_listing_gain_pct(X).values

    scored = pd.DataFrame({
        "date": X[end_date_col].values,
        "company": X[company_col].values,
        "prob": probs,
        "actual_return_pct": actual_returns,
        "is_holdout": is_holdout,
    })

    def _alloc_one_day(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["weight"] = compute_allocation(
            group["prob"].values, t_min=t_min, alpha=alpha, normalize=True,
        )
        return group

    scored = scored.groupby("date", group_keys=False).apply(_alloc_one_day)
    scored["contribution_pct"] = scored["weight"] * scored["actual_return_pct"]
    scored["allocated"] = scored["weight"] > 0

    # Only reorders columns
    return scored[[
        "date", "company", "prob", "weight",
        "actual_return_pct", "contribution_pct",
        "allocated", "is_holdout",
    ]]