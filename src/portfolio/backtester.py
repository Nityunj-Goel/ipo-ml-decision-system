import math

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from configs.feature_config import RAW_FEATURES, TARGET
from src.models.trainer import train
from src.portfolio.allocator import compute_allocation
from src.utils.utils import load_config, load_raw_dataset


def _compute_listing_gain_pct(df: pd.DataFrame) -> pd.Series:
    return (
        (df[TARGET["listing_price"]] - df[TARGET["issue_price"]])
        / df[TARGET["issue_price"]]
        * 100
    )


def run_backtest(
    model_type: str,
    t_min: float,
    alpha: float,
    n_splits: int | None = None,
    gap: int | None = None,
    holdout_fraction: float | None = None,
    listing_gain_threshold_perc: float | None = None,
    verbose: bool = False,
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
        t_min: Minimum probability threshold passed to the allocator.
        alpha: Concentration exponent passed to the allocator.
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
    df = load_raw_dataset()
    config = load_config()
    listing_date_col = RAW_FEATURES["listing_date"]
    cv_cfg = config["cv"]

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
        day_returns = []
        for day, day_df in val_df.groupby("date", sort=False):
            probs = day_df["prob"].values
            returns = day_df["actual_return"].values

            # 4. compute allocation
            weights = compute_allocation(probs, t_min=t_min, alpha=alpha)

            # 5. portfolio return for this day
            portfolio_return_day = float(np.sum(weights * returns))
            day_returns.append(portfolio_return_day)

            # if verbose:
            #     print(
            #         f"  Fold {fold} | {day.date()} | "
            #         f"#IPOs={len(day_df)} | "
            #         f"allocated={weights.sum():.2f} | "
            #         f"return={portfolio_return_day:+.2f}%"
            #     )

        # 6. mean daily return for this fold
        mean_fold_return = float(np.mean(day_returns)) if day_returns else 0.0

        fold_results.append({
            "fold": fold,
            "n_days": len(day_returns),
            "mean_daily_return": mean_fold_return,
        })

        if verbose:
            print(
                f"  Fold {fold} summary: {len(day_returns)} days, "
                f"mean daily return = {mean_fold_return:+.2f}%\n"
            )

    fold_df = pd.DataFrame(fold_results)
    overall_mean_return = fold_df["mean_daily_return"].mean()

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
                f"{int(row['n_days'])} days, "
                f"mean return = {row['mean_daily_return']:+.2f}%"
            )
        print(f"\n  Overall mean daily return: {overall_mean_return:+.2f}%")
        print("=" * 60)

    return {
        "fold_results": fold_df,
        "mean_daily_return": overall_mean_return,
        "final_pipeline": final_pipeline,
        "holdout": X_holdout,
    }
