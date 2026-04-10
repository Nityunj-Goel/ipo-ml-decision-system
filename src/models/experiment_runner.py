import math
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

from configs.feature_config import RAW_FEATURES
from src.models.eval import plot_roc_curve_and_auc
from src.pipelines import data_pipeline
from src.utils.utils import load_config, load_raw_dataset
from src.models.trainer import train


def run_cv_experiment(
    model_type: str,
    n_splits: int | None = None,
    gap: int | None = None,
    holdout_fraction: float | None = None,
    listing_gain_threshold_perc: float | None = None,
    verbose: bool = False,
    **model_kwargs
):
    """Run time-series cross-validation for a model and train a final estimator.

    Args:
        model_type: Identifier of the model pipeline to train.
        n_splits: Number of TimeSeriesSplit folds. Uses config default when None.
        gap: Number of samples excluded between train and validation windows.
            Uses config default when None.
        holdout_fraction: Fraction of most recent IPO rows reserved as holdout.
            Uses config default when None.
        listing_gain_threshold_perc: Threshold used to binarize listing gain
            target. Uses config default when None.
        verbose: If True, prints per-fold summary and averaged classification
            metrics.
        **model_kwargs: Extra keyword arguments forwarded to the model trainer.

    Returns:
        A dictionary containing fold-level AUC metrics, aggregated CV AUC
        statistics, the final pipeline fitted on the CV window, and the
        holdout split as (X_holdout, y_holdout).
    """
    df = load_raw_dataset()
    config = load_config()
    date_col = RAW_FEATURES['listing_date']
    cv_cfg = config["cv"]
    n_splits = n_splits if n_splits is not None else cv_cfg["n_splits"]
    gap = gap if gap is not None else cv_cfg["gap"]
    holdout_fraction = holdout_fraction if holdout_fraction is not None else cv_cfg["holdout_fraction"]
    threshold = listing_gain_threshold_perc if listing_gain_threshold_perc is not None else config["target"]["listing_gain_threshold_perc"]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    y = data_pipeline.compute_target(df, listing_gain_threshold=threshold)

    # Hold out the most recent IPOs for future evaluation
    holdout_start = len(df) - math.ceil(len(df) * holdout_fraction)
    X_cv, X_holdout = df.iloc[:holdout_start], df.iloc[holdout_start:]
    y_cv, y_holdout = y.iloc[:holdout_start], y.iloc[holdout_start:]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics = []
    fold_reports = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_cv), start=1):
        X_train, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
        y_val = y_cv.iloc[val_idx]

        pipeline = train(X_train, model_type, listing_gain_threshold_perc=threshold, **model_kwargs)

        y_pred = pipeline.predict(X_val)
        pos_idx = list(pipeline.classes_).index(1)
        y_proba = pipeline.predict_proba(X_val)[:, pos_idx]

        plot_roc = verbose and (fold == n_splits)
        auc = plot_roc_curve_and_auc(y_val, y_proba, plot_roc_curve=plot_roc)

        fold_metrics.append({
            "fold": fold,
            "auc": auc,
        })

        if verbose:
            fold_reports.append(classification_report(y_val, y_pred, output_dict=True))

    fold_df = pd.DataFrame(fold_metrics)
    cv_mean_auc = fold_df["auc"].mean()
    cv_std_auc = fold_df["auc"].std()

    if verbose:
        _print_cv_summary(fold_df, cv_mean_auc, cv_std_auc, fold_reports)

    final_pipeline = train(X_cv, model_type, listing_gain_threshold_perc=threshold, **model_kwargs)

    return {
        "fold_metrics": fold_df,
        "cv_mean_auc": cv_mean_auc,
        "cv_std_auc": cv_std_auc,
        "final_pipeline": final_pipeline,
        "holdout": (X_holdout, y_holdout),
    }


def _print_cv_summary(fold_df, mean_auc, std_auc, fold_reports):
    print("=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print("\nPer-Fold AUC:")
    for _, row in fold_df.iterrows():
        print(f"  Fold {int(row['fold'])}: AUC = {row['auc']:.4f}")

    print(f"\nMean AUC: {mean_auc:.4f} +/- Std Dev {std_auc:.4f}")

    if fold_reports:
        dict_keys = [k for k in fold_reports[0] if isinstance(fold_reports[0][k], dict)]
        avg_report = {}
        for key in dict_keys:
            avg_report[key] = {}
            for metric in fold_reports[0][key]:
                values = [r[key][metric] for r in fold_reports if key in r]
                avg_report[key][metric] = np.mean(values)

        print("\nAveraged Classification Metrics Across Folds:")
        print(f"  {'':15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
        print(f"  {'-' * 55}")
        for key in dict_keys:
            print(f"  {str(key):15s} {avg_report[key]['precision']:10.4f} {avg_report[key]['recall']:10.4f} {avg_report[key]['f1-score']:10.4f} {avg_report[key]['support']:10.1f}")

        mean_acc = np.mean([r['accuracy'] for r in fold_reports])
        print(f"  {'accuracy':15s} {mean_acc:10.4f}")

    print("=" * 60)
