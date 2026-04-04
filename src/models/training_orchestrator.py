import math
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from configs.feature_config import RAW_FEATURES
from src.models.eval import plot_roc_curve_and_auc
from src.pipelines import data_pipeline, pipeline_factory
from src.utils.utils import load_config, load_raw_dataset
from src.models.trainer import train


def run_time_series_training_loop(
    model_type: str,
    n_splits: int = 5,
    holdout_size: float = 0.15,
    listing_gain_threshold_perc: float | None = None,
    plot_holdout_roc: bool = True,  # might set this to false
):
    # Load the raw dataset and configuration needed to derive the target label.
    df = load_raw_dataset()
    config = load_config()
    date_col = RAW_FEATURES['ipo_start_date']
    threshold = listing_gain_threshold_perc
    if threshold is None:
        threshold = config["target"]["listing_gain_threshold_perc"]

    # Sorting the dataset by date helps achieve time series split cross validation
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    y = data_pipeline.compute_target(df, listing_gain_threshold=threshold)

    # Holdout the latest IPOs as test set
    holdout_start = len(df) - math.ceil(len(df) * holdout_size)
    X_cv, X_holdout = df.iloc[:holdout_start], df.iloc[holdout_start:]
    y_cv, y_holdout = y.iloc[:holdout_start], y.iloc[holdout_start:]

    ############################################ CHECKPOINT ############################################

    # Build a forward-chaining splitter and collect metrics from each
    # validation fold.
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_cv), start=1):
        # Slice the chronological training and validation windows for this fold.
        X_train, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
        y_train, y_val = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]

        # Create and fit a fresh pipeline so each fold is trained independently.
        pipeline = pipeline_factory.get_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        # Generate class predictions and positive-class probabilities for fold
        # level accuracy and ROC AUC measurement.
        y_pred = pipeline.predict(X_val)
        pos_idx = list(pipeline.classes_).index(1)
        y_proba = pipeline.predict_proba(X_val)[:, pos_idx]

        auc = plot_roc_curve_and_auc(y_val, y_proba, plot_roc_curve=False)

        # Store the key metrics for later aggregation across all folds.
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_val, y_pred),
                "auc": auc,
            }
        )

    # Retrain the selected pipeline on the entire cross-validation window
    # before evaluating on the unseen holdout set.
    final_pipeline = pipeline_factory.get_pipeline(model_type)
    final_pipeline.fit(X_cv, y_cv)

    # Score the holdout set, and compute holdout AUC only when both classes are
    # present so ROC AUC is well-defined.
    y_holdout_pred = final_pipeline.predict(X_holdout)
    holdout_auc = None
    if y_holdout.nunique() == 2:
        pos_idx = list(final_pipeline.classes_).index(1)
        y_holdout_proba = final_pipeline.predict_proba(X_holdout)[:, pos_idx]
        holdout_auc = plot_roc_curve_and_auc(
            y_holdout, y_holdout_proba, plot_roc_curve=plot_holdout_roc
        )

    # Convert fold metrics to a DataFrame and return both aggregate metrics and
    # the fitted final pipeline for downstream use.
    fold_df = pd.DataFrame(fold_metrics)

    ## consider dumping trained model to pickle
    return {
        "fold_metrics": fold_df,
        "cv_mean_auc": fold_df["auc"].mean(),
        "cv_mean_accuracy": fold_df["accuracy"].mean(),
        "holdout_accuracy": accuracy_score(y_holdout, y_holdout_pred),
        "holdout_auc": holdout_auc,
        "holdout_report": classification_report(y_holdout, y_holdout_pred),
        "final_pipeline": final_pipeline,
    }
