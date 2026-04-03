"""
Orchestrator: load data → split → train → evaluate.
All pipeline assembly is handled by pipeline_factory.
"""
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.pipelines.data_pipeline import compute_target
from src.pipelines.pipeline_factory import get_pipeline
from src.utils.utils import load_config, load_raw_dataset


def train(model_type: str, test_size: float = 0.2, random_state: int = 42):
    df, cfg = load_raw_dataset(), load_config()
    threshold = cfg["target"]["listing_gain_threshold_perc"]

    y = compute_target(df, listing_gain_threshold=threshold)

    # need custom stratified logic for train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    print(f'Training model: {model_type}...')

    pipeline = get_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    print(f'Training completed for {model_type}. Proceeding to metrics evaluation')

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    print(f"[{model_type}] Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["report"])

    return pipeline, metrics


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    train(model_name)