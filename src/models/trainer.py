import pandas as pd
from sklearn.pipeline import Pipeline
from src.pipelines import data_pipeline, prediction_pipeline


def train(X: pd.DataFrame, model_type: str, listing_gain_threshold_perc: float = 5.0, **model_kwargs) -> Pipeline:
    """
    Train a prediction pipeline on the provided dataset.

    The target is computed as a binary label:
    1 if listing gain percentage is greater than listing_gain_threshold_perc,
    otherwise 0.

    Args:
        X: Input dataframe with raw columns required by the data pipeline.
        model_type: Pipeline key supported by prediction_pipeline.get_prediction_pipeline.
        listing_gain_threshold_perc: Percent threshold used to derive the target.

    Returns:
        A fitted sklearn Pipeline.
    """
    y = data_pipeline.compute_target(X, listing_gain_threshold=listing_gain_threshold_perc)
    pipeline = prediction_pipeline.get_prediction_pipeline(model_type, **model_kwargs)
    
    # print(f'Training model: {model_type} with threshold {listing_gain_threshold_perc}%')
    
    pipeline.fit(X, y)
    return pipeline