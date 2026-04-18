"""
Factory: assemble a full pipeline (data pipeline + model-specific pipeline)
for a given model type.
"""
from sklearn.pipeline import Pipeline

from src.pipelines.data_pipeline import get_data_pipeline
from src.pipelines.model_pipelines.logistic_regression import get_logistic_regression_pipeline
from src.pipelines.model_pipelines.trees import get_random_forest_pipeline, get_xgboost_pipeline, get_lightgbm_pipeline
from configs.feature_config import CleaningMode

_MODEL_PIPELINES = {
    "logistic_regression": get_logistic_regression_pipeline,
    "random_forest": get_random_forest_pipeline,
    "xgboost": get_xgboost_pipeline,
    "lightgbm": get_lightgbm_pipeline,
}


def get_prediction_pipeline(model_type: str, training_mode: CleaningMode = "strict", **model_kwargs) -> Pipeline:
    data_pipeline = get_data_pipeline(mode=training_mode)
    model_pipeline = _MODEL_PIPELINES[model_type](**model_kwargs)

    return Pipeline([
        ("data", data_pipeline),
        ("model", model_pipeline),
    ])
