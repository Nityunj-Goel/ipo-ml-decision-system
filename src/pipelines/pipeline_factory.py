"""
Factory: assemble a full pipeline (data pipeline + model-specific pipeline)
for a given model type.
"""
from sklearn.pipeline import Pipeline

from src.pipelines.data_pipeline import get_data_pipeline
from src.pipelines.model_pipelines.logistic_regression import get_logistic_regression_pipeline
from src.pipelines.model_pipelines.tree_pipeline import get_tree_pipeline
from configs.feature_config import CleaningMode

_MODEL_PIPELINES = {
    "logistic_regression": get_logistic_regression_pipeline,
    "tree": get_tree_pipeline,
}


def get_pipeline(model_type: str, mode: CleaningMode = "strict") -> Pipeline:
    data_pipeline = get_data_pipeline(mode=mode)
    model_pipeline = _MODEL_PIPELINES[model_type]()

    return Pipeline([
        ("data", data_pipeline),
        ("model", model_pipeline),
    ])
