"""Tree-based model pipelines"""
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_random_forest_pipeline(**model_kwargs) -> Pipeline:
    return Pipeline([
        ("model", RandomForestClassifier(**model_kwargs)),
    ])

def get_xgboost_pipeline(**model_kwargs) -> Pipeline:
    return Pipeline([
        ("model", XGBClassifier(**model_kwargs)),
    ])

def get_lightgbm_pipeline(**model_kwargs) -> Pipeline:
    return Pipeline([
        ("model", LGBMClassifier(**model_kwargs)),
    ])