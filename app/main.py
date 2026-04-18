from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI

from app.inference import InferenceService
from app.logger import setup_logging
from app.router import router
from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.utils import get_project_root, load_config

_ROOT = get_project_root()
_CFG = load_config()
_MODEL_PATH = _ROOT / _CFG["paths"]["model"]
_LOG_DIR = _ROOT / _CFG["paths"]["logs"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(_LOG_DIR)
    trained_pipeline = joblib.load(_MODEL_PATH)
    portfolio_cfg = _CFG["portfolio"]
    inference_pipeline = InferencePipeline(
        fitted_prediction_pipeline=trained_pipeline,
        t_min=portfolio_cfg["trade_threshold"],
        alpha=portfolio_cfg["alpha"],
    )
    app.state.inference_service = InferenceService(inference_pipeline)
    yield


app = FastAPI(title="IPO Listing Gain Prediction", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
