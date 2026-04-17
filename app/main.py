import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.inference import InferencePipeline
from app.logger import setup_logging
from app.router import router
from src.utils.utils import get_project_root

_ROOT = get_project_root()
_MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_ROOT / "models" / "prediction_pipeline.joblib")))
_LOG_DIR = Path(os.environ.get("LOG_DIR", str(_ROOT / "logs")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(_LOG_DIR)
    app.state.inference_pipeline = InferencePipeline(_MODEL_PATH)
    yield


app = FastAPI(title="IPO Listing Gain Prediction", lifespan=lifespan)
app.include_router(router)
