import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import PredictRequest, PredictResponse

_logger = logging.getLogger("app.predictions")


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "predictions.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def log_prediction(request: PredictRequest, response: PredictResponse, latency_ms: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "num_ipos": len(request.ipos),
        "input": [ipo.model_dump() for ipo in request.ipos],
        "output": [alloc.model_dump() for alloc in response.allocations],
    }
    _logger.info(json.dumps(entry))
