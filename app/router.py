import time

from fastapi import APIRouter, Request

from app.logger import log_prediction
from app.schemas import IpoAllocation, PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    start = time.perf_counter()

    service = request.app.state.inference_service
    ipos = [ipo.model_dump() for ipo in body.ipos]
    probabilities, weights = service.predict(ipos)

    allocations = [
        IpoAllocation(
            probability=round(float(p), 6),
            allocation_weight=round(float(w), 6),
        )
        for p, w in zip(probabilities, weights)
    ]
    response = PredictResponse(allocations=allocations)

    latency_ms = (time.perf_counter() - start) * 1000
    log_prediction(body, response, latency_ms)

    return response
