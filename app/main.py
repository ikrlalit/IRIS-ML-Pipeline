from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import joblib
import os
import logging
import time
import numpy as np

# Structured Logging
from pythonjsonlogger import jsonlogger

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Tracing Setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Logging Setup
logger = logging.getLogger("iris-log-ml-service")
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = jsonlogger.JsonFormatter()
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.joblib")
model = joblib.load(MODEL_PATH)

# FastAPI App
app = FastAPI()

# App state for readiness/liveness
app_state = {"is_ready": False, "is_alive": True}

# Input schema
class IrisInput(BaseModel):
    data: List[float]  # should be 4 floats

@app.on_event("startup")
async def startup_event():
    time.sleep(2)  # Simulate model load
    app_state["is_ready"] = True

@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.get("/live_check", tags=["Probe"])
def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    logger.exception("Unhandled exception", extra={
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    })

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict", tags=["Inference"])
def predict(input: IrisInput):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            features = np.array(input.data).reshape(1, -1)
            prediction = model.predict(features)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info("Prediction successful", extra={
                "event": "prediction",
                "trace_id": trace_id,
                "input": input.dict(),
                "result": {"class": prediction[0]},
                "latency_ms": latency,
                "status": "success"
            })

            return {"class": prediction[0]}

        except Exception as e:
            logger.exception("Prediction failed", extra={
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail="Prediction failed")
