from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

from .model_utils import SpamModel

app = FastAPI(title="Spam Detection API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Startup: load & train once ----
MODEL = SpamModel()
STARTUP_METRICS = MODEL.train()

# ---- Schemas ----
class PredictIn(BaseModel):
    text: str = Field(..., min_length=1)

class BatchPredictIn(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=200)

# ---- Routes ----
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "MultinomialNB+TFIDF",
        "version": app.version,
        "metrics_at_start": STARTUP_METRICS
    }

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        return MODEL.predict_one(inp.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
def batch_predict(inp: BatchPredictIn):
    try:
        return {"results": MODEL.predict_batch(inp.texts)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    return MODEL.metrics()

@app.get("/charts-data")
def charts_data():
    return MODEL.charts_payload()

# ---- Advanced (HD) ----
@app.get("/pr-curve")
def pr_curve():
    return MODEL.pr_curve()

@app.get("/calibration")
def calibration():
    return MODEL.calibration()

@app.get("/kmeans/elbow")
def kmeans_elbow():
    return MODEL.kmeans_elbow()

@app.get("/kmeans/scores")
def kmeans_scores():
    return MODEL.kmeans_scores()
