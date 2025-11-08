from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

from model_utils import SpamModel

app = FastAPI(title="Spam Detection API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

#startup: try to load model or train new one
MODEL = SpamModel()
if MODEL.load_model():
    print("Loaded existing model from spam.pkl")
    STARTUP_METRICS = MODEL.metrics()
else:
    print("Training new model...")
    STARTUP_METRICS = MODEL.train()

PRED_HISTORY = []

#schemas
class PredictIn(BaseModel):
    text: str = Field(..., min_length=1)

class BatchPredictIn(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=200)

#routes
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
        result = MODEL.predict_one(inp.text)
        PRED_HISTORY.append(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
def batch_predict(inp: BatchPredictIn):
    try:
        results = MODEL.predict_batch(inp.texts)
        PRED_HISTORY.extend(results)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    return MODEL.metrics()

@app.get("/charts-data")
def charts_data():
    return MODEL.charts_payload()

@app.get("/pr-curve")
def pr_curve():
    return MODEL.pr_curve()

@app.get("/calibration")
def calibration():
    return MODEL.calibration()

# /kmeans/elbow endpoint removed (feature deprecated)

@app.get("/kmeans/scores")
def kmeans_scores():
    return MODEL.kmeans_scores()

@app.get("/prediction-stats")
def prediction_stats():
    """Return counts of spam and ham predictions from session history."""
    spam_count = sum(1 for p in PRED_HISTORY if p.get("label") == 1)
    ham_count = sum(1 for p in PRED_HISTORY if p.get("label") == 0)
    return {
        "spam": spam_count,
        "ham": ham_count,
        "total": len(PRED_HISTORY)
    }

@app.post("/reset-predictions")
def reset_predictions():
    """Reset the prediction history."""
    global PRED_HISTORY
    PRED_HISTORY.clear()
    return {"message": "Prediction history reset successfully", "total": 0}