# api/app.py
"""
FastAPI service for Titanic survival prediction.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/05_RandomForest_PCA-False_Tuning-False.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="FastAPI service for predicting Titanic passenger survival",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    return m

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Prediction request with passenger information."""
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "pclass": 3,
                        "sex": "male",
                        "age": 22.0,
                        "siblings_spouses": 1,
                        "parents_children": 0,
                        "fare": 7.25,
                        "port_code": "S"
                    }
                ]
            }
        }

class PredictResponse(BaseModel):
    predictions: List[str]
    probabilities: List[float]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": ["Died", "Survived"],
                "probabilities": [0.23, 0.87],
                "count": 2
            }
        }

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "model": "RandomForest",
        "f1_score": 0.7385,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
        "model_type": "RandomForest Classifier"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided."
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format: {e}"
        )

    required_columns = ["pclass", "sex", "age", "siblings_spouses", 
                       "parents_children", "fare", "port_code"]
    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}"
        )

    try:
        # Encode sex
        le_sex = LabelEncoder()
        le_sex.fit(['male', 'female'])
        X['sex_encoded'] = le_sex.transform(X['sex'])
        
        # Encode port
        le_port = LabelEncoder()
        le_port.fit(['C', 'Q', 'S'])
        X['port_encoded'] = le_port.transform(X['port_code'])
        
        # Select features in correct order
        feature_columns = ['age', 'pclass', 'fare', 'siblings_spouses', 
                          'parents_children', 'sex_encoded', 'port_encoded']
        X_features = X[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )

    # Convert to readable labels
    pred_labels = ["Survived" if p == 1 else "Died" for p in preds]
    prob_list = [float(p) for p in probs]

    return PredictResponse(
        predictions=pred_labels,
        probabilities=prob_list,
        count=len(pred_labels)
    )

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Titanic Survival Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model: RandomForest (Best F1-Score: 0.7385)")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")
