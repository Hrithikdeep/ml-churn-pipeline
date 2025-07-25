from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("models/best_model.pkl")

# FastAPI app
app = FastAPI(title="Churn Prediction API")

# Input schema
class CustomerData(BaseModel):
    features: list  # Must match feature order used during training

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running 🎯"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]
        return {
            "prediction": int(pred),
            "churn_probability": round(prob, 4)
        }
    except Exception as e:
        return {"error": str(e)}
