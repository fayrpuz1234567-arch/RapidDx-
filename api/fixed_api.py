from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import os

app = FastAPI()

# Load models with correct scaler
models_path = os.path.join(os.path.dirname(__file__), '../models/')
scaler = joblib.load(os.path.join(models_path, 'new_scaler_9features.pkl'))
model = joblib.load(os.path.join(models_path, 'symptom_model.pkl'))

print("✅ Models loaded successfully")
print(f"✅ Scaler expects {scaler.n_features_in_} features")
print(f"✅ Model expects {model.n_features_in_} features")

class PatientData(BaseModel):
    age: float
    temperature: float
    respiratory_rate: float
    cough: int = 0
    weight_gain: int = 0
    non_adherence: int = 0
    chest_pain: int = 0
    orthopnea: int = 0
    leg_swelling: int = 0
    bnp: Optional[float] = None
    xray_congestion: Optional[int] = None
    xray_infiltrate: Optional[int] = None

# Order of features must match training
FEATURE_NAMES = ['age', 'temperature', 'respiratory_rate', 'cough', 
                 'weight_gain', 'non_adherence', 'chest_pain', 
                 'orthopnea', 'leg_swelling']

@app.get("/")
def home():
    return {"system": "RapidDx", "status": "running", "features": len(FEATURE_NAMES)}

@app.post("/diagnose")
def diagnose(patient: PatientData):
    # Build feature array in correct order
    features = np.array([[
        patient.age,
        patient.temperature,
        patient.respiratory_rate,
        patient.cough,
        patient.weight_gain,
        patient.non_adherence,
        patient.chest_pain,
        patient.orthopnea,
        patient.leg_swelling
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prob = model.predict_proba(features_scaled)[0][1]
    
    if prob > 0.7:
        diagnosis = "Heart Failure Exacerbation"
        recommendation = "Check BNP, Echocardiogram, Diuretics"
        action = "ADMIT TO CARDIOLOGY"
    elif prob > 0.4:
        diagnosis = "Suspected Heart Failure - Further tests needed"
        recommendation = "Follow up, lab tests"
        action = "OBSERVATION"
    else:
        diagnosis = "Other causes of dyspnea"
        recommendation = "Check alternative causes (PE, anxiety, anemia)"
        action = "FURTHER WORKUP"
    
    return {
        "hfe_probability": float(prob),
        "ari_probability": float(1 - prob),
        "primary_diagnosis": diagnosis,
        "recommendation": recommendation,
        "action_required": action,
        "red_flags": {"severity": "HIGH" if prob > 0.7 else "LOW", "indicators": []},
        "confidence": "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.6 else "LOW"
    }
