from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import os

app = FastAPI()

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النماذج
models_path = os.path.join(os.path.dirname(__file__), '../models/')
print(f"Loading models from: {models_path}")

try:
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    # استخدام النموذج اللي بيتقبل 12 متغير (full model)
    model = joblib.load(os.path.join(models_path, 'imaging_model.pkl'))
    print("✅ Models loaded successfully (Full model with 12 features)")
    print(f"✅ Model expects: {model.n_features_in_} features")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    scaler = None
    model = None

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
    bnp: Optional[float] = 100
    xray_congestion: Optional[int] = 0
    xray_infiltrate: Optional[int] = 0

@app.get("/")
def home():
    return {"system": "RapidDx", "status": "running", "message": "RapidDx API is running!"}

@app.post("/diagnose")
def diagnose(patient: PatientData):
    try:
        if model is None or scaler is None:
            return {"error": "Models not loaded properly"}
        
        features = {
            'age': patient.age,
            'temperature': patient.temperature,
            'respiratory_rate': patient.respiratory_rate,
            'cough': patient.cough,
            'weight_gain': patient.weight_gain,
            'non_adherence': patient.non_adherence,
            'chest_pain': patient.chest_pain,
            'orthopnea': patient.orthopnea,
            'leg_swelling': patient.leg_swelling,
            'bnp': patient.bnp if patient.bnp else 100,
            'xray_congestion': patient.xray_congestion if patient.xray_congestion else 0,
            'xray_infiltrate': patient.xray_infiltrate if patient.xray_infiltrate else 0
        }
        
        # ترتيب المتغيرات بنفس ترتيب التدريب
        feature_names = ['age', 'temperature', 'respiratory_rate', 'cough', 
                        'weight_gain', 'non_adherence', 'chest_pain', 
                        'orthopnea', 'leg_swelling', 'bnp', 
                        'xray_congestion', 'xray_infiltrate']
        
        feature_values = np.array([[features[f] for f in feature_names]])
        feature_scaled = scaler.transform(feature_values)
        
        prob = model.predict_proba(feature_scaled)[0][1]
        
        if prob > 0.7:
            diagnosis = "فشل قلب متقدم"
            recommendation = "قياس BNP، إيكو، مدرات بول"
            action = "ADMIT TO CARDIOLOGY"
        elif prob > 0.4:
            diagnosis = "اشتباه فشل قلب - يحتاج فحوصات"
            recommendation = "متابعة، فحوصات معملية"
            action = "OBSERVATION"
        else:
            diagnosis = "أسباب أخرى لضيق التنفس"
            recommendation = "فحص أسباب بديلة"
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
    except Exception as e:
        return {"error": str(e)}
