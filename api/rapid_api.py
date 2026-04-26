from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clinical_rules.decision_rules import ClinicalDecisionRules

app = FastAPI(title="RapidDx - Emergency Dyspnea Diagnostic System")

models_path = os.path.join(os.path.dirname(__file__), '../models/')
scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
model_hfe_symptom = joblib.load(os.path.join(models_path, 'symptom_model.pkl'))
model_hfe_bnp = joblib.load(os.path.join(models_path, 'bnp_model.pkl'))
model_hfe_full = joblib.load(os.path.join(models_path, 'imaging_model.pkl'))
model_ari_symptom = joblib.load(os.path.join(models_path, 'ari_symptom_model.pkl'))
model_ari_bnp = joblib.load(os.path.join(models_path, 'ari_bnp_model.pkl'))
model_ari_full = joblib.load(os.path.join(models_path, 'ari_imaging_model.pkl'))

rules = ClinicalDecisionRules()

class PatientData(BaseModel):
    age: float = Field(..., ge=18, le=120)
    temperature: float = Field(..., ge=35, le=42)
    respiratory_rate: float = Field(..., ge=8, le=50)
    cough: int = Field(0, ge=0, le=1)
    weight_gain: int = Field(0, ge=0, le=1)
    non_adherence: int = Field(0, ge=0, le=1)
    chest_pain: int = Field(0, ge=0, le=1)
    orthopnea: int = Field(0, ge=0, le=1)
    leg_swelling: int = Field(0, ge=0, le=1)
    bnp: Optional[float] = None
    xray_congestion: Optional[int] = None
    xray_infiltrate: Optional[int] = None

@app.post("/diagnose")
def diagnose(patient: PatientData):
    try:
        has_bnp = patient.bnp is not None
        has_xray = patient.xray_congestion is not None or patient.xray_infiltrate is not None
        
        if has_xray and has_bnp:
            hfe_model = model_hfe_full
            ari_model = model_ari_full
            level_desc = "تشخيص كامل (أعراض + BNP + أشعة)"
        elif has_bnp:
            hfe_model = model_hfe_bnp
            ari_model = model_ari_bnp
            level_desc = "تشخيص سريع (أعراض + BNP)"
        else:
            hfe_model = model_hfe_symptom
            ari_model = model_ari_symptom
            level_desc = "فرز أولي (أعراض فقط)"
        
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
        
        feature_names = list(features.keys())
        feature_values = np.array([[features[f] for f in feature_names]])
        feature_scaled = scaler.transform(feature_values)
        
        hfe_prob = hfe_model.predict_proba(feature_scaled)[0][1]
        ari_prob = ari_model.predict_proba(feature_scaled)[0][1]
        
        if patient.bnp:
            _, _, bnp_level = rules.bnp_interpretation(patient.bnp)
        else:
            bnp_level = "unknown"
        
        hfe_score_level, hfe_score_text, _ = rules.heart_failure_score(
            orthopnea=patient.orthopnea,
            leg_swelling=patient.leg_swelling,
            weight_gain=patient.weight_gain,
            bnp_level=bnp_level
        )
        
        severity, red_flags = rules.emergency_indicators(
            age=patient.age,
            rr=patient.respiratory_rate,
            spo2=None,
            bnp=patient.bnp,
            troponin=None,
            chest_pain=patient.chest_pain
        )
        
        if severity == "CRITICAL":
            diagnosis = "حالة حرجة - تدخل فوري مطلوب"
            recommendation = "نقل للعناية المركزة، أكسجين، فحوصات عاجلة"
            action = "IMMEDIATE ADMISSION"
        elif hfe_prob > 0.7:
            diagnosis = "فشل قلب متقدم"
            recommendation = "قياس BNP، إيكو، مدرات بول"
            action = "ADMIT TO CARDIOLOGY"
        elif ari_prob > 0.7:
            diagnosis = "مرض تنفسي حاد"
            recommendation = "صورة صدر، مضاد حيوي، مراقبة تنفس"
            action = "ADMIT TO RESPIRATORY"
        elif hfe_prob > 0.4 or ari_prob > 0.4:
            diagnosis = "غير مؤكد - يحتاج فحوصات إضافية"
            recommendation = "متابعة، فحوصات معملية"
            action = "OBSERVATION"
        else:
            diagnosis = "أسباب أخرى لضيق التنفس"
            recommendation = "فحص أسباب بديلة"
            action = "FURTHER WORKUP"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "diagnosis_level": level_desc,
            "hfe_probability": round(hfe_prob, 3),
            "ari_probability": round(ari_prob, 3),
            "primary_diagnosis": diagnosis,
            "recommendation": recommendation,
            "action_required": action,
            "clinical_rules": {
                "heart_failure_score": hfe_score_text,
                "bnp_status": bnp_level
            },
            "red_flags": {
                "severity": severity,
                "indicators": red_flags
            },
            "confidence": "HIGH" if max(hfe_prob, ari_prob) > 0.8 else "MEDIUM" if max(hfe_prob, ari_prob) > 0.6 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"system": "RapidDx", "version": "2.0", "status": "running"}
