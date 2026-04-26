import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("🏥 RapidDx - Training with REAL Hospital Data")
print("=" * 60)

# ============================================
# 1. تحميل البيانات الحقيقية من المستشفى
# ============================================

data_path = os.path.join(os.path.dirname(__file__), '../data/patients_data.csv')

if os.path.exists(data_path):
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} patient records from hospital")
else:
    print(f"❌ File not found: {data_path}")
    print("Please place your CSV file at: data/patients_data.csv")
    exit(1)

# عرض أول 5 صفوف للتأكد
print("\n📋 First 5 rows of data:")
print(df.head())

# ============================================
# 2. التحقق من البيانات
# ============================================

print(f"\n📊 Data Info:")
print(df.info())

print(f"\n📈 Diagnosis distribution:")
print(f"   HFE (Heart Failure): {df['hfe'].sum()} ({df['hfe'].mean() * 100:.1f}%)")
print(f"   ARI (Respiratory): {df['ari'].sum()} ({df['ari'].mean() * 100:.1f}%)")

# ============================================
# 3. اختيار المتغيرات (Features)
# ============================================

# المستوى 1: الأعراض فقط (9 متغيرات)
symptom_features = ['age', 'temperature', 'respiratory_rate', 'cough',
                    'weight_gain', 'non_adherence', 'chest_pain',
                    'orthopnea', 'leg_swelling']

# المستوى 2: الأعراض + BNP (10 متغيرات)
bnp_features = symptom_features + ['bnp']

# المستوى 3: شامل + أشعة (12 متغيرات)
full_features = bnp_features + ['xray_congestion', 'xray_infiltrate']

X_symptom = df[symptom_features]
X_bnp = df[bnp_features]
X_full = df[full_features]

y_hfe = df['hfe']
y_ari = df['ari']

# ============================================
# 4. تطبيع البيانات
# ============================================

scaler = StandardScaler()
X_symptom_scaled = scaler.fit_transform(X_symptom)
X_bnp_scaled = scaler.fit_transform(X_bnp)
X_full_scaled = scaler.fit_transform(X_full)


# ============================================
# 5. تدريب وتقييم النماذج
# ============================================

def train_and_evaluate(X, y, model_name, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 {target_name} - {model_name}:")
    print(f"   AUC = {auc:.3f}")
    print(f"   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[f'No {target_name}', target_name]))

    return model, auc


print("\n" + "=" * 60)
print("🫀 Training HEART FAILURE (HFE) Models")
print("=" * 60)

model_hfe_symptom, auc1 = train_and_evaluate(X_symptom_scaled, y_hfe, "Symptoms Only", "HFE")
model_hfe_bnp, auc2 = train_and_evaluate(X_bnp_scaled, y_hfe, "Symptoms + BNP", "HFE")
model_hfe_full, auc3 = train_and_evaluate(X_full_scaled, y_hfe, "Full + X-ray", "HFE")

print("\n" + "=" * 60)
print("🫁 Training RESPIRATORY (ARI) Models")
print("=" * 60)

model_ari_symptom, auc4 = train_and_evaluate(X_symptom_scaled, y_ari, "Symptoms Only", "ARI")
model_ari_bnp, auc5 = train_and_evaluate(X_bnp_scaled, y_ari, "Symptoms + BNP", "ARI")
model_ari_full, auc6 = train_and_evaluate(X_full_scaled, y_ari, "Full + X-ray", "ARI")

# ============================================
# 6. حفظ النماذج
# ============================================

print("\n💾 Saving models...")
joblib.dump(model_hfe_symptom, 'symptom_model.pkl')
joblib.dump(model_hfe_bnp, 'bnp_model.pkl')
joblib.dump(model_hfe_full, 'imaging_model.pkl')
joblib.dump(model_ari_symptom, 'ari_symptom_model.pkl')
joblib.dump(model_ari_bnp, 'ari_bnp_model.pkl')
joblib.dump(model_ari_full, 'ari_imaging_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ All models saved successfully!")
print("=" * 60)

# ============================================
# 7. عرض ملخص النتائج
# ============================================

print("\n📋 MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'Model':<20} {'HFE AUC':<12} {'ARI AUC':<12}")
print("-" * 44)
print(f"{'Symptoms Only':<20} {auc1:<12.3f} {auc4:<12.3f}")
print(f"{'Symptoms + BNP':<20} {auc2:<12.3f} {auc5:<12.3f}")
print(f"{'Full + X-ray':<20} {auc3:<12.3f} {auc6:<12.3f}")
print("=" * 60)