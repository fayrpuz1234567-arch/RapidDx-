import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="RapidDx - التحليل الإحصائي للمريض",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
    .formula-box { background: #e9ecef; padding: 1rem; border-radius: 10px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 RapidDx | التحليل الإحصائي التفصيلي للمريض</h1>
    <p>شرح خطوات التشخيص - Logistic Regression Model</p>
</div>
""", unsafe_allow_html=True)


# تحميل النماذج
@st.cache_resource
def load_models():
    models_path = os.path.join(os.path.dirname(__file__), '../models/')
    try:
        scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        model = joblib.load(os.path.join(models_path, 'imaging_model.pkl'))
        return scaler, model
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {e}")
        return None, None


scaler, model = load_models()

if scaler is None:
    st.error("❌ النماذج غير متوفرة. قم بتشغيل train_with_real_data.py أولاً")
    st.stop()

features = [
    'age', 'temperature', 'respiratory_rate', 'cough',
    'weight_gain', 'non_adherence', 'chest_pain',
    'orthopnea', 'leg_swelling', 'bnp',
    'xray_congestion', 'xray_infiltrate'
]

# إدخال بيانات المريض
st.header("📝 إدخال بيانات المريض")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("المعلومات الأساسية")
    age = st.number_input("العمر (سنوات)", min_value=18, max_value=120, value=72, step=1)
    temp = st.number_input("درجة الحرارة (مئوية)", min_value=35.0, max_value=42.0, value=37.2, step=0.1)
    rr = st.number_input("معدل التنفس (نفس/دقيقة)", min_value=8, max_value=50, value=24, step=1)

with col2:
    st.subheader("الأعراض")
    cough = 1 if st.radio("كحة", ["لا", "نعم"], index=0) == "نعم" else 0
    orthopnea = 1 if st.radio("ضيق تنفس بالاستلقاء", ["لا", "نعم"], index=0) == "نعم" else 0
    leg_swelling = 1 if st.radio("تورم الرجلين", ["لا", "نعم"], index=0) == "نعم" else 0

with col3:
    st.subheader("عوامل أخرى")
    weight_gain = 1 if st.radio("زيادة وزن حديثة", ["لا", "نعم"], index=0) == "نعم" else 0
    has_bnp = st.checkbox("متوفر BNP")
    bnp_val = st.number_input("BNP (pg/mL)", min_value=0, max_value=5000, value=650, step=10) if has_bnp else 100
    has_xray = st.checkbox("متوفرة صورة صدر")
    if has_xray:
        xray_congestion = 1 if st.radio("احتقان رئوي", ["لا", "نعم"], index=0) == "نعم" else 0
        xray_infiltrate = 1 if st.radio("ارتشاحات", ["لا", "نعم"], index=0) == "نعم" else 0
    else:
        xray_congestion = 0
        xray_infiltrate = 0

diagnose_btn = st.button("🔍 تحليل وتشخيص", type="primary", use_container_width=True)

if diagnose_btn:
    # قيم المتغيرات
    feature_values = [
        age, temp, rr, cough, weight_gain,
        0, 0, orthopnea, leg_swelling, bnp_val,
        xray_congestion, xray_infiltrate
    ]

    # ==================== الخطوة 1: البيانات المدخلة ====================
    st.header("📋 الخطوة 1: البيانات المدخلة")

    input_df = pd.DataFrame({
        'المتغير': features,
        'القيمة': feature_values,
        'الوصف': [
            'العمر بالسنوات',
            'درجة الحرارة (مئوية)',
            'معدل التنفس (نفس/دقيقة)',
            'كحة (1=نعم, 0=لا)',
            'زيادة وزن حديثة (1=نعم, 0=لا)',
            'عدم التزام بالعلاج (1=نعم, 0=لا)',
            'ألم في الصدر (1=نعم, 0=لا)',
            'ضيق تنفس بالاستلقاء (1=نعم, 0=لا)',
            'تورم الرجلين (1=نعم, 0=لا)',
            'مستوى BNP (pg/mL)',
            'احتقان رئوي بالأشعة (1=نعم, 0=لا)',
            'ارتشاحات بالأشعة (1=نعم, 0=لا)'
        ]
    })
    st.dataframe(input_df, use_container_width=True)

    st.header("📐 الخطوة 2: تطبيع البيانات (Standardization)")

    st.markdown("""
    **الصيغة الرياضية:**  
    `Z = (X - μ) / σ`

    حيث:
    - **X**: القيمة الأصلية
    - **μ**: متوسط القيم في بيانات التدريب
    - **σ**: الانحراف المعياري في بيانات التدريب
    """)

    X_input = np.array([feature_values])
    X_scaled = scaler.transform(X_input)

    scaling_df = pd.DataFrame({
        'المتغير': features,
        'القيمة الأصلية (X)': feature_values,
        'المتوسط (μ)': scaler.mean_,
        'الانحراف المعياري (σ)': scaler.scale_,
        'القيمة بعد التطبيع (Z)': X_scaled[0]
    })
    st.dataframe(scaling_df.style.format({
        'القيمة الأصلية (X)': '{:.2f}',
        'المتوسط (μ)': '{:.2f}',
        'الانحراف المعياري (σ)': '{:.2f}',
        'القيمة بعد التطبيع (Z)': '{:.3f}'
    }), use_container_width=True)

    # ==================== الخطوة 3: معاملات النموذج ====================
    st.header(" الخطوة 3: معاملات نموذج Logistic Regression")

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    coeff_df = pd.DataFrame({
        'المتغير': features,
        'المعامل (β)': coefficients,
        'التأثير': ['عكسي (يقلل الاحتمالية)' if c < 0 else 'طردي (يزيد الاحتمالية)' for c in coefficients],
        'الأهمية': ['عالية' if abs(c) > 0.5 else 'متوسطة' if abs(c) > 0.2 else 'منخفضة' for c in coefficients]
    })
    st.dataframe(coeff_df.style.format({'المعامل (β)': '{:.4f}'}), use_container_width=True)

    # ==================== الخطوة 4: المجموع الخطي ====================
    st.header("🧮 الخطوة 4: حساب المجموع الخطي (Linear Combination)")

    linear_combination = intercept + np.sum(coefficients * X_scaled[0])

    st.markdown("**الحساب:**")
    st.markdown(f"- `z = {intercept:.4f}`")
    for i, (coef, val) in enumerate(zip(coefficients, X_scaled[0])):
        st.markdown(f"- `+ ({coef:.4f} × {val:.4f}) = {coef * val:.4f}`")
    st.markdown(f"**النتيجة:** `z = {linear_combination:.4f}`")

    # ==================== الخطوة 5: دالة Sigmoid ====================
    st.header("📈 الخطوة 5: تطبيق دالة Sigmoid (Logistic Function)")

    st.markdown("""
    **الصيغة الرياضية:**
P(Heart Failure = 1) = 1 / (1 + e⁻ᶻ)
حيث:
- **z**: المجموع الخطي المحسوب في الخطوة السابقة
- **P**: احتمالية وجود فشل القلب (قيمة بين 0 و 1)
""")

probability = 1 / (1 + np.exp(-linear_combination))

st.markdown(f"""
**الحساب:**
z = {linear_combination:.4f}
e⁻ᶻ = e⁻{linear_combination:.4f} = {np.exp(-linear_combination):.6f}
P = 1 / (1 + {np.exp(-linear_combination):.6f})
P = {probability:.4f}""")

# عرض الاحتمالية بشكل بارز
col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
with col_p2:
    st.markdown(f"""
    <div style="text-align: center; background: #2a5298; padding: 1rem; border-radius: 10px;">
        <h2 style="color: white;">🎯 احتمالية فشل القلب</h2>
        <p style="color: white; font-size: 3rem; font-weight: bold; margin: 0;">{probability*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
st.progress(probability)

# ==================== الخطوة 6: القرار النهائي ====================
st.header("✅ الخطوة 6: القرار النهائي")

threshold = 0.5
st.markdown(f"**عتبة القرار (Threshold) = {threshold}** (أي إذا كانت الاحتمالية > {threshold*100:.0f}% يتم تشخيص فشل القلب)")

if probability > threshold:
    st.markdown(f"""
    <div style="background: #ff6b6b; padding: 1.5rem; border-radius: 10px; color: white;">
        <h2 style="color: white;">🚨 التشخيص: فشل قلب متقدم</h2>
        <p>📊 احتمالية الإصابة: <strong>{probability*100:.1f}%</strong></p>
        <p> المقارنة: {probability*100:.1f}% > {threshold*100:.0f}%</p>
        <p style="margin-top: 1rem;">💊 <strong>التوصية الطبية:</strong><br>
        - قياس BNP<br>
        - إيكو على القلب (Echocardiography)<br>
        - بدء مدرات البول (Diuretics)<br>
        - متابعة وظائف الكلى والكهارل</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="background: #51cf66; padding: 1.5rem; border-radius: 10px; color: white;">
        <h2 style="color: white;">✅ التشخيص: احتمالية منخفضة لفشل القلب</h2>
        <p>📊 احتمالية الإصابة: <strong>{probability*100:.1f}%</strong></p>
        <p> المقارنة: {probability*100:.1f}% < {threshold*100:.0f}%</p>
        <p style="margin-top: 1rem;">💊 <strong>التوصية الطبية:</strong><br>
        - فحص أسباب أخرى لضيق التنفس<br>
        - قد يكون: مرض تنفسي، انصمام رئوي، قلق، فقر دم<br>
        - إجراء الفحوصات المناسبة حسب الحالة</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== تفسير المعاملات ====================
st.header("🔍 تفسير المعاملات (Feature Importance)")

st.markdown("""
**كيف تقرأ المعاملات في الانحدار اللوجستي:**

- **المعامل الموجب (+):** زيادة في قيمة هذا المتغير تزيد من احتمالية فشل القلب
- **المعامل السالب (-):** زيادة في قيمة هذا المتغير تقلل من احتمالية فشل القلب
- **القيمة المطلقة الكبيرة:** هذا المتغير له تأثير قوي على التشخيص
""")

importance_df = pd.DataFrame({
    'المتغير': features,
    'المعامل (β)': coefficients,
    'التأثير على الاحتمالية': ['⬆️ يزيد بقوة' if c > 0.5 else '⬆️ يزيد' if c > 0 else '⬇️ يقلل' if c < 0 else 'محايد' for c in coefficients],
    'قوة التأثير': ['مرتفع 🔴' if abs(c) > 0.5 else 'متوسط 🟡' if abs(c) > 0.2 else 'منخفض 🟢' for c in coefficients]
}).sort_values('المعامل (β)', key=abs, ascending=False)

st.dataframe(importance_df, use_container_width=True)

# نموذج المعادلة النهائية
st.header("📝 ملخص المعادلة النهائية")

equation = f"logit(p) = {intercept:.4f}"
for i, coef in enumerate(coefficients):
    if coef > 0:
        equation += f" + {coef:.4f}×{features[i]}"
    elif coef < 0:
        equation += f" - {abs(coef):.4f}×{features[i]}"

st.markdown(f"""
<div class="formula-box">
    <code>P(Failure) = 1 / (1 + e⁻ᶻ)</code><br>
    <code>z = {equation}</code>
</div>
""", unsafe_allow_html=True)

# تحذير
st.info("""
""")

st.markdown("---")
st.caption("📊 RapidDx - نظام دعم القرار التشخيصي | Logistic Regression Analysis")