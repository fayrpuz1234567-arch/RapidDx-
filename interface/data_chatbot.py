
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="RapidDx - الشات الذكي للبيانات",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
    .chat-message-user { background: #2a5298; color: white; padding: 0.8rem; border-radius: 15px; margin: 0.5rem 0; text-align: right; }
    .chat-message-bot { background: #e9ecef; color: #333; padding: 0.8rem; border-radius: 15px; margin: 0.5rem 0; }
    .suggestions { background: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖 RapidDx | الشات الذكي للبيانات</h1>
    <p>اسأل عن بيانات المستشفى، النتائج الإحصائية، وتفاصيل المرضى - بالعربية</p>
</div>
""", unsafe_allow_html=True)


# تحميل البيانات والنماذج
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/patients_data.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if 'ari' in df.columns:
            df['ari'] = df['ari'].fillna(0).astype(int)
        return df
    return None


@st.cache_resource
def load_models():
    models_path = os.path.join(os.path.dirname(__file__), '../models/')
    try:
        scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        model = joblib.load(os.path.join(models_path, 'imaging_model.pkl'))
        return scaler, model
    except:
        return None, None


df = load_data()
scaler, model = load_models()

if df is None:
    st.error("❌ لا توجد بيانات. قم بتشغيل train_with_real_data.py أولاً")
    st.stop()


# دوال معالجة الأسئلة
def get_data_summary():
    """ملخص عام للبيانات"""
    total = len(df)
    hfe_count = df['hfe'].sum()
    ari_count = df['ari'].sum()
    both_count = ((df['hfe'] == 1) & (df['ari'] == 1)).sum()

    return f"""
    📊 **ملخص البيانات:**
    - عدد المرضى الإجمالي: {total} مريض
    - حالات فشل القلب: {hfe_count} حالة ({hfe_count / total * 100:.1f}%)
    - حالات الأمراض التنفسية: {ari_count} حالة ({ari_count / total * 100:.1f}%)
    - حالات الاتنين معاً: {both_count} حالة ({both_count / total * 100:.1f}%)
    - الحالات السليمة: {total - (hfe_count + ari_count - both_count)} حالة
    """


def get_patient_info(patient_id):
    """معلومات مريض محدد"""
    if patient_id < 0 or patient_id >= len(df):
        return f"❌ لا يوجد مريض بالرقم {patient_id}. عدد المرضى المتاحين: {len(df)}"

    patient = df.iloc[patient_id]
    return f"""
    👤 **المريض رقم {patient_id}:**
    - العمر: {patient['age']:.0f} سنة
    - درجة الحرارة: {patient['temperature']:.1f}°C
    - معدل التنفس: {patient['respiratory_rate']:.0f} نفس/دقيقة
    - كحة: {'نعم' if patient['cough'] == 1 else 'لا'}
    - تورم الرجلين: {'نعم' if patient['leg_swelling'] == 1 else 'لا'}
    - BNP: {patient['bnp']:.0f} pg/mL
    - التشخيص النهائي: {'فشل قلب' if patient['hfe'] == 1 else 'لا يوجد فشل قلب'}
    """


def get_statistics(variable):
    """إحصائيات متغير معين"""
    var = variable.lower().strip()

    stats_map = {
        'العمر': ('age', 'سنة'),
        'age': ('age', 'سنة'),
        'الحرارة': ('temperature', 'درجة مئوية'),
        'temperature': ('temperature', 'درجة مئوية'),
        'معدل التنفس': ('respiratory_rate', 'نفس/دقيقة'),
        'respiratory_rate': ('respiratory_rate', 'نفس/دقيقة'),
        'bnp': ('bnp', 'pg/mL')
    }

    for key, (col, unit) in stats_map.items():
        if key in var:
            data = df[col].dropna()
            return f"""
            📈 **إحصائيات {key}:**
            - المتوسط: {data.mean():.1f} {unit}
            - الوسيط: {data.median():.1f} {unit}
            - أقل قيمة: {data.min():.1f} {unit}
            - أعلى قيمة: {data.max():.1f} {unit}
            - الانحراف المعياري: {data.std():.2f}
            """

    return f"""
    ❓ لم أفهم المتغير "{variable}"

    المتغيرات المتاحة للسؤال عنها:
    - العمر
    - الحرارة
    - معدل التنفس
    - BNP (نسبة BNP في الدم)
    - hfe (حالات فشل القلب)
    - ari (حالات الأمراض التنفسية)
    """


def get_correlation(var1, var2):
    """العلاقة بين متغيرين"""
    col_map = {
        'العمر': 'age', 'السن': 'age', 'age': 'age',
        'الحرارة': 'temperature', 'temperature': 'temperature', 'حرارة': 'temperature',
        'التنفس': 'respiratory_rate', 'معدل التنفس': 'respiratory_rate',
        'bnp': 'bnp', 'بي إن بي': 'bnp'
    }

    col1 = None
    col2 = None

    for key, col in col_map.items():
        if key in var1.lower():
            col1 = col
        if key in var2.lower():
            col2 = col

    if col1 and col2 and col1 in df.columns and col2 in df.columns:
        corr = df[col1].corr(df[col2])
        strength = "قوية جداً" if abs(corr) > 0.7 else "متوسطة" if abs(corr) > 0.4 else "ضعيفة"
        direction = "طردية" if corr > 0 else "عكسية"

        return f"""
        📊 **العلاقة بين {var1} و {var2}:**
        - معامل الارتباط: {corr:.3f}
        - قوة العلاقة: {strength}
        - اتجاه العلاقة: {direction}

        **تفسير:**
        - عندما يزيد {var1}، {'يزيد' if corr > 0 else 'يقل'} {var2}
        - نسبة التفسير: {abs(corr) * 100:.1f}%
        """

    return f"❓ لم أتمكن من حساب العلاقة بين '{var1}' و '{var2}'"


def get_model_performance():
    """أداء النموذج"""
    if model is None:
        return "❌ النموذج غير متوفر حالياً"

    from sklearn.metrics import roc_auc_score, accuracy_score

    features = ['age', 'temperature', 'respiratory_rate', 'cough',
                'weight_gain', 'non_adherence', 'chest_pain',
                'orthopnea', 'leg_swelling', 'bnp',
                'xray_congestion', 'xray_infiltrate']

    X = df[features]
    y = df['hfe']

    from sklearn.preprocessing import StandardScaler
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X)

    # تقييم سريع
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_scaled, y, cv=min(3, len(df)), scoring='roc_auc')

    return f"""
    🎯 **أداء نموذج Logistic Regression:**
    - متوسط AUC: {scores.mean():.3f}
    - انحراف معياري: {scores.std():.3f}

    **تفسير:**
    {'✅ أداء ممتاز' if scores.mean() > 0.8 else '🟡 أداء مقبول' if scores.mean() > 0.6 else '🔴 يحتاج تحسين'}

    💡 **ملاحظة:** الدقة تعتمد على كمية البيانات. كلما زادت البيانات، تحسن الأداء.
    """


def process_question(question):
    """معالجة السؤال وإرجاع الرد"""
    q = question.lower().strip()

    # أسئلة عن ملخص البيانات
    if any(word in q for word in ['ملخص', 'إحصائيات', 'كم مريض', 'عدد المرضى', 'كام مريض', 'حالات']):
        return get_data_summary()

    # أسئلة عن مريض محدد
    if 'مريض' in q or 'patient' in q:
        import re
        numbers = re.findall(r'\d+', q)
        if numbers:
            return get_patient_info(int(numbers[0]))
        return f"🧑‍⚕️ عدد المرضى في قاعدة البيانات: {len(df)} مريض\n\nللاستعلام عن مريض محدد، اكتب مثلاً: 'مريض 1' أو 'عرض المريض رقم 2'"

    # أسئلة عن العلاقة بين متغيرين
    if any(word in q for word in ['العلاقة', 'علاقة', 'ارتباط', 'بين']) and 'و' in q:
        parts = q.split('و')
        var1 = parts[0].replace('العلاقة', '').replace('بين', '').replace('ما', '').strip()
        var2 = parts[1].split('؟')[0].split(' ')[0].strip()
        if var1 and var2:
            return get_correlation(var1, var2)

    # أسئلة عن أداء النموذج
    if any(word in q for word in ['أداء', 'دقة', 'performance', 'accuracy', 'النموذج', 'AUC']):
        return get_model_performance()

    # أسئلة عن متغير معين
    for var in ['العمر', 'age', 'الحرارة', 'temperature', 'حرارة', 'التنفس', 'معدل التنفس', 'bnp', 'بي إن بي']:
        if var in q:
            return get_statistics(var)

    # مساعدة عامة
    return f"""
    🤔 **لم أفهم سؤالك. إليك بعض الأسئلة التي يمكنك طرحها:**

    **أسئلة عن البيانات:**
    - "ملخص البيانات" - عرض إحصائيات عامة
    - "كم مريض" - عدد المرضى في قاعدة البيانات
    - "مريض 0" - عرض بيانات مريض محدد (0-{len(df) - 1})
    - "العمر" - إحصائيات الأعمار
    - "BNP" - إحصائيات BNP

    **أسئلة عن العلاقات:**
    - "العلاقة بين العمر و BNP" - معامل الارتباط

    **أسئلة عن النموذج:**
    - "أداء النموذج" - تقييم دقة التشخيص
    - "دقة النموذج" - نسبة صحة التنبؤات

    **للاستفسار عن مريض محدد كتابة رقمه مثل "مريض 1"**
    """


# ==================== واجهة الشات ====================

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 إحصائيات سريعة")
    st.metric("عدد المرضى", len(df))
    st.metric("حالات فشل القلب", f"{df['hfe'].sum()} ({df['hfe'].mean() * 100:.0f}%)")
    st.metric("حالات مرض تنفسي", f"{df['ari'].sum()} ({df['ari'].mean() * 100:.0f}%)")

    if model:
        st.subheader("🎯 أداء النموذج")
        st.info("اسأل عن 'أداء النموذج' لتفاصيل أكثر")

    st.subheader("💡 أمثلة")
    st.markdown("""
    - `ملخص البيانات`
    - `كم مريض`
    - `مريض 0`
    - `العمر`
    - `مريض 3`
    - `أداء النموذج`
    - `العلاقة بين العمر و BNP`
    """)

with col1:
    st.subheader("💬 اسأل عن بيانات المستشفى")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": f"👋 مرحباً! أنا مساعدك الذكي للبيانات.\n\nلدي بيانات {len(df)} مريض. اسألني أي شيء عنهم!\n\nمثال: 'ملخص البيانات' أو 'مريض 1'"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message-user">🙋‍♂️ **أنت:** {message["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-bot">🤖 **مساعد البيانات:** {message["content"]}</div>',
                        unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("اكتب سؤالك هنا...")

    if user_question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.markdown(f'<div class="chat-message-user">🙋‍♂️ **أنت:** {user_question}</div>', unsafe_allow_html=True)

        # Process and add response
        with st.spinner("جاري التحليل..."):
            response = process_question(user_question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f'<div class="chat-message-bot">🤖 **مساعد البيانات:** {response}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("🤖 RapidDx - شات ذكي للاستعلام عن بيانات المستشفى وتقييم النموذج")
