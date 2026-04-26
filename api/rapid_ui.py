import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(
    page_title="RapidDx - التشخيص السريع لضيق التنفس",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
    .critical-box { background: #ff6b6b; padding: 1rem; border-radius: 10px; color: white; }
    .warning-box { background: #ffd43b; padding: 1rem; border-radius: 10px; color: #333; }
    .success-box { background: #51cf66; padding: 1rem; border-radius: 10px; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏥 RapidDx | التشخيص السريع لضيق التنفس</h1>
    <p>نظام دعم قرار مبني على الذكاء الاصطناعي + إرشادات طبية عالمية</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 معلومات النظام")
    st.info("""
    **المستويات التشخيصية:**
    🔰 المستوى 1: أعراض فقط (دقة ~72%)
    ⚡ المستوى 2: أعراض + BNP (دقة ~88%)
    ✅ المستوى 3: شامل + أشعة (دقة ~94%)
    """)
    st.warning("⚠️ هذا النظام للدعم التشخيصي فقط")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 إدخال بيانات المريض")
    with st.expander("🩺 الأعراض الأساسية", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.number_input("العمر", 18, 120, 65)
            temp = st.number_input("الحرارة", 35.0, 42.0, 37.0, 0.1)
            rr = st.number_input("معدل التنفس", 8, 50, 18)
        with col_b:
            cough = 1 if st.radio("كحة", ["لا", "نعم"], index=0) == "نعم" else 0
            orthopnea = 1 if st.radio("ضيق تنفس بالاستلقاء", ["لا", "نعم"], index=0) == "نعم" else 0
        with col_c:
            leg_swelling = 1 if st.radio("تورم الرجلين", ["لا", "نعم"], index=0) == "نعم" else 0
            weight_gain = 1 if st.radio("زيادة وزن حديثة", ["لا", "نعم"], index=0) == "نعم" else 0
    
    with st.expander("🧪 التحاليل السريعة"):
        has_bnp = st.checkbox("متوفر BNP")
        bnp_val = st.number_input("BNP", 0, 5000, 250) if has_bnp else None
    
    with st.expander("🩻 الأشعة"):
        has_xray = st.checkbox("متوفرة صورة صدر")
        if has_xray:
            xray_congestion = 1 if st.radio("احتقان رئوي", ["لا", "نعم"], index=0) == "نعم" else 0
            xray_infiltrate = 1 if st.radio("ارتشاحات", ["لا", "نعم"], index=0) == "نعم" else 0
        else:
            xray_congestion = None
            xray_infiltrate = None
    
    diagnose_btn = st.button("🔍 بدء التشخيص", type="primary", use_container_width=True)

with col2:
    if has_xray and has_bnp:
        st.success("✅ مستوى 3: تشخيص كامل (دقة ~94%)")
    elif has_bnp:
        st.info("⚡ مستوى 2: تشخيص سريع (دقة ~88%)")
    else:
        st.warning("🔰 مستوى 1: فرز أولي (دقة ~72%)")

if diagnose_btn:
    # تجهيز البيانات بالشكل الصحيح
    patient_data = {
        "age": float(age),
        "temperature": float(temp),
        "respiratory_rate": float(rr),
        "cough": int(cough),
        "weight_gain": int(weight_gain),
        "non_adherence": 0,
        "chest_pain": 0,
        "orthopnea": int(orthopnea),
        "leg_swelling": int(leg_swelling),
        "bnp": float(bnp_val) if has_bnp and bnp_val else None,
        "xray_congestion": int(xray_congestion) if has_xray and xray_congestion is not None else None,
        "xray_infiltrate": int(xray_infiltrate) if has_xray and xray_infiltrate is not None else None
    }
    
    with st.spinner("جاري التشخيص..."):
        try:
            # استخدام 127.0.0.1 بدلاً من localhost
            response = requests.post("http://127.0.0.1:8000/diagnose", json=patient_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.markdown("---")
                st.subheader("📊 نتيجة التشخيص")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    hfe_prob = result.get('hfe_probability', 0)
                    st.metric("🫀 فشل القلب", f"{hfe_prob*100:.1f}%")
                    st.progress(hfe_prob)
                with col_r2:
                    ari_prob = result.get('ari_probability', 0)
                    st.metric("🫁 مرض تنفسي", f"{ari_prob*100:.1f}%")
                    st.progress(ari_prob)
                
                severity = result.get('red_flags', {}).get('severity', 'LOW')
                if severity == "CRITICAL":
                    st.markdown('<div class="critical-box"><h3>🚨 حالة حرجة - تدخل فوري!</h3></div>', unsafe_allow_html=True)
                elif severity == "HIGH":
                    st.markdown('<div class="warning-box"><h3>⚠️ حالة عالية الخطورة</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box"><h3>✅ حالة مستقرة</h3></div>', unsafe_allow_html=True)
                
                st.success(f"**التشخيص:** {result.get('primary_diagnosis', 'غير محدد')}")
                st.info(f"**التوصية:** {result.get('recommendation', 'لا توجد')}")
                st.caption(f"🕐 {result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
                
            else:
                st.error(f"خطأ {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ خطأ في الاتصال - تأكد من تشغيل الـ API أولاً")
        except Exception as e:
            st.error(f"❌ خطأ: {str(e)}")
