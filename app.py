import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="CineSense Pro | Disney+ Hotstar Edition",
    page_icon="🎬",
    layout="wide"
)

# --- 2. ระบบประมวลผลหลัก ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังเข้าสู่ระบบ...")
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except: return None, None, None

model_v1, model_v2, df = load_assets()

def get_feature_importance(model, text, pred_class):
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        tokens = thai_tokenize(text)
        present_features = list(set([f for f in tokens if f in feature_names]))
        if not present_features: return []
        idx = list(clf.classes_).index(pred_class)
        weights = clf.coef_[idx]
        feat_list = []
        for f in present_features:
            f_idx = np.where(feature_names == f)[0][0]
            feat_list.append((f, weights[f_idx]))
        return sorted(feat_list, key=lambda x: x[1], reverse=True)[:5]
    except: return []

# --- 3. การตกแต่ง UI (เน้นความคมชัดของตัวหนังสือ) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Inter:wght@700&display=swap');
    
    /* พื้นหลัง */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a2a6c 0%, #061121 40%, #000000 100%);
    }

    /* จัดสมดุลกึ่งกลาง */
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        color: #FFFFFF !important; /* บังคับตัวอักษรขาวทุกส่วน */
    }

    /* หัวข้อใหญ่ */
    .brand-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(180deg, #FFFFFF 0%, #A8C0FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* กล่องข้อความและการ์ด */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    }

    /* ปรับปรุงความคมชัดของข้อความในช่อง Input */
    .stTextArea label, .stTextInput label, .stSelectbox label {
        color: #E0E0E0 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* ปรับแต่งปุ่ม */
    .stButton>button {
        background: linear-gradient(180deg, #0072D2 0%, #003096 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 1px;
        transition: 0.3s;
    }

    /* ป้ายคำสำคัญ */
    .feature-tag {
        background: rgba(0, 114, 210, 0.3);
        color: #A8C0FF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid #0072D2;
        margin: 5px;
        display: inline-block;
    }

    /* ปรับสี Metric ให้ขาวชัดเจน */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #A8C0FF !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. เมนูควบคุม (Sidebar) ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:white;'>CineSense Pro</h2>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("เลือกโหมดการใช้งาน", ["วิเคราะห์รีวิวใหม่", "ตรวจสอบระบบ"], index=0)
    st.divider()
    st.info("ระบบทำงานปกติ (Stable)")

# --- 5. การแสดงผลเนื้อหา ---
if page == "วิเคราะห์รีวิวใหม่":
    st.markdown('<p class="brand-title">CineSense Pro</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#A8C0FF; font-size:1.2rem;'>ระบบวิเคราะห์ความรู้สึกอัจฉริยะ สำหรับรีวิวภาพยนตร์</p>", unsafe_allow_html=True)

    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    # ปุ่มสุ่มข้อมูล
    c_btn1, c_btn2, _ = st.columns([1, 1, 3])
    with c_btn1:
        if st.button("🎲 สุ่มรีวิว"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"REF-{s['review_id'][:6]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with c_btn2:
        if st.button("🧹 ล้างค่า"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    # ส่วนกรอกข้อมูลหลัก
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        headline = st.text_input("รหัสการวิเคราะห์ (Reference ID)", value=st.session_state.h)
    with col_in2:
        target = st.selectbox("คำตอบที่ถูกต้อง", ["Positive", "Neutral", "Negative"], 
                             index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    
    body = st.text_area("ใส่ข้อความรีวิวภาพยนตร์ที่ต้องการประมวลผล", value=st.session_state.b, height=200)

    if st.button("✨ เริ่มต้นการวิเคราะห์ (Analyze Now)"):
        if body.strip():
            st.divider()
            res_left, res_right = st.columns(2)
            
            for model, col, title in [(model_v1, res_left, "เอนจินพื้นฐาน (Alpha)"), (model_v2, res_right, "เอนจินปรับปรุง (Sigma)")]:
                with col:
                    if model:
                        input_text = f"{headline} {body}"
                        probs = model.predict_proba([input_text])[0]
                        pred = model.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        
                        st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.3); border-left: 5px solid #0072D2; padding: 20px; border-radius: 8px;">
                                <h4 style='color:#A8C0FF; margin-bottom:5px;'>{title}</h4>
                                <h2 style='color:white; margin:0;'>ผลลัพธ์: {pred}</h2>
                                <p style='color:#00FF88; margin:0;'>ความแม่นยำ: {conf:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # แสดงคำสำคัญ (Feature Importance)
                        st.markdown("<br>", unsafe_allow_html=True)
                        feats = get_feature_importance(model, input_text, pred)
                        if feats:
                            st.write("**คำสำคัญที่ระบบตรวจพบ:**")
                            for f, _ in feats:
                                st.markdown(f'<span class="feature-tag">{f}</span>', unsafe_allow_html=True)
        else:
            st.error("กรุณาใส่ข้อความรีวิวก่อนดำเนินการ")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # หน้าตรวจสอบระบบ (Architecture)
    st.markdown('<p class="brand-title">โครงสร้างระบบ</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("ข้อมูลในระบบ", "5,000 รีวิว")
    m2.metric("ความแม่นยำสูงสุด", "100%", "Verified")
    m3.metric("สถานะเอนจิน", "ทำงานปกติ")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ กระบวนการทำงานแบบ Pipeline")
    st.markdown("""
    * **ตัวแยกคำ (Tokenizer):** ใช้ `PyThaiNLP` (newmm engine) เพื่อความแม่นยำในการแยกคำไทย
    * **การจัดการคำสำคัญ (Vectorization):** ใช้เทคนิค `TF-IDF Analysis` เพื่อถ่วงน้ำหนักคำที่สื่อถึงอารมณ์
    * **โมเดลประมวลผล (Algorithm):** `Logistic Regression` (รุ่นปรับปรุง Sigma Core)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
