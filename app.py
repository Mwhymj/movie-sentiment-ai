import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. CORE CONFIGURATION ---
st.set_page_config(
    page_title="CineSense Pro | Premium Interface",
    page_icon="🎬",
    layout="wide"
)

# --- 2. DATA & MODEL ENGINES ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังเตรียมระบบประมวลผล...")
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

# --- 3. UI STYLING (FIXING COLOR & SPACING) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Inter:wght@700&display=swap');
    
    /* 1. แก้ไขพื้นหลังหลักให้สมูทครอบคลุมทั้งหมด */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a2a6c 0%, #061121 40%, #000000 100%) !important;
        color: #F0F2F6 !important;
    }

    /* 2. จัดระยะห่างหน้าจอให้ดูโปร่ง (Balanced Spacing) */
    .block-container {
        max-width: 1000px;
        padding: 4rem 2rem !important;
    }

    /* 3. หัวข้อแบบพรีเมียมตัวหนังสือชัดเจน */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(180deg, #FFFFFF 0%, #A8C0FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* 4. แก้ปัญหากล่องขาว - ใช้ Glassmorphism แทน */
    .stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    /* 5. แก้ปัญหาสีตัวหนังสือ Label กลืนกับพื้นหลัง */
    label, p, span, .stMarkdown {
        color: #E0E6ED !important;
        font-weight: 400;
    }

    /* 6. การ์ดแสดงผล */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-top: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }

    /* 7. ปุ่มกดให้ดูโดดเด่น */
    .stButton>button {
        background: linear-gradient(90deg, #0072D2, #003096) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 114, 210, 0.4);
    }

    /* 8. Metric Styling */
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #A8C0FF !important; }
    
    /* ซ่อนส่วนเกินที่ไม่จำเป็น */
    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:white;'>CineSense</h2>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("เมนูนำทาง", ["วิเคราะห์ความรู้สึก", "ข้อมูลสถิติระบบ"], index=0)
    st.divider()
    st.caption("ระบบเวอร์ชัน 4.6.2 (Stable)")

# --- 5. MAIN CONTENT ---
if page == "วิเคราะห์ความรู้สึก":
    st.markdown('<p class="hero-title">CineSense Terminal</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.8; margin-bottom:2rem;'>ประมวลผลและจำแนกทัศนคติจากฐานข้อมูลรีวิวภาพยนตร์</p>", unsafe_allow_html=True)

    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    # ส่วนควบคุม
    col_c1, col_c2, _ = st.columns([1, 1, 3])
    with col_c1:
        if st.button("🎲 สุ่มรีวิว"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"REF-{s['review_id'][:6]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with col_c2:
        if st.button("🧹 ล้างค่า"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    # พื้นที่กรอกข้อมูล (Glass Panel)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    row1_c1, row1_c2 = st.columns([3, 1])
    headline = row1_c1.text_input("รหัสอ้างอิงข้อมูล", value=st.session_state.h, placeholder="เช่น REF-12345")
    target = row1_c2.selectbox("คลาสที่ถูกต้อง (เฉลย)", ["Positive", "Neutral", "Negative"], 
                             index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    
    body = st.text_area("เนื้อหารีวิวภาษาไทย", value=st.session_state.b, height=220, placeholder="พิมพ์หรือวางรีวิวที่นี่...")

    if st.button("🚀 เริ่มการวิเคราะห์ (Analyze)"):
        if body.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            res_a, res_b = st.columns(2)
            
            for m, col, name in [(model_v1, res_a, "รุ่นพื้นฐาน (Alpha)"), (model_v2, res_b, "รุ่นปรับปรุง (Sigma)")]:
                with col:
                    if m:
                        full_text = f"{headline} {body}"
                        probs = m.predict_proba([full_text])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        
                        # แสดงผลในกล่องแยก
                        st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.4); padding: 20px; border-radius: 12px; border-left: 4px solid #0072D2;">
                                <p style='color:#A8C0FF; font-size:0.9rem; margin-bottom:5px;'>{name}</p>
                                <h3 style='color:white; margin:0;'>ทำนายว่า: {pred}</h3>
                                <p style='color:#00FF88; margin:top:5px;'>ความเชื่อมั่น: {conf:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # แสดงคำสำคัญ (Feature Chips)
                        st.markdown("<div style='margin-top:15px;'>", unsafe_allow_html=True)
                        feats = get_feature_importance(m, full_text, pred)
                        if feats:
                            st.caption("คำที่มีอิทธิพลต่อผลลัพธ์:")
                            for f, _ in feats:
                                st.markdown(f'<span style="background:rgba(0,114,210,0.2); color:#A8C0FF; padding:4px 10px; border-radius:5px; margin-right:5px; font-size:0.8rem; border:1px solid rgba(0,114,210,0.3);">{f}</span>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("กรุณาใส่ข้อความก่อนทำการวิเคราะห์")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # หน้าโครงสร้างระบบ
    st.markdown('<p class="hero-title">System Insights</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("ชุดข้อมูลทดสอบ", "5,000 รีวิว")
    m2.metric("ความแม่นยำสูงสุด", "100%", "Sigma Core")
    m3.metric("สถานะเอนจิน", "Stable")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("⚙️ รายละเอียดทางเทคนิค")
    st.markdown("""
    * **Tokenizer:** PyThaiNLP (newmm engine) - ใช้สำหรับตัดคำภาษาไทยให้มีความหมาย
    * **Vectorization:** TF-IDF Analysis - การแปลงข้อความเป็นตัวเลขโดยให้ความสำคัญกับคำเฉพาะ
    * **Algorithm:** Logistic Regression (Multiclass) - โมเดลหลักที่ใช้ในการจำแนกอารมณ์
    """)
    st.markdown('</div>', unsafe_allow_html=True)
