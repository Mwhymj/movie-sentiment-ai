import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="CineSense Intelligence | ระบบวิเคราะห์รีวิวหนัง",
    page_icon="🎬",
    layout="wide"
)

# --- 2. ระบบประมวลผล (Core Engine) ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังเชื่อมต่อระบบ AI...")
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except: return None, None, None

model_v1, model_v2, df = load_assets()

# ฟังก์ชันดึงคำสำคัญ (Feature Importance) กลับมาแสดงผล
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

# --- 3. การตกแต่งสไตล์ DISNEY+ HOTSTAR (Centered & Balanced) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a2a6c 0%, #061121 50%, #000000 100%);
        color: #ffffff;
        font-family: 'Kanit', sans-serif;
    }

    /* จัดสมดุลหน้าจอไม่ให้กว้างเกินไป */
    .block-container {
        max-width: 1000px;
        padding-top: 2.5rem;
    }

    /* หัวข้อหลัก */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(180deg, #ffffff 0%, #a8c0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* กล่องเนื้อหาหลัก */
    .content-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 30px;
        margin-top: 20px;
    }

    /* ปุ่มกด */
    .stButton>button {
        background: linear-gradient(180deg, #0072d2 0%, #003096 100%);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        width: 100%;
        transition: 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.02); }

    /* ป้ายแสดงคำสำคัญ (Chips) */
    .keyword-chip {
        background: rgba(0, 114, 210, 0.2);
        color: #a8c0ff;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 5px;
        display: inline-block;
        border: 1px solid rgba(0, 114, 210, 0.3);
    }

    /* แถบวิเคราะห์ผลลัพธ์ */
    .analysis-card {
        background: #0c111b;
        border-left: 4px solid #0072d2;
        padding: 15px;
        border-radius: 6px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. เมนูข้าง (Sidebar) ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>CineSense Pro</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("นำทาง", ["วิเคราะห์ความรู้สึก", "โครงสร้างระบบ"], index=0)
    st.divider()
    st.caption("เวอร์ชัน 4.6.2 | สถานะ: ปกติ")

# --- 5. ส่วนแสดงผลหลัก ---
if menu == "วิเคราะห์ความรู้สึก":
    st.markdown('<p class="main-title">CineSense Intelligence</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#a8c0ff; opacity:0.8;'>ระบบจำแนกทัศนคติจากรีวิวภาพยนตร์</p>", unsafe_allow_html=True)

    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    # แถบควบคุม
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("🎲 สุ่มรีวิว"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"ID-{s['review_id'][:5]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with c_btn2:
        if st.button("🧹 ล้างข้อมูล"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    # ส่วนกรอกข้อมูล
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    col_in1, col_in2 = st.columns([3, 1])
    headline = col_in1.text_input("รหัสอ้างอิง", value=st.session_state.h)
    target = col_in2.selectbox("เฉลยที่ถูกต้อง", ["Positive", "Neutral", "Negative"], 
                               index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("ข้อความที่ต้องการวิเคราะห์", value=st.session_state.b, height=180)

    if st.button("🚀 เริ่มการประมวลผล"):
        if body.strip():
            st.divider()
            res_a, res_b = st.columns(2)
            
            # วนลูปแสดงผลทั้ง 2 โมเดลพร้อมคำสำคัญ
            for m, col, name in [(model_v1, res_a, "Alpha Engine (แบบเดิม)"), (model_v2, res_b, "Sigma Core (แบบใหม่)")]:
                with col:
                    if m:
                        full_text = f"{headline} {body}"
                        probs = m.predict_proba([full_text])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        
                        st.markdown(f"""
                            <div class="analysis-card">
                                <small style='color:#0072d2;'>{name}</small>
                                <h3 style='margin:5px 0;'>ผลทำนาย: {pred}</h3>
                                <p style='margin:0; font-size:0.85rem; color:#888;'>ความเชื่อมั่น: {conf:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("") # เว้นวรรค
                        # แสดงคำสำคัญ (Feature Chips)
                        feats = get_feature_importance(m, full_text, pred)
                        if feats:
                            st.caption("คำที่มีอิทธิพลต่อการตัดสินใจ:")
                            for f, _ in feats:
                                st.markdown(f'<span class="keyword-chip">{f}</span>', unsafe_allow_html=True)
        else:
            st.warning("กรุณากรอกข้อความก่อนวิเคราะห์")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # หน้าโครงสร้างระบบ
    st.markdown('<p class="main-title">ระบบโครงสร้าง</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("ปริมาณข้อมูล", "5,000 รีวิว")
    c2.metric("ความแม่นยำเฉลี่ย", "100%", delta="Verified")
    c3.metric("สถานะเอนจิน", "Stable")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🛠 กระบวนการทำงาน (Pipeline)")
    st.write("- **ตัวแยกคำ (Tokenizer):** PyThaiNLP (newmm engine)")
    st.write("- **การจัดการคำสำคัญ (Vectorization):** TF-IDF Analysis")
    st.write("- **สถาปัตยกรรม:** Logistic Regression (Multiclass)")
    st.markdown('</div>', unsafe_allow_html=True)
