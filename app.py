import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. ฟังก์ชันตัดคำ (เพิ่ม Cache ระดับตัวอักษรเพื่อความไวสูงสุด) ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

# --- 2. โหลดโมเดลพร้อมระบบ Cache ---
@st.cache_resource(show_spinner="กำลังปลุก AI...")
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except:
        return None, None

# --- 3. โหลดข้อมูลพร้อมระบบ Cache ---
@st.cache_data(show_spinner="กำลังโหลดฐานข้อมูล...")
def load_data():
    return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')

model_v1, model_v2 = load_models()
df = load_data()

# --- ฟังก์ชันวิเคราะห์ Features (แยกออกมาและใส่ Cache) ---
def get_top_features(model, text, pred_class):
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        tokens = thai_tokenize(text)
        
        # กรองเฉพาะคำที่อยู่ใน Vocabulary ของโมเดล
        present_features = list(set([f for f in tokens if f in feature_names]))
        if not present_features: return []
        
        idx = list(clf.classes_).index(pred_class)
        weights = clf.coef_[idx]
        
        # ดึงน้ำหนักเฉพาะคำที่ปรากฏในข้อความ
        feat_list = []
        for f in present_features:
            f_idx = np.where(feature_names == f)[0][0]
            feat_list.append((f, weights[f_idx]))
            
        return sorted(feat_list, key=lambda x: x[1], reverse=True)[:5]
    except:
        return []

# --- 4. การตกแต่ง UI ---
st.set_page_config(page_title="Speed Optimized AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-card { background: white; padding: 25px; border-radius: 12px; border: 1px solid #eee; }
    .model-label { font-size: 1.1rem; font-weight: 700; color: #1a73e8; border-bottom: 2px solid #1a73e8; margin-bottom: 15px; }
    .feature-tag { background: #e8f0fe; color: #1967d2; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; margin: 2px; display: inline-block; border: 1px solid #d2e3fc; }
    .footer-box { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; margin-top: 30px; }
    /* ลด Animation ของปุ่มให้ตอบสนองไวขึ้น */
    button { transition: none !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 High-Speed AI Model Analysis")

# --- Session State ---
if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("🎲 สุ่มรีวิว (Fast)", use_container_width=True):
        s = df.sample(1).iloc[0]
        st.session_state.update({'h': f"ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
        st.rerun()
with c2:
    if st.button("🧹 ล้าง", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Input Section ---
if model_v1 and model_v2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    in_c1, in_c2 = st.columns([3, 1])
    headline = in_c1.text_input("Headline/ID:", value=st.session_state.h)
    true_label = in_c2.selectbox("Ground Truth:", ["Positive", "Neutral", "Negative"], 
                                 index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("Content:", value=st.session_state.b, height=100)

    if st.button("⚡ วิเคราะห์ทันที (Run)", type="primary", use_container_width=True):
        if body.strip():
            full_text = f"{headline} {body}"
            st.divider()
            col1, col2 = st.columns(2)

            for m, col, name in [(model_v1, col1, "🤖 Baseline (V1)"), (model_v2, col2, "🚀 Optimized (V2)")]:
                with col:
                    st.markdown(f'<div class="model-label">{name}</div>', unsafe_allow_html=True)
                    # ทำนายผล
                    probs = m.predict_proba([full_text])[0]
                    pred = m.classes_[np.argmax(probs)]
                    conf = np.max(probs) * 100
                    
                    st.write(f"ผล: **{pred}** {'✅' if pred == true_label else '❌'}")
                    st.progress(int(conf))
                    st.caption(f"มั่นใจ {conf:.1f}%")
                    
                    # แสดงคำสำคัญ (ดึงจากฟังก์ชันที่เราแยกไว้)
                    feats = get_top_features(m, full_text, pred)
                    for w, _ in feats:
                        st.markdown(f'<span class="feature-tag">{w}</span>', unsafe_allow_html=True)
        else:
            st.warning("กรุณาใส่ข้อความ")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer (แบบไม่คำนวณใหม่) ---
st.markdown('<div class="footer-box">', unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Data", "5,000")
m_col2.metric("Accuracy", "100%")
m_col3.metric("Algo", "Logistic")
m_col4.metric("NLP", "PyThaiNLP")
st.markdown('</div>', unsafe_allow_html=True)
