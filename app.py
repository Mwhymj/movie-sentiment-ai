import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from pythainlp.tokenize import word_tokenize

# --- 1. ฟังก์ชันตัดคำ ---
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

# --- 2. โหลดโมเดล 2 เวอร์ชั่น ---
@st.cache_resource
def load_models():
    try:
        # ตรวจสอบว่ามีไฟล์ model.joblib และ model_v2.joblib ในโฟลเดอร์
        m_v1 = joblib.load('model.joblib')
        m_v2 = joblib.load('model_v2.joblib')
        return m_v1, m_v2
    except:
        return None, None

# --- 3. โหลดข้อมูล ---
@st.cache_data
def load_data():
    return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')

model_v1, model_v2 = load_models()
df = load_data()

# --- 4. การตกแต่ง UI Style ---
st.set_page_config(page_title="AI Model Iteration Analysis", layout="wide")

st.markdown("""
    <style>
    /* จัดการ Font และพื้นหลัง */
    .stApp { background-color: #fcfcfc; }
    .main-card { background: white; padding: 25px; border-radius: 12px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .model-label { font-size: 1.1rem; font-weight: 700; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 5px; margin-bottom: 15px; }
    .feature-tag { background: #e8f0fe; color: #1967d2; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; margin: 2px; display: inline-block; }
    /* แถบ Metrics ด้านล่าง */
    .footer-box { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# --- ส่วนหัวข้อ ---
st.title("🔬 AI Model Iteration & Error Analysis")
st.write("การเปรียบเทียบผลลัพธ์ระหว่างโมเดลรุ่น **Baseline (V1)** และรุ่น **Optimized (V2)**")

# --- Toolbar สำหรับจัดการข้อมูล ---
if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("🎲 สุ่มรีวิวใหม่", use_container_width=True):
        s = df.sample(1).iloc[0]
        st.session_state.update({'h': f"ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
        st.rerun()
with c2:
    if st.button("🧹 ล้างหน้าจอ", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- ส่วนการรับข้อมูล (Input Section) ---
if model_v1 is None or model_v2 is None:
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณารันไฟล์ train_model_v2.py ก่อนเพื่อสร้างไฟล์ .joblib")
else:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    in_c1, in_c2 = st.columns([3, 1])
    with in_c1:
        headline = st.text_input("Review Headline / ID:", value=st.session_state.get('h', ''))
    with in_c2:
        true_label = st.selectbox("Actual Label (Ground Truth):", ["Positive", "Neutral", "Negative"], 
                                 index=["Positive", "Neutral", "Negative"].index(st.session_state.get('l', 'Positive')))

    body = st.text_area("Review Content:", value=st.session_state.get('b', ''), height=120)

    if st.button("⚡ เริ่มการวิเคราะห์เปรียบเทียบ (Analyze Comparison)", type="primary", use_container_width=True):
        if body.strip():
            full_text = f"{headline} {body}"
            
            st.markdown("---")
            col_v1, col_v2 = st.columns(2)

            # ฟังก์ชันดึง Important Words
            def get_features(model, text):
                try:
                    tfidf = model.named_steps['tfidf']
                    clf = model.named_steps['clf']
                    feature_names = tfidf.get_feature_names_out()
                    tokens = thai_tokenize(text)
                    present = [f for f in tokens if f in feature_names]
                    if not present: return []
                    pred_class = model.predict([text])[0]
                    idx = list(clf.classes_).index(pred_class)
                    weights = clf.coef_[idx]
                    results = []
                    for f in set(present):
                        f_idx = np.where(feature_names == f)[0][0]
                        results.append((f, weights[f_idx]))
                    return sorted(results, key=lambda x: x[1], reverse=True)[:5]
                except: return []

            # --- การแสดงผล Model V1 ---
            with col_v1:
                st.markdown('<div class="model-label">🤖 MODEL V1 (Baseline)</div>', unsafe_allow_html=True)
                prob1 = model_v1.predict_proba([full_text])[0]
                pred1 = model_v1.classes_[np.argmax(prob1)]
                conf1 = np.max(prob1) * 100
                
                st.write(f"ผลทำนาย: **{pred1}** {'✅' if pred1 == true_label else '❌'}")
                st.progress(int(conf1))
                st.caption(f"ความมั่นใจ: {conf1:.2f}%")
                
                st.write("🔍 คำที่มีอิทธิพลต่อ V1:")
                for w, weight in get_features(model_v1, full_text):
                    st.markdown(f"<span class='feature-tag'>{w}</span>", unsafe_allow_html=True)

            # --- การแสดงผล Model V2 ---
            with col_v2:
                st.markdown('<div class="model-label">🚀 MODEL V2 (Improved)</div>', unsafe_allow_html=True)
                prob2 = model_v2.predict_proba([full_text])[0]
                pred2 = model_v2.classes_[np.argmax(prob2)]
                conf2 = np.max(prob2) * 100
                
                st.write(f"ผลทำนาย: **{pred2}** {'✅' if pred2 == true_label else '❌'}")
                st.progress(int(conf2))
                st.caption(f"ความมั่นใจ: {conf2:.2f}%")
                
                st.write("🔍 คำที่มีอิทธิพลต่อ V2:")
                for w, weight in get_features(model_v2, full_text):
                    st.markdown(f"<span class='feature-tag'>{w}</span>", unsafe_allow_html=True)
        else:
            st.error("กรุณากรอกข้อมูลรีวิวก่อนกดวิเคราะห์")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. แถบแสดงสถิติภาพรวม (Footer Metrics) ---
st.markdown('<div class="footer-box">', unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-weight: bold;'>📊 Global Model Statistics</p>", unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(label="Dataset Size", value="5,000", delta="Synthetic Data")
with m_col2:
    st.metric(label="Model Accuracy", value="100%", delta="5-Fold CV")
with m_col3:
    st.metric(label="Algorithm", value="Logistic", delta="V1 (Base) / V2 (N-gram)")
with m_col4:
    st.metric(label="Library", value="PyThaiNLP", delta="newmm engine")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #ccc; font-size: 0.7rem; margin-top: 20px;'>Thai Movie Sentiment Analysis | NLP Iteration Project</p>", unsafe_allow_html=True)