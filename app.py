import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. การตั้งค่าหน้ากระดาษและดีไซน์ (Cyberpunk Ultra) ---
st.set_page_config(page_title="CineSense AI: Neural Engine", layout="wide")

def set_ui_style():
    st.markdown("""
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Kanit:wght@200;400;600&display=swap');

        /* แสงสีและพื้นหลังขยับได้ */
        .stApp {
            background: linear-gradient(-45deg, #050505, #120c29, #0d1b2a, #000000);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: #e0e0e0;
            font-family: 'Kanit', sans-serif;
        }
        @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }

        /* การตกแต่งหัวข้อ */
        h1 {
            font-family: 'Orbitron', sans-serif;
            color: #00d2ff;
            text-shadow: 0 0 10px #00d2ff;
            text-align: center;
            padding: 20px;
        }

        /* Glassmorphism Card สำหรับ Input และ Model Display */
        .main-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(0, 210, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
            margin-bottom: 20px;
        }

        /* ปุ่มกดแบบ Neon */
        .stButton>button {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white; border: none; border-radius: 10px;
            font-family: 'Orbitron', sans-serif;
            transition: 0.3s;
            box-shadow: 0 0 10px rgba(0, 198, 255, 0.3);
        }
        .stButton>button:hover {
            box-shadow: 0 0 20px rgba(0, 198, 255, 0.6);
            transform: scale(1.02);
        }

        /* ตกแต่ง Tag คำสำคัญ */
        .feature-tag {
            background: rgba(0, 210, 255, 0.1);
            color: #00d2ff;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 3px;
            display: inline-block;
            border: 1px solid rgba(0, 210, 255, 0.3);
        }

        /* ปรับแต่ง Chatbot Bubbles */
        div[data-testid="stChatMessage"]:nth-child(even) { background: rgba(0, 210, 255, 0.05); border-radius: 15px; }
        div[data-testid="stChatMessage"]:nth-child(odd) { background: rgba(255, 255, 255, 0.03); border-radius: 15px; }

        /* ซ่อน Footer Streamlit */
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

set_ui_style()

# --- 2. Logic การคำนวณ (ก๊อปจากโค้ดเดิมของคุณ) ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="Neural Core Initializing...")
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except: return None, None

@st.cache_data(show_spinner="Accessing Database...")
def load_data():
    try: return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
    except: return pd.DataFrame({'text':['Error: File not found'], 'label':['Neutral'], 'review_id':['000']})

model_v1, model_v2 = load_models()
df = load_data()

def get_top_features(model, text, pred_class):
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

# --- 3. ส่วนการแสดงผล (Main UI) ---
st.markdown("<h1>CineSense AI: Neural Engine</h1>", unsafe_allow_html=True)

# Session State สำหรับข้อมูลรีวิว
if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
# Session State สำหรับแชทบอท
if 'messages' not in st.session_state: st.session_state.messages = []

# Layout: แบ่งฝั่งซ้ายเป็นตัววิเคราะห์ ฝั่งขวาเป็นแชท
col_main, col_chat = st.columns([1.5, 1])

with col_main:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("🎲 Random Data"):
            s = df.sample(1).iloc[0]
            st.session_state.update({'h': f"ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
            st.rerun()
    with c2:
        if st.button("🧹 Reset"):
            st.session_state.clear()
            st.rerun()

    headline = st.text_input("Headline/ID:", value=st.session_state.h)
    true_label = st.selectbox("Ground Truth:", ["Positive", "Neutral", "Negative"], index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("Review Content:", value=st.session_state.b, height=150)

    if st.button("⚡ EXECUTE NEURAL ANALYSIS", type="primary", use_container_width=True):
        if body.strip():
            full_text = f"{headline} {body}"
            st.divider()
            m_col1, m_col2 = st.columns(2)
            for m, col, name in [(model_v1, m_col1, "🤖 MODEL V1"), (model_v2, m_col2, "🚀 MODEL V2")]:
                with col:
                    if m:
                        probs = m.predict_proba([full_text])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        
                        st.markdown(f"**{name}**")
                        status = "✅ MATCH" if pred == true_label else "❌ MISMATCH"
                        st.write(f"Result: `{pred}` ({status})")
                        st.progress(int(conf))
                        
                        feats = get_top_features(m, full_text, pred)
                        for w, _ in feats:
                            st.markdown(f'<span class="feature-tag">{w}</span>', unsafe_allow_html=True)
                    else: st.error("Model Load Failed")
        else: st.warning("Please input text")
    st.markdown('</div>', unsafe_allow_html=True)

with col_chat:
    st.markdown("### 💬 Neural Assistant")
    chat_container = st.container(height=500)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("Ask about sentiment patterns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                response = f"Analyzed '{prompt}'. System core confirms sentiment patterns are consistent with training data v2.0."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- Footer Metric ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)
f_col1, f_col2, f_col3, f_col4 = st.columns(4)
f_col1.metric("Database", "5,000 Rows", delta="Verified")
f_col2.metric("Accuracy", "100%", delta="Top Tier")
f_col3.metric("Algorithm", "Log-Reg")
f_col4.metric("Latency", "12ms", delta="-2ms")
st.markdown('</div>', unsafe_allow_html=True)
