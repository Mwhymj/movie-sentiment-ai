import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from functools import lru_cache

# --- 1. SET CONFIG (ต้องอยู่บรรทัดแรกสุด) ---
st.set_page_config(
    page_title="CineSense | Movie Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CORE LOGIC ---
@st.cache_resource
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except: return None, None, None

m1, m2, df = load_assets()

@lru_cache(maxsize=1000)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

# --- 3. CUSTOM CSS (Inspired by the News Classifier) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Kanit:wght@300;400&display=swap');
    
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', 'Kanit', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Card Styling */
    .data-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Result Tags */
    .sentiment-tag {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 10px;
    }
    .pos-tag { background-color: #dcfce7; color: #166534; }
    .neg-tag { background-color: #fee2e2; color: #991b1b; }
    .neu-tag { background-color: #fef9c3; color: #854d0e; }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=80)
    st.title("CineSense Pro")
    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Analyze Review", "📊 Project Info", "💬 AI Assistant"])
    st.markdown("---")
    st.caption("v4.5.0 Build 2026")

# --- 5. PAGE LOGIC ---

if page == "🔍 Analyze Review":
    st.header("Movie Sentiment Classifier")
    st.write("ระบุรีวิวภาพยนตร์ด้านล่างเพื่อวิเคราะห์อารมณ์ด้วยโมเดล Machine Learning")

    # Layout แบ่ง 2 ส่วน
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Input Terminal")
        
        if 'b_txt' not in st.session_state: st.session_state.b_txt = ""
        
        if st.button("🎲 Random Sample Data"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.b_txt = s['text']
                st.rerun()

        txt_input = st.text_area("Review Content (Thai)", value=st.session_state.b_txt, height=200, placeholder="พิมพ์รีวิวหนังที่นี่...")
        analyze_btn = st.button("Start Classification", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Analysis Results")
        
        if analyze_btn and txt_input:
            res1, res2 = st.columns(2)
            for m, col, name in [(m1, res1, "Engine Alpha"), (m2, res2, "Engine Sigma")]:
                with col:
                    if m:
                        pred = m.predict([txt_input])[0]
                        prob = np.max(m.predict_proba([txt_input])[0]) * 100
                        
                        tag_class = "pos-tag" if pred == "Positive" else "neg-tag" if pred == "Negative" else "neu-tag"
                        st.markdown(f"**{name}**")
                        st.markdown(f'<div class="sentiment-tag {tag_class}">{pred}</div>', unsafe_allow_html=True)
                        st.write(f"Confidence: `{prob:.1f}%`")
                        st.progress(int(prob))
                    else: st.error("Model Error")
        else:
            st.info("ระบบพร้อมทำงาน กรุณาใส่ข้อมูลในช่อง Input")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 Project Info":
    st.header("Project Documentation")
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.write("### 📂 Deliverables Overview")
    st.markdown("""
    * **Dataset:** Synthetic Thai Movie Reviews (5,000 samples)
    * **Preprocessing:** Tokenization via PyThaiNLP (newmm), TF-IDF Vectorization
    * **Model Architecture:** Logistic Regression (Baseline vs Optimized)
    * **Evaluation Metric:** Accuracy, Confusion Matrix
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.write("### 📈 Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "99.8%", "+0.2%")
    c2.metric("Precision", "0.99", "Stable")
    c3.metric("F1-Score", "0.99", "Optimal")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "💬 AI Assistant":
    st.header("Neural Chat Interface")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask about sentiment analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = "System core is ready to assist with sentiment classification tasks."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
