import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. CORE CONFIGURATION ---
st.set_page_config(
    page_title="CineSense Pro | Netflix Edition",
    page_icon="🎬",
    layout="wide"
)

# --- 2. DATA ENGINE ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="Loading Cinematic Intelligence...")
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except:
        return None, None, None

model_v1, model_v2, df = load_assets()

# --- 3. PREMIUM DARK STYLING (Netflix Style) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Kanit:wght@300;500&display=swap');
    
    /* Main Background */
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
        font-family: 'Inter', 'Kanit', sans-serif;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #333;
    }

    /* Card Styling */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Typography */
    h1, h2, h3 { color: #E50914 !important; font-weight: 700 !important; }
    p, label, .stMarkdown { color: #e5e5e5 !important; }

    /* Input & Text Area */
    .stTextArea textarea, .stTextInput input {
        background-color: #222 !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }

    /* Custom Buttons (Netflix Red) */
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.3s ease all;
    }
    .stButton>button:hover {
        background-color: #ff0a16 !important;
        transform: scale(1.02);
    }

    /* Feature Chips */
    .feature-chip {
        background: rgba(229, 9, 20, 0.15);
        color: #ff4b55;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 8px;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }

    /* Error Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom:0;'>CineSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#808080;'>PRO EDITION v4.6.2</p>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("SELECT MODE", ["MAIN TERMINAL", "ERROR ANALYSIS", "ARCHITECTURE"], index=0)
    st.divider()
    st.markdown("### Model Status")
    st.success("● Sigma Core: Online")
    st.caption("Last Updated: Feb 2026")

# --- 5. PAGE ROUTING ---

if menu == "MAIN TERMINAL":
    st.markdown("<h2>Analyze Cinematic Sentiment</h2>", unsafe_allow_html=True)
    
    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("🎲 RANDOM"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"REF-{s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with col_btn2:
        if st.button("🧹 CLEAR"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    headline = c1.text_input("Entry Reference", value=st.session_state.h, placeholder="e.g. MOVIE-REVIEW-001")
    target = c2.selectbox("True Label", ["Positive", "Neutral", "Negative"], 
                        index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("Review Content", value=st.session_state.b, height=200, placeholder="Paste Thai movie review here...")

    if st.button("INITIATE AI ANALYSIS", use_container_width=True):
        if body.strip():
            st.markdown("### Analysis Results")
            r_col1, r_col2 = st.columns(2)
            for m, col, name in [(model_v1, r_col1, "Alpha (Baseline)"), (model_v2, r_col2, "Sigma (Advanced)")]:
                with col:
                    if m:
                        probs = m.predict_proba([f"{headline} {body}"])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        is_correct = pred == target
                        
                        status_color = "#00FF88" if is_correct else "#FF4B4B"
                        st.markdown(f"""
                            <div style="border-left: 5px solid {status_color}; background: rgba(255,255,255,0.03); padding: 20px; border-radius: 0 10px 10px 0;">
                                <p style='margin:0; font-weight:bold; color:#E50914;'>{name}</p>
                                <h2 style='margin:10px 0; color:white !important;'>{pred}</h2>
                                <p style='margin:0; font-size: 0.9rem;'>Accuracy: <span style='color:{status_color}'>{'MATCHED' if is_correct else 'MISMATCHED'}</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(conf))
                        st.caption(f"Confidence Level: {conf:.2f}%")
        else: st.warning("Please enter text for analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "ERROR ANALYSIS":
    st.markdown("<h2>Error Analysis Dashboard</h2>", unsafe_allow_html=True)
    st.write("ตรวจสอบและวิเคราะห์เคสที่โมเดล Sigma Core ทำนายผิดพลาดเพื่อหาแนวทางปรับปรุง")

    if df is not None and model_v2 is not None:
        # ประมวลผลหา Mismatch (ในแอปจริงควรสุ่มมาโชว์เพื่อความเร็ว)
        test_sample = df.sample(100) # สุ่มมา 100 เคสเพื่อความเร็วหน้าเว็บ
        preds = model_v2.predict(test_sample['text'])
        test_sample['Prediction'] = preds
        errors = test_sample[test_sample['label'] != test_sample['Prediction']]

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.write(f"พบข้อผิดพลาด **{len(errors)}** เคส จากการสุ่มตรวจสอบ 100 รายการ")
        
        if not errors.empty:
            for i, row in errors.head(5).iterrows():
                with st.expander(f"❌ Error ใน Review ID: {row['review_id'][:8]} (True: {row['label']} | AI: {row['Prediction']})"):
                    st.write(f"**ข้อความ:** {row['text']}")
                    st.divider()
                    st.caption("Possible Reason: Contextual Ambiguity / Sarcasm")
        else:
            st.success("No errors found in this sample batch!")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("<h2>System Architecture</h2>", unsafe_allow_html=True)
    
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Total Records", "5,000")
    a2.metric("Architecture", "Logistic Regression")
    a3.metric("N-Gram Range", "(1, 2)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("Technical Pipeline")
    st.write("1. **Tokenization:** PyThaiNLP (newmm engine)")
    st.write("2. **Vectorization:** TF-IDF with Bi-gram support")
    st.write("3. **Classifier:** Logistic Regression with Hyperparameter C=2.0")
    st.markdown('</div>', unsafe_allow_html=True)
