import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. ฟังก์ชันตั้งค่า Netflix Theme (CSS) ---
def set_netflix_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700&display=swap');

        /* พื้นหลังดำ Netflix */
        .stApp {
            background-color: #141414;
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }

        /* กล่องเนื้อหา */
        div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
            background: #181818;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }

        /* หัวข้อ Bebas Neue แบบ Netflix */
        h1 {
            font-family: 'Bebas Neue', cursive;
            color: #E50914;
            font-size: 4.5rem !important;
            text-align: center;
            margin-bottom: 0px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        /* ปุ่มแดง Netflix */
        .stButton>button {
            background-color: #E50914;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 12px 24px;
            font-weight: 700;
            text-transform: uppercase;
            width: 100%;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #ff0a16;
            color: white;
        }

        /* ช่อง Input */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #333333 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px;
        }

        /* สไตล์ Card ผลลัพธ์ */
        .result-card {
            background: #2f2f2f;
            border-left: 5px solid #E50914;
            padding: 20px;
            border-radius: 4px;
            margin-top: 10px;
        }

        .feature-tag {
            background: #444;
            color: #E50914;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            margin: 2px;
            display: inline-block;
            border: 1px solid #E50914;
        }

        #MainMenu, footer, header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# --- 2. ระบบหลังบ้าน (Optimized Caching) ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except:
        return None, None

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')

# เรียกใช้งานฟังก์ชัน
set_netflix_theme()
model_v1, model_v2 = load_models()
df = load_data()

# --- 3. ส่วนการแสดงผล (UI) ---
st.markdown("<h1>NETFLIX</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#E50914; font-weight:700; margin-top:-25px; letter-spacing:3px;'>SENTIMENT ANALYSIS LAB</p>", unsafe_allow_html=True)

if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

# Toolbar
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button("🎲 RANDOM"):
        s = df.sample(1).iloc[0]
        st.session_state.update({'h': f"ID: {s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
        st.rerun()
with c2:
    if st.button("🧹 CLEAR"):
        st.session_state.clear()
        st.rerun()

# Input Section
if model_v1 and model_v2:
    col_in1, col_in2 = st.columns([3, 1])
    headline = col_in1.text_input("Review ID", value=st.session_state.h)
    true_label = col_in2.selectbox("Ground Truth", ["Positive", "Neutral", "Negative"], 
                                  index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("Review Content", value=st.session_state.b, height=100)

    if st.button("ANALYZE SENTIMENT", type="primary"):
        if body.strip():
            full_text = f"{headline} {body}"
            st.divider()
            
            res_c1, res_c2 = st.columns(2)

            def get_features(model, text, pred_class):
                try:
                    tfidf = model.named_steps['tfidf']
                    clf = model.named_steps['clf']
                    feature_names = tfidf.get_feature_names_out()
                    tokens = thai_tokenize(text)
                    present = list(set([f for f in tokens if f in feature_names]))
                    idx = list(clf.classes_).index(pred_class)
                    weights = clf.coef_[idx]
                    results = [(f, weights[np.where(feature_names == f)[0][0]]) for f in present]
                    return sorted(results, key=lambda x: x[1], reverse=True)[:5]
                except: return []

            for m, col, name in [(model_v1, res_c1, "MODEL V1 (Baseline)"), (model_v2, res_c2, "MODEL V2 (Optimized)")]:
                probs = m.predict_proba([full_text])[0]
                pred = m.classes_[np.argmax(probs)]
                conf = np.max(probs) * 100
                with col:
                    st.markdown(f"### {name}")
                    st.markdown(f"""
                        <div class="result-card">
                            <h2 style='color:#E50914; margin:0;'>{pred.upper()}</h2>
                            <p style='margin:0;'>Confidence: {conf:.1f}% {'✅' if pred == true_label else '❌'}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    feats = get_features(m, full_text, pred)
                    for f, _ in feats:
                        st.markdown(f'<span class="feature-tag">{f}</span>', unsafe_allow_html=True)
        else:
            st.error("Please enter some text to analyze.")

# Footer
st.markdown("<br><hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
f1.caption("DATABASE: 5,000 REVIEWS")
f2.caption("ENGINE: LOGISTIC REGRESSION")
f3.caption("STATUS: SYSTEM OPERATIONAL")
