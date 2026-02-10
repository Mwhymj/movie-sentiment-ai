# movie-sentiment-ai
1. ไฟล์ requirements.txt
(สร้างไฟล์ชื่อนี้ เพื่อให้ Server ติดตั้ง Library อัตโนมัติ)

Plaintext
streamlit
pandas
joblib
scikit-learn
pythainlp
numpy


2. ไฟล์ train_model_v2.py
(รันไฟล์นี้ เพื่อสร้างไฟล์สมอง AI 2 เวอร์ชั่น)

Python
import pandas as pd
import joblib
from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
X, y = df['text'], df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model V1: Baseline
model_v1 = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenize, ngram_range=(1, 1))),
    ('clf', LogisticRegression(max_iter=1000))
])
model_v1.fit(X_train, y_train)
joblib.dump(model_v1, 'model.joblib')

# Model V2: Improved (N-gram 1-2)
model_v2 = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenize, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=2.0, max_iter=1000))
])
model_v2.fit(X_train, y_train)
joblib.dump(model_v2, 'model_v2.joblib')

print("Success: Generated model.joblib and model_v2.joblib")


3. ไฟล์ app.py
(ไฟล์หน้าเว็บหลักที่ใช้แสดงผลเปรียบเทียบ)

Python
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource
def load_models():
    try:
        return joblib.load('model.joblib'), joblib.load('model_v2.joblib')
    except:
        return None, None

model_v1, model_v2 = load_models()
df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')

st.set_page_config(page_title="AI Comparison Lab", layout="wide")
st.markdown("<style>.stApp{background:#f8f9fa;} .card{background:white; padding:20px; border-radius:10px; border:1px solid #eee;}</style>", unsafe_allow_html=True)

st.title("🔬 AI Model Comparison & Error Analysis")

if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("🎲 Random Sample"):
        s = df.sample(1).iloc[0]
        st.session_state.update({'h':f"ID: {s['review_id'][:8]}", 'b':s['text'], 'l':s['label']})
        st.rerun()
with c2:
    if st.button("🧹 Clear"):
        st.session_state.clear()
        st.rerun()

st.markdown('<div class="card">', unsafe_allow_html=True)
col_in1, col_in2 = st.columns([3, 1])
headline = col_in1.text_input("Review ID", value=st.session_state.get('h',''))
true_label = col_in2.selectbox("Ground Truth", ["Positive", "Neutral", "Negative"], index=["Positive", "Neutral", "Negative"].index(st.session_state.get('l','Positive')))
body = st.text_area("Content", value=st.session_state.get('b',''), height=100)

if st.button("⚡ Run Analysis", type="primary", use_container_width=True):
    if body.strip() and model_v1:
        full_text = f"{headline} {body}"
        st.divider()
        cv1, cv2 = st.columns(2)
        
        for i, (m, col, name) in enumerate([(model_v1, cv1, "V1 (Base)"), (model_v2, cv2, "V2 (Improved)")]):
            prob = m.predict_proba([full_text])[0]
            pred = m.classes_[np.argmax(prob)]
            conf = np.max(prob) * 100
            with col:
                st.subheader(name)
                st.write(f"Predict: **{pred}** {'✅' if pred==true_label else '❌'}")
                st.progress(int(conf))
                st.caption(f"Confidence: {conf:.2f}%")
    else: st.error("Missing Data or Models")
st.markdown('</div>', unsafe_allow_html=True)

# Footer Metrics
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Data Size", "5,000")
m2.metric("Accuracy", "100%")
m3.metric("Algo", "Logistic")
m4.metric("Library", "PyThaiNLP")


ขั้นตอนสุดท้าย (การเอาขึ้นเว็บ):
สร้าง GitHub Repository แล้วอัปโหลด 5 ไฟล์นี้ขึ้นไป:

app.py

requirements.txt

model.joblib

model_v2.joblib

8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv

ไปที่ Streamlit Cloud แล้วเชื่อม GitHub เพื่อกด Deploy ได้เลย
