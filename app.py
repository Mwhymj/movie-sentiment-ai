import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="CineSense Netflix Edition",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD ASSETS ----------------
@st.cache_resource
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except:
        return None, None, None

model_v1, model_v2, df = load_assets()

# ---------------- TOKEN ----------------
def thai_tokenize(text):
    return word_tokenize(str(text), engine="newmm")

# ---------------- NETFLIX CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: black;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
}

/* Hero */
.hero {
    background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,1)),
    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
    padding: 60px;
    border-radius: 20px;
}

/* Card */
.netflix-card {
    background-color: #141414;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Badge */
.badge {
    color: #e50914;
    font-weight: bold;
}

/* Button */
.stButton>button {
    background-color: #e50914;
    border-radius: 30px;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='color:#e50914'>CineSense</h1>
""", unsafe_allow_html=True)

# ---------------- HERO INPUT ----------------
st.markdown("<div class='hero'>", unsafe_allow_html=True)

st.subheader("Alpha Terminal")

headline = st.text_input("Analysis ID")
review = st.text_area("Movie Review", height=150)

col1, col2 = st.columns(2)

run = col1.button("EXECUTE ANALYSIS")
random_btn = col2.button("Random")

if random_btn and df is not None:
    row = df.sample(1).iloc[0]
    review = row["text"]
    headline = row["review_id"]
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if run and review.strip():

    colA, colB = st.columns(2)

    for model, col, name in [
        (model_v1, colA, "Alpha Engine"),
        (model_v2, colB, "Sigma Core")
    ]:

        with col:
            st.markdown("<div class='netflix-card'>", unsafe_allow_html=True)

            st.markdown(f"### {name}")

            probs = model.predict_proba([review])[0]
            pred = model.classes_[np.argmax(probs)]
            conf = np.max(probs) * 100

            st.metric("Prediction", pred)
            st.progress(int(conf))
            st.caption(f"Confidence {conf:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR CHAT MOCK ----------------
with st.sidebar:
    st.markdown("### Tactical Comms")

    st.chat_message("assistant").write(
        "Neural analysis ready. Input review to begin sentiment scan."
    )

    user_input = st.chat_input("Ask CineSense...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(
            "Feature not connected yet — demo UI only."
        )
