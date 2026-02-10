import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
import time

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="CineSense Netflix")

# ---------------- LOAD ----------------
@st.cache_resource
def load_assets():
    try:
        m1 = joblib.load("model.joblib")
        m2 = joblib.load("model_v2.joblib")
        df = pd.read_csv("8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv")
        return m1, m2, df
    except:
        return None, None, None

model_v1, model_v2, df = load_assets()

# ---------------- CSS Netflix ----------------
st.markdown("""
<style>

body {
    background-color: black;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
}

/* Navbar */
.navbar {
    backdrop-filter: blur(12px);
    background-color: rgba(20,20,20,0.7);
    padding: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
