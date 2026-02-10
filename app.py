import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from functools import lru_cache

# --- [STEP 1] ต้องอยู่บรรทัดแรกสุด ห้ามมีอะไรอยู่ก่อน ---
st.set_page_config(page_title="CineSense Pro | Sentiment AI", layout="wide")

# --- [STEP 2] ฟังก์ชัน OPTIMIZATION (ทำให้รันไว) ---
@lru_cache(maxsize=1000)
def thai_tokenize_fast(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="Initializing Neural Core...")
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except Exception as e:
        return None, None, None

model_v1, model_v2, df = load_assets()

# --- [STEP 3] CUSTOM CSS (ดีไซน์สไตล์ Modern & Clean) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=Kanit:wght@300;400&display=swap');
    
    .stApp {
        background-color: #f8fafc;
        font-family: 'Plus Jakarta Sans', 'Kanit', sans-serif;
    }
    
    /* บังคับตกแต่ง Card */
    .css-1y4p8pa, .st-emotion-cache-1y4p8pa {
        padding: 1.5rem;
    }

    /* ตกแต่งกล่องข้อความ */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: white !important;
        border-radius: 16px !important;
        padding:
