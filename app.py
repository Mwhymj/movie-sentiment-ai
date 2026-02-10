import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. UI CONFIG: THE "FUTURISTIC GLASS" DESIGN ---
st.set_page_config(page_title="CineSense Neural Interface", layout="wide")

def apply_pro_design():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600&family=JetBrains+Mono:wght@300&display=swap');

        /* พื้นหลังแบบลึก (Deep Space) */
        .stApp {
            background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Glass Container ที่ดูแพง */
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-bottom: 25px;
        }

        /* หัวข้อแบบ Gradient Text */
        .hero-title {
            background: linear-gradient(to right, #38bdf8, #81
