import streamlit as st

def set_ultra_modern_css():
    st.markdown("""
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Kanit:wght@200;400;600&display=swap');

        /* พื้นหลังแบบ Animated Gradient */
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #000000);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: #ffffff;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ปรับแต่งกล่อง Card ให้ดูเป็นกระจก Sci-Fi (Glassmorphism) */
        div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0, 198, 255, 0.1);
            transition: transform 0.3s ease;
        }

        /* หัวข้อวิบวับแบบ Neon */
        h1 {
            font-family: 'Orbitron', sans-serif;
            color: #00d2ff;
            text-shadow: 0 0 10px #00d2ff, 0 0 20px #00d2ff;
            text-align: center;
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        /* ปุ่มกดแบบ Cyberpunk Neon */
        .stButton>button {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 30px;
            font-weight: bold;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            box-shadow: 0 0 15px rgba(0, 198, 255, 0.5);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
            width: 100%;
        }

        .stButton>button:hover {
            box-shadow: 0 0 30px rgba(0, 198, 255, 0.8);
            transform: scale(1.05);
            color: #ffffff;
        }

        /* ช่อง Input ที่ดูโปร่งแสง */
        .stTextInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: white !important;
            border: 1px solid rgba(0, 198, 255, 0.3) !important;
            border-radius: 15px;
            padding: 10px 20px;
            font-family: 'Kanit', sans-serif;
        }

        /* ซ่อนขยะที่ไม่อยากให้เห็น */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* สไตล์ข้อความทั่วไป */
        p, span, label {
            font-family: 'Kanit', sans-serif;
            font-weight: 200;
        }
        </style>
    """, unsafe_allow_html=True)

set_ultra_modern_css()

# --- Content Area ---
st.markdown("<h1>Movie Sentiment AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00d2ff; opacity:0.8;'>Deep Learning Neural Engine v3.0</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])
with col2:
    review = st.text_input("", placeholder="Enter movie review to analyze...")
    if st.button("EXECUTE ANALYSIS"):
        with st.spinner('Accessing Neural Core...'):
            # ตัวอย่างผลลัพธ์แบบเล่นใหญ่
            st.markdown("""
                <div style='background:rgba(0, 255, 127, 0.1); border-left: 5px solid #00ff7f; padding: 15px; border-radius: 10px;'>
                    <h3 style='color:#00ff7f; margin:0;'>POSITIVE DETECTED</h3>
                    <p style='color:#ffffff; margin:0;'>Confidence Score: 98.4%</p>
                </div>
            """, unsafe_allow_html=True)

