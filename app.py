import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. ตั้งค่าเบื้องต้น ---
st.set_page_config(
    page_title="CineSense Pro | ระบบวิเคราะห์รีวิวหนัง",
    page_icon="🎬",
    layout="wide"
)

# --- 2. ฟังก์ชันระบบหลังบ้าน ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังเปิดระบบวิเคราะห์อัจฉริยะ...")
def load_assets():
    try:
        m1 = joblib.load('model.joblib')
        m2 = joblib.load('model_v2.joblib')
        df = pd.read_csv('8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv')
        return m1, m2, df
    except:
        return None, None, None

model_v1, model_v2, df = load_assets()

def get_feature_importance(model, text, pred_class):
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

# --- 3. การตกแต่งสไตล์ Netflix (Dark & Elegant) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    
    .stApp { background-color: #0b0b0b; color: #ffffff; font-family: 'Kanit', sans-serif; }
    
    /* การ์ดและกล่องข้อมูล */
    .premium-card {
        background: rgba(30, 30, 30, 0.8);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* หัวข้อและตัวหนังสือ */
    h1, h2, h3 { color: #E50914 !important; }
    label, p { color: #cccccc !important; font-size: 1.1rem !important; }

    /* ช่องกรอกข้อมูล */
    .stTextArea textarea, .stTextInput input {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* ปุ่มกดสีแดง Netflix */
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        width: 100%;
    }
    
    /* แถบสีแสดงสถานะ */
    .match-tag { color: #00FF88; font-weight: bold; border: 1px solid #00FF88; padding: 2px 8px; border-radius: 4px; }
    .mismatch-tag { color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 4px; }
    
    /* คำสำคัญ */
    .keyword-chip {
        background: rgba(229, 9, 20, 0.2);
        color: #ff4b55;
        padding: 4px 10px;
        border-radius: 15px;
        margin-right: 5px;
        font-size: 0.85rem;
        border: 1px solid #E50914;
        display: inline-block;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. เมนูข้างทาง ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>CineSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>ระบบวิเคราะห์อารมณ์รีวิวหนัง</p>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("เลือกหน้าเมนู", ["หนูหน้าหลัก (วิเคราะห์)", "เจาะลึกข้อผิดพลาด", "ข้อมูลทางเทคนิค"], index=0)
    st.divider()
    st.info("สถานะระบบ: พร้อมใช้งาน")

# --- 5. เนื้อหาแต่ละหน้า ---

if menu == "หนูหน้าหลัก (วิเคราะห์)":
    st.markdown("## 🎬 เริ่มการวิเคราะห์รีวิวของคุณ")
    
    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("🎲 สุ่มรีวิว"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"ID-{s['review_id'][:6]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with c2:
        if st.button("🧹 ล้างหน้าจอ"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    r1, r2 = st.columns([3, 1])
    h_input = r1.text_input("รหัสอ้างอิงข้อมูล", value=st.session_state.h, placeholder="เช่น REF-001")
    l_input = r2.selectbox("เฉลยจริง (Label)", ["Positive", "Neutral", "Negative"], 
                         index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    
    b_input = st.text_area("ใส่เนื้อหารีวิวหนังที่นี่", value=st.session_state.b, height=200)

    if st.button("🚀 เริ่มวิเคราะห์เดี๋ยวนี้"):
        if b_input.strip():
            st.markdown("### ผลการวิเคราะห์จากระบบ AI")
            col_v1, col_v2 = st.columns(2)
            
            for model, col, name in [(model_v1, col_v1, "ระบบรุ่นมาตรฐาน (Alpha)"), (model_v2, col_v2, "ระบบรุ่นอัจฉริยะ (Sigma)")]:
                with col:
                    if model:
                        # สร้างประโยคสำหรับวิเคราะห์ (เลียนแบบการรวม Text ในขั้นตอน Preprocessing)
                        input_text = f"{h_input} {b_input}"
                        probs = model.predict_proba([input_text])[0]
                        pred = model.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        is_match = pred == l_input
                        
                        tag = f'<span class="match-tag">ทายถูก</span>' if is_match else f'<span class="mismatch-tag">ทายผิด</span>'
                        
                        st.markdown(f"""
                            <div style="background:#222; padding:20px; border-radius:10px; border-top: 4px solid #E50914; min-height: 180px;">
                                <p style='margin:0;'>{name}</p>
                                <h2 style='margin:10px 0; color:white !important;'>{pred} {tag}</h2>
                                <p style='font-size:0.9rem; color:#888;'>ระดับความมั่นใจ: {conf:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(conf))
                        
                        # --- ส่วนที่เพิ่มเข้าไป: แสดงคำสำคัญสำหรับแต่ละโมเดล ---
                        st.markdown("<p style='margin-top:10px; font-weight:bold;'>คำสำคัญที่ใช้ตัดสินใจ:</p>", unsafe_allow_html=True)
                        feats = get_feature_importance(model, input_text, pred)
                        if feats:
                            for w, _ in feats:
                                st.markdown(f'<span class="keyword-chip">{w}</span>', unsafe_allow_html=True)
                        else:
                            st.caption("ไม่พบคำสำคัญเด่นชัด")
                        # ---------------------------------------------------
                        
        else: st.error("กรุณาใส่ข้อความรีวิวก่อนครับ")
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "เจาะลึกข้อผิดพลาด":
    st.markdown("## 🔍 วิเคราะห์จุดที่ AI ทายผิด (Error Analysis)")
    st.write("ตารางด้านล่างแสดงตัวอย่างที่โมเดล Sigma ทายไม่ตรงกับเฉลยจริง")

    if df is not None and model_v2 is not None:
        sample_df = df.sample(50)
        sample_df['AI_Predict'] = model_v2.predict(sample_df['text'])
        errors = sample_df[sample_df['label'] != sample_df['AI_Predict']]

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if not errors.empty:
            for idx, row in errors.head(5).iterrows():
                with st.expander(f"❌ รีวิวรหัส {row['review_id'][:6]} | เฉลย: {row['label']} | AI ทายว่า: {row['AI_Predict']}"):
                    st.write(f"**เนื้อหารีวิว:** {row['text']}")
                    st.info("สาเหตุที่อาจผิด: คำประชดประชัน หรือ ประโยคมีความหมายก้ำกึ่ง")
        else:
            st.success("ยอดเยี่ยม! ในชุดข้อมูลสุ่มนี้ AI ทายถูกทั้งหมด")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("## ⚙️ ข้อมูลเชิงเทคนิค")
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.write("- **ฐานข้อมูล:** รีวิวหนังไทย 5,000 รายการ")
    st.write("- **เทคนิคการตัดคำ:** PyThaiNLP (ตัดคำภาษาไทยแบบแม่นยำ)")
    st.write("- **โมเดลที่ใช้:** Logistic Regression (ตัวเลือกที่ดีที่สุดสำหรับข้อความ)")
    st.write("- **รุ่น Sigma:** เพิ่มพลังด้วย N-gram ทำให้เข้าใจบริบทของคำที่อยู่ติดกัน")
    st.markdown('</div>', unsafe_allow_html=True)
