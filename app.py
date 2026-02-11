import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize

# --- 1. การตั้งค่าระบบหลัก ---
st.set_page_config(
    page_title="CineSense Pro | ระบบวิเคราะห์อารมณ์รีวิวหนัง",
    page_icon="🎬",
    layout="wide"
)

# --- 2. เครื่องมือประมวลผลข้อมูล ---
@st.cache_data(show_spinner=False)
def thai_tokenize(text):
    return word_tokenize(str(text), engine='newmm')

@st.cache_resource(show_spinner="กำลังโหลดสมองกลอัจฉริยะ...")
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

# --- 3. ตกแต่งดีไซน์พรีเมียม (Dark Mode - Netflix Style) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Kanit:wght@300;500&display=swap');
    
    .stApp { background-color: #0f0f0f; color: #ffffff; font-family: 'Inter', 'Kanit', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #141414 !important; border-right: 1px solid #333; }

    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    h1, h2, h3 { color: #E50914 !important; font-weight: 700 !important; }
    p, label, .stMarkdown { color: #e5e5e5 !important; }

    .stTextArea textarea, .stTextInput input {
        background-color: #222 !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }

    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        width: 100%;
    }
    
    .keyword-tag {
        background: rgba(229, 9, 20, 0.15);
        color: #ff4b55;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 8px;
        display: inline-block;
        margin-top: 8px;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. เมนูนำทาง ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom:0;'>CineSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#808080;'>รุ่นโปร v4.6.2</p>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("เมนูการใช้งาน", ["หน้าวิเคราะห์หลัก", "เจาะลึกข้อผิดพลาด", "โครงสร้างระบบ"], index=0)
    st.divider()
    st.success("● ระบบ Sigma Core: ออนไลน์")

# --- 5. การแสดงผลแต่ละหน้า ---

if menu == "หน้าวิเคราะห์หลัก":
    st.markdown("<h2>วิเคราะห์ความรู้สึกจากรีวิวหนัง</h2>", unsafe_allow_html=True)
    
    if 'h' not in st.session_state: st.session_state.update({'h':'', 'b':'', 'l':'Positive'})

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("🎲 สุ่มข้อมูล"):
            if df is not None:
                s = df.sample(1).iloc[0]
                st.session_state.update({'h': f"ID-{s['review_id'][:8]}", 'b': s['text'], 'l': s['label']})
                st.rerun()
    with col_btn2:
        if st.button("🧹 ล้างหน้าจอ"):
            st.session_state.update({'h':'', 'b':'', 'l':'Positive'})
            st.rerun()

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    headline = c1.text_input("รหัสอ้างอิงรีวิว", value=st.session_state.h, placeholder="เช่น MOVIE-001")
    target = c2.selectbox("ผลลัพธ์ที่ถูกต้อง (เฉลย)", ["Positive", "Neutral", "Negative"], 
                        index=["Positive", "Neutral", "Negative"].index(st.session_state.l))
    body = st.text_area("เนื้อหารีวิวหนัง", value=st.session_state.b, height=200, placeholder="พิมพ์หรือวางรีวิวหนังที่นี่...")

    if st.button("เริ่มการวิเคราะห์ด้วย AI", use_container_width=True):
        if body.strip():
            st.markdown("### ผลลัพธ์จากการประมวลผล")
            r_col1, r_col2 = st.columns(2)
            input_full = f"{headline} {body}"
            
            for m, col, name in [(model_v1, r_col1, "รุ่นมาตรฐาน (Alpha)"), (model_v2, r_col2, "รุ่นอัจฉริยะ (Sigma)")]:
                with col:
                    if m:
                        probs = m.predict_proba([input_full])[0]
                        pred = m.classes_[np.argmax(probs)]
                        conf = np.max(probs) * 100
                        is_correct = pred == target
                        status_color = "#00FF88" if is_correct else "#FF4B4B"
                        
                        st.markdown(f"""
                            <div style="border-left: 5px solid {status_color}; background: rgba(255,255,255,0.03); padding: 20px; border-radius: 0 10px 10px 0; min-height: 150px;">
                                <p style='margin:0; font-weight:bold; color:#E50914;'>{name}</p>
                                <h2 style='margin:10px 0; color:white !important;'>{pred}</h2>
                                <p style='margin:0; font-size: 0.9rem;'>สถานะ: <span style='color:{status_color}'>{'ตรงกับเฉลย' if is_correct else 'ไม่ตรงกับเฉลย'}</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(conf))
                        st.caption(f"ระดับความมั่นใจของ AI: {conf:.2f}%")
                        
                        # แสดงคำสำคัญที่ใช้ตัดสินใจ
                        st.write("คำสำคัญที่ใช้ตัดสินใจ:")
                        feats = get_feature_importance(m, input_full, pred)
                        if feats:
                            for word, _ in feats:
                                st.markdown(f'<span class="keyword-tag">{word}</span>', unsafe_allow_html=True)
                        else:
                            st.caption("ไม่พบคำสำคัญที่เด่นชัด")
        else: st.warning("กรุณากรอกรีวิวก่อนเริ่มการวิเคราะห์")
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "เจาะลึกข้อผิดพลาด":
    st.markdown("<h2>วิเคราะห์จุดอ่อนของโมเดล (Error Analysis)</h2>", unsafe_allow_html=True)
    st.write("ตรวจสอบตัวอย่างที่โมเดล Sigma ทายผิดเพื่อนำไปพัฒนาต่อ")
    
    if df is not None and model_v2 is not None:
        test_sample = df.sample(100)
        preds = model_v2.predict(test_sample['text'])
        test_sample['Prediction'] = preds
        errors = test_sample[test_sample['label'] != test_sample['Prediction']]

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.write(f"พบเคสที่ทายผิด **{len(errors)}** รายการ จากข้อมูลสุ่ม 100 รายการ")
        for i, row in errors.head(5).iterrows():
            with st.expander(f"❌ รีวิว ID: {row['review_id'][:8]} (เฉลย: {row['label']} | AI ทาย: {row['Prediction']})"):
                st.write(f"**เนื้อหา:** {row['text']}")
                st.divider()
                st.caption("สาเหตุที่เป็นไปได้: การประชดประชัน หรือ ประโยคมีความหมายคลุมเครือ")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("<h2>โครงสร้างทางเทคนิค</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("จำนวนข้อมูล", "5,000 รายการ")
    a2.metric("โมเดลหลัก", "Logistic Regression")
    a3.metric("เทคนิคพิเศษ", "N-Gram (1, 2)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("ขั้นตอนการทำงานของระบบ")
    st.write("1. **การตัดคำ:** ใช้ PyThaiNLP (Engine: newmm)")
    st.write("2. **แปลงเป็นตัวเลข:** ใช้ TF-IDF (รองรับคำคู่ Bi-grams)")
    st.write("3. **การจำแนก:** ใช้ Logistic Regression (ปรับแต่งค่า C=2.0 เพื่อความแม่นยำ)")
    st.markdown('</div>', unsafe_allow_html=True)
