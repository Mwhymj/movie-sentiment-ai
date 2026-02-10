import streamlit as st
from textblob import TextBlob

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1, h2, h3, p {
    color: white;
}

.stTextArea textarea {
    background-color: #262730;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("พิมพ์รีวิวหนัง แล้วระบบจะวิเคราะห์อารมณ์ให้")

# -------------------------------
# Input Text
# -------------------------------
review = st.text_area("✏️ ใส่รีวิวหนังของคุณ")

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("🔍 วิเคราะห์ความรู้สึก"):

    if review.strip() == "":
        st.warning("กรุณาใส่ข้อความก่อน")
    else:
        blob = TextBlob(review)
        polarity = blob.sentiment.polarity

        # -----------------------
        # Sentiment Result
        # -----------------------
        if polarity > 0:
            st.success("😊 ความรู้สึกเชิงบวก (Positive)")
        elif polarity < 0:
            st.error("😡 ความรู้สึกเชิงลบ (Negative)")
        else:
            st.info("😐 ความรู้สึกเป็นกลาง (Neutral)")

        st.write(f"คะแนน Sentiment: {polarity:.2f}")
