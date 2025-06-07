import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("model_spam.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Aplikasi Deteksi Spam")

text = st.text_area("Masukkan pesan di sini:")

if st.button("Deteksi"):
    if text.strip() == "":
        st.warning("Tolong masukkan pesan.")
    else:
        text_vector = vectorizer.transform([text])
        result = model.predict(text_vector)[0]
        st.success(f"Hasil: **{'SPAM' if result == 1 else 'BUKAN SPAM'}**")
