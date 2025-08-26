import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
import pytesseract
from nltk.sentiment import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Setup NLTK
nltk.download("vader_lexicon")
nltk.download("punkt")

sia = SentimentIntensityAnalyzer()
stemmer = StemmerFactory().create_stemmer()

# === Ekstrak teks dari PDF dengan OCR ===
def extract_text_ocr(pdf_file):
    images = convert_from_bytes(pdf_file.read())  # convert pdf ke list gambar
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="ind") + "\n"
    return text

# === Analisis sentimen sederhana (Bahasa Indonesia) ===
def get_sentiment(text):
    text_stem = stemmer.stem(text.lower())
    score = sia.polarity_scores(text_stem)["compound"]
    if score >= 0.05:
        return "Positif"
    elif score <= -0.05:
        return "Negatif"
    else:
        return "Netral"

# === Streamlit App ===
st.title("📊 Analisis Sentimen Saran Mahasiswa (OCR + Bahasa Indonesia)")

uploaded_file = st.file_uploader("Upload file PDF Kuesioner", type="pdf")

if uploaded_file is not None:
    extracted_text = extract_text_ocr(uploaded_file)
    suggestions = [line.strip() for line in extracted_text.split("\n") if line.strip()]

    df = pd.DataFrame(suggestions, columns=["Saran"])
    df["Sentimen"] = df["Saran"].apply(get_sentiment)

    if not df.empty:
        st.subheader("Hasil Analisis Sentimen")
        st.dataframe(df, use_container_width=True)

        st.subheader("Distribusi Sentimen")
        st.bar_chart(df["Sentimen"].value_counts())

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="sentimen_saran.csv", mime="text/csv")

        # Filter
        st.subheader("🔎 Filter Sentimen")
        filter_option = st.radio("Pilih kategori:", ["Semua", "Positif", "Negatif", "Netral"])
        if filter_option != "Semua":
            st.dataframe(df[df["Sentimen"] == filter_option], use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada teks yang berhasil di-OCR dari PDF.")
