# app.py

import streamlit as st
import numpy as np
import pickle

# ============================
# Load Model Random Forest
# ============================
with open("models/modelRF_penyakit_jantung.pkl", "rb") as file:
    model = pickle.load(file)

# ============================
# UI Streamlit
# ============================

# Judul
st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.write("Masukkan data kesehatan Anda di bawah untuk mengetahui kemungkinan risiko penyakit jantung.")

# ============================
# Input User
# ============================

umur = st.slider("Umur", min_value=30, max_value=70, value=40)
tekanan_darah = st.slider("Tekanan Darah (mmHg)", min_value=100, max_value=180, value=120)
kolesterol = st.slider("Kolesterol (mg/dL)", min_value=150, max_value=300, value=200)
detak_jantung = st.slider("Detak Jantung Maksimum", min_value=100, max_value=180, value=130)

# ============================
# Prediksi
# ============================
if st.button("🔍 Prediksi Sekarang"):
    data_input = np.array([[umur, tekanan_darah, kolesterol, detak_jantung]])
    hasil_prediksi = model.predict(data_input)[0]

    if hasil_prediksi == 1:
        st.error("⚠️ Anda berisiko terkena penyakit jantung.")
    else:
        st.success("✅ Anda tidak berisiko terkena penyakit jantung.")

# ============================
# Keterangan Tambahan
# ============================
st.markdown("---")
st.caption("Model ini menggunakan algoritma Random Forest yang dilatih dengan data simulasi.")
