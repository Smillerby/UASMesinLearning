import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Fungsi untuk mengonversi input data menjadi format yang dapat digunakan oleh model
def preprocess_input(input_data):
    input_data_numpy = np.asarray(input_data)
    return input_data_numpy.reshape(1, -1)

# Fungsi untuk memprediksi kategori keparahan kanker
def predict_severity(input_data, model):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction[0]

# Load model dari file pickle
filename = 'cancer patient data sets.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Judul aplikasi web
st.title("Aplikasi Keparahan Kanker Paru-Paru")

# Input form untuk pengguna
st.sidebar.header("Masukkan Data Pasien:")

# Inisialisasi data frame dengan data pasien
data_columns = ['Age', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 
                'Genetic Risk', 'chronic Lung Disease', 'Fatigue', 'Weight Loss', 
                'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 
                'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']
input_data = {}
for column in data_columns:
    input_data[column] = st.sidebar.number_input(column, min_value=0, value=0)

# Tombol untuk memprediksi
if st.sidebar.button("Prediksi"):
    severity_prediction = predict_severity(list(input_data.values()), loaded_model)
    st.subheader("Hasil Prediksi:")
    if severity_prediction == 0:
        st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi')
    elif severity_prediction == 1:
        st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang')
    else:
        st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah')
