import streamlit as st
import numpy as np
import pickle

# Load model from pickle file
filename = 'cancer patient data sets.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Fungsi untuk memprediksi tingkat keparahan kanker
def predict_severity(input_data):
    input_data_numpy = np.asarray(input_data)
    data_reshaped = input_data_numpy.reshape(1, -1)
    prediksi = loaded_model.predict(data_reshaped)
    return prediksi[0]

# Tampilkan judul aplikasi
st.title('Prediksi Keparahan Kanker Paru-Paru dengan KNN')

# Tambahkan input pengguna menggunakan widget streamlit
age = st.slider('Usia Pasien', min_value=1, max_value=100, value=50)
# Tambahkan input lain sesuai kebutuhan

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Lakukan prediksi menggunakan fungsi predict_severity
    input_data = (1, age, 1, 3, 1, 5, 3, 4, 2, 2, 2, 2, 4, 2, 3)  # Sesuaikan dengan input pengguna
    hasil_prediksi = predict_severity(input_data)

    # Tampilkan hasil prediksi
    if hasil_prediksi == 0:
        st.error('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi')
    elif hasil_prediksi == 1:
        st.warning('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang')
    else:
        st.success('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah')
