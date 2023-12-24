import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('LungCancer.sav', 'rb'))

st.title('Prediksi Penyakit Kanker Paru-Paru')
c1, c2, c3 = st.columns(3)

with c1:
    Patient_Id = st.number_input('Identitas Pasien')
    Air_Pollution = st.number_input('Polusi Udara')
    OccuPational_Hazards = st.number_input('Bahaya Pekerjaan')
    Balanced_Diet = st.number_input('Diet Seimbang Pasien')
    Passive_Smoker = st.number_input('Riwayat Rokok Pasif')
    Fatigue = st.number_input('Gejala Kelelahan')
    Wheezing = st.number_input('Suara Saluran Pernapasan')
    Dry_Cough = st.number_input('Batuk Kering')

with c2:
    Age = st.number_input('Umur Pasien')
    Alcohol_use = st.number_input('Alkohol yang di konsumsi')
    Genetic_Risk = st.number_input('Resiko Genetik')
    Obesity = st.number_input('Riwayat Obesitas')
    Chest_Pain = st.number_input('Nyeri Dada')
    Weight_Loss  = st.number_input('Penurunan Berat Badan')
    Swallowing_Difficulty = st.number_input('Kesulitan Menelan')

with c3:
    Gender = st.number_input('Jenis Kelamin Pasien')
    Dust_Allergy = st.number_input('Alergi Debu')
    chronic_Lung_Disease = st.number_input('Riwayat Penyakit Paru-paru Kronis')
    Smoking = st.number_input('Riwayat Paien Merokok')
    Coughing_of_Blood = st.number_input('Batuk Berdarah')
    Shortness_of_Breath = st.number_input('Sesak Nafas')
    Clubbing_of_Finger_Nails = st.number_input('Kuku jari Tabuh')

prediksi = ''
if st.button('Hasil Prediksi'):
    prediksi = model.predict([[Patient_Id, Age,	Gender,	Air_Pollution, Alcohol_use, Dust_Allergy, OccuPational_Hazards,	Genetic_Risk, 
                               chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking, Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue, 
                               Weight_Loss, Shortness_of_Breath, Wheezing, Swallowing_Difficulty, Clubbing_of_Finger_Nails, Dry_Cough]])

    if (prediksi [0] == 0):
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi'
    elif(prediksi == 2):
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang'
    else:
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah'
st.success(prediksi)