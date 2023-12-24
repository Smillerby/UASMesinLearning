import streamlit as st
import pickle
import numpy as np

# Load the saved model
filename = 'cancer patient data sets.sav'
model = pickle.load(open(filename, 'rb'))

# Streamlit UI
st.title('Keparahan Kanker Paru-Paru Predictor')

# Input form for user
input_data = st.text_area('Masukkan data pasien (pisahkan dengan koma)', '1,17,1,3,5,3,4,2,3,3,7,8,6,2,7')
input_data_list = [float(x.strip()) for x in input_data.split(',')]

# Reshape the input data
data_reshaped = np.asarray(input_data_list).reshape(1, -1)

# Make prediction
prediction = model.predict(data_reshaped)[0]

# Display the prediction
if prediction == 2:
    st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi')
elif prediction == 1:
    st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang')
elif prediction == 0:
    st.write('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah')
else:
    st.write('Tingkat keparahan tidak dikenali')
