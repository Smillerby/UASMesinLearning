import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ... (rest of your code)

# Assuming X and Y are your original features and labels
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

# Load the model
filename = 'cancer patient data sets.sav'
model1 = pickle.load(open(filename, 'rb'))

# ...

# Evaluasi model
y_true = y_test
y_pred = model1.predict(x_test)

# Streamlit App
st.title("Keparahan Kanker Paru-Paru Prediction App")

# Input form for user
st.sidebar.header("Masukkan Data Pasien:")
input_data = []
for i in range(15):
    input_data.append(st.sidebar.number_input(f"Fitur {i+1}", min_value=0, value=0))

# Button to make predictions
if st.sidebar.button("Prediksi"):
    severity_prediction = model1.predict(np.asarray(input_data).reshape(1, -1))[0]
    st.subheader("Hasil Prediksi:")
    st.write(f'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat {severity_prediction}')

# Display evaluation metrics
st.subheader("Confusion Matrix:")
st.text(confusion_matrix(y_true, y_pred))

st.subheader("Classification Report:")
st.text(classification_report(y_true, y_pred))
