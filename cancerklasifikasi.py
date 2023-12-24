import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ... (bagian lain dari kode Anda)

# Mengasumsikan data awal Anda berada dalam DataFrame yang disebut 'df'
X = df.drop('Level', axis=1)  # Fitur
Y = df['Level']  # Label

# Membagi data menjadi set pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

# Memuat model
filename = 'cancer-patients-and-air-pollution-a-new-link/cancer patient data sets.sav'
model1 = pickle.load(open(filename, 'rb'))

# ...

# Evaluasi model
y_true = y_test
y_pred = model1.predict(x_test)

# Aplikasi Streamlit
st.title("Aplikasi Prediksi Keparahan Kanker Paru-Paru")

# Formulir input untuk pengguna
st.sidebar.header("Masukkan Data Pasien:")
input_data = []
for i in range(15):
    input_data.append(st.sidebar.number_input(f"Fitur {i+1}", min_value=0, value=0))

# Tombol untuk membuat prediksi
if st.sidebar.button("Prediksi"):
    severity_prediction = model1.predict(np.asarray(input_data).reshape(1, -1))[0]
    st.subheader("Hasil Prediksi:")
    st.write(f'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat {severity_prediction}')

# Menampilkan metrik evaluasi
st.subheader("Confusion Matrix:")
st.text(confusion_matrix(y_true, y_pred))

st.subheader("Classification Report:")
st.text(classification_report(y_true, y_pred))
