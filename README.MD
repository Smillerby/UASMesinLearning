# Laporan Proyek Machine Learning
### Nama : Muhammad Irpan
### Nim : 211351094
### Kelas : Malam A

## Domain Proyek
Proyek ini bertujuan untuk melakukan prediksi dan analisis terkait kanker paru-paru. Data yang digunakan mencakup informasi tentang pasien, termasuk riwayat medis, faktor-faktor risiko, dan gejala yang dialami. Tujuan dari proyek ini adalah untuk mengembangkan model prediksi yang dapat membantu dalam menentukan tingkat keparahan kanker paru-paru pada pasien. Analisis ini bertujuan untuk memahami faktor-faktor yang berkontribusi terhadap risiko kanker paru-paru, serta memprediksi tingkat keparahan berdasarkan data pasien. Hasil dari proyek ini dapat membantu dalam diagnosis dini dan perencanaan perawatan pasien kanker paru-paru.

## Business Understanding
Sekarang, kita sedang fokus untuk memahami betul tentang penyakit kanker paru-paru. Ini melibatkan penggalian lebih dalam terkait faktor-faktor risiko yang bisa menyebabkan kanker paru-paru muncul, juga pengetahuan tentang gejala-gejala yang mungkin dialami oleh penderita. Selain itu, kita ingin tahu bagaimana informasi medis dan data pasien bisa digunakan untuk memprediksi seberapa parahnya kanker paru-paru.

Proyek ini punya tujuan besar, yaitu mengembangkan model prediksi yang sangat akurat. Model ini diharapkan bisa jadi alat yang membantu dalam mendiagnosis kanker paru-paru lebih awal dan menentukan seberapa parahnya penyakit pada pasien. Dengan melakukan analisis ini, kita berusaha memahami sejauh mana faktor-faktor risiko berpengaruh pada penyakit ini dan mencari cara efektif untuk memprediksi tingkat keparahannya berdasarkan data pasien.

### Problem Statements
- Keterbatasan Diagnosis Dini: Terdapat kendala dalam mendeteksi kanker paru-paru pada tahap awal, yang dapat mempengaruhi efektivitas pengobatan dan tingkat kesembuhan pasien.

- Kompleksitas Faktor Risiko: Faktor-faktor risiko yang berkontribusi terhadap munculnya kanker paru-paru masih belum sepenuhnya dipahami, mempersulit upaya pencegahan dan intervensi.

- Ketidakpastian Tingkat Keparahan: Penentuan tingkat keparahan kanker paru-paru yang tepat saat diagnosis seringkali menantang, sehingga menyulitkan perencanaan perawatan yang optimal.

 ### Goals
 - Tujuan dari proyek ini adalah untuk mengembangkan model prediksi yang akurat untuk menentukan tingkat keparahan kanker paru-paru pada pasien. Model ini akan digunakan untuk diagnosis dini, membantu dalam perencanaan perawatan yang lebih efektif, dan meningkatkan pemahaman tentang faktor-faktor risiko yang berkontribusi terhadap penyakit kanker paru-paru.

  ### Solution statements
Saya akan mengembangkan model prediksi untuk membantu memprediksi seberapa parah kanker paru-paru pada pasien. Model ini akan memanfaatkan informasi medis dan faktor-faktor risiko pasien dengan menggunakan algoritma Machine Learning yang disebut sebagai K-Nearest Neighbors (KNN).

  ## Data Understanding
  Pada tahap ini, kita sedang membangun dasar analisis untuk penelitian dengan merangkum dan mengidentifikasi potensi masalah dalam data yang telah dikumpulkan terkait prediksi kanker paru-paru.

  https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data

  ### Variabel-variabel pada Lung Cancer Prediction  adalah sebagai berikut:    
 1. Patient_Id = identitas Pasien = object
 2. Age = Umur Pasien = int64
 3. Gender = Jenis Kelamin Pasien = int64 
 4. Air_Pollution = Polusi Udara = int64 
 5. Alcohol_use = Alohol yang di konsumsi = int64 
 6. Dust_Allergy = Alergi Debu =  int64 
 7. OccuPational_Hazards = Bahaya Pekerjaan = int64 
 8. Genetic_Risk = Resiko Genetik = int64 
 9. chronic_Lung_Disease = Riwayat penyakit paru paru kronis = int64 
 10. Balanced_Diet = Diet seimbang pasien = int64 
 11. Obesity = Riwayat obesitas = int64 
 12. Smoking = Riwayat Merokok = int64 
 13. Passive_Smoker = Merokok Passif =  int64 
 14. Chest_Pain = Nyeri pada dada = int64 
 15. Coughing_of_Blood = Batuk berdarah = int64 
 16. Fatigue = Gejala Kelelahan = int64 
 17. Weight_Loss = Penurunan berat badan = int64 
 18. Shortness_of_Breath = Sesak nafas = int64 
 19. Wheezing = Suara saluran pernapasan = int64 
 24. Snoring = int64 
 25. Level = object
 
## Data Preparation
### Data Collection

Data ini merupakan koleksi informasi medis yang berharga yang mencakup riwayat penyakit, faktor-faktor risiko, dan gejala pasien. Data tersebut dikumpulkan dari catatan medis pasien di berbagai lembaga medis dan rumah sakit. Data ini berperan penting dalam upaya diagnosis dini dan pemahaman faktor-faktor yang berkontribusi terhadap penyakit kanker paru-paru.

### Data Discovery And Profiling
Library-library yang saya butuhkan 
``` bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
```
Cara pemanggilan dataset dan melihat 5 data paling atas
```bash
df = pd.read_csv("cancer_patient.csv")
df.head()
```
melihat type data yang digunakan
``` bash
df.info()
```
Membuat sebuah histogram dari data di dalam DataFrame.
``` bash
p = df.hist(figsize = (20,20))
![Alt text](gambar1.png) <br>
```
metransformasi data atau memisahkan numerik dan kategori
Dengan cara ini, code tersebut melakukan preprocessing data dengan mengisi nilai yang hilang sesuai dengan karakteristik data numerik atau kategorikal, yaitu dengan median untuk data numerik dan modus untuk data kategorikal.
``` bash
numerical = []
catgcols = []

for col in df.columns:
if df[col].dtype == 'float64':
numerical.append(col)
elif df[col].dtype == 'int64':
numerical.append(col)
else:
catgcols.append(col)

for col in df.columns:
if col in numerical:
df[col].fillna(df[col].median(), inplace=True)
else:
df[col].fillna(df[col].mode()[0], inplace=True)
```
disini kita akan melakukan pemisahan data untuk membuat model prediksi
``` bash
X = df.drop(columns=['Level', 'index', 'Frequent_Cold', 'Snoring'], axis=1)
Y = df['Level']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, x_train.shape, x_test.shape)
```
## Modeling
lalu kita akan membuat sebuah objek DecisionTreeClassifier, yang merupakan model pohon keputusan untuk tugas klasifikasi.
``` bash
desicion = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    random_state=42, splitter='best'
)
model = desicion.fit(x_train, y_train)
```

selanjutnya kita akan mengukur akurasi dari model yang kita gunakan
``` bash
x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)
print('Akurasi data trainin: ', training_data_accuracy)

x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)
print('Akurasi data testing : ', test_data_accuracy)
```
Akurasi data training :  0.99125
Akurasi data testing :  0.985

kita akan membuat grafik pie (pie chart) yang menggambarkan peluang kanker paru-paru berdasarkan tingkat polusi udara.Hasilnya adalah grafik pie yang menggambarkan proporsi dari tingkat polusi udara yang berkontribusi terhadap peluang terkena kanker paru-paru.
``` bash
plt.figure(figsize = (11, 9))
plt.title("Lung Cancer Chances Due to Air Polution")
plt.pie(df['Level'].value_counts(), explode = (0.1, 0.02, 0.02), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%", shadow = True)
plt.legend(title = "Lung Cancer Chances", loc = "lower left")
```
![Alt text](gambar1.png) <br>

ini digunakan untuk membuat dan menampilkan visualisasi pohon keputusan (decision tree).
``` bash
fig = plt.figure(figsize=(25, 25))
_ = tree.plot_tree(
model,
feature_names = independen,
class_names = ['Low', 'Medium', 'High'],
filled = True)
```
![Alt text](gambar2.png) <br>

## Evaluasi
Disini kita akan memprediksi tingkat keparahan kanker paru-paru berdasarkan data masukan.
``` bash
input_data = (1,17,1,3,1,5,3,4,2,2,2,2,4,2,3,1,3,7,8,6,2,7)
input_data_numpy = np.asarray(input_data)
data_reshaped = input_data_numpy.reshape(1, -1)
prediksi = model.predict(data_reshaped)
print(prediksi)

if (prediksi[0] == 0):
print('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi')
elif(prediksi == 2):
print('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang')
else:
print('Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah')
```


## Deployment

  [App](https://pembelajaranmesin-lgvbhzfkthzbmshpdbz7tx.streamlit.app/)

  ![Alt text](gambar3.png)
