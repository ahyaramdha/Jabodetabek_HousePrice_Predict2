import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('HargaRumahJabodetabek_filter.csv')

# Dictionary
mapping = {'unfurnished': 0 ,'semi furnished': 1, 'furnished': 2, 'baru': 0 }

# Menggunakan metode map untuk mengubah isi kolom
df['furnishing'] = df['furnishing'].map(mapping).astype('float')

df.rename({'Kecamatan':'Wilayah'}, inplace = True)

# Daftar kota yang ingin disertakan
kota_terpilih = [' Bekasi']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi', 'TahunRumah'], axis =1)

# feature encoding
#df['Kondisi'] = df['Kondisi'].astype('category').cat.codes
#df['furnishing'] = df['furnishing'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Wilayah'])

x = df.drop(columns = ['Harga', 'Wilayah_Cibubur'], axis =1)
y = df['Harga']

from sklearn.model_selection import train_test_split

# Apply logarithmic transformation to the target variable
y_log = np.log1p(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.4, random_state=0)

# Scaling
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

from sklearn.linear_model import Ridge

# Inisialisasi model Ridge dengan alpha (faktor regularisasi) tertentu
alpha = 1.0 
rd = Ridge(alpha=alpha)

rd.fit(x_train, y_train_scaled)

y_pred_scaled = rd.predict(x_test)

# kembali ke harga semula
y_pred = np.expm1(scaler.inverse_transform(y_pred_scaled))

# Input your new data here
new_data = pd.DataFrame({
    'K. Tidur': [2],
    'K. Mandi': [1],
    'Luas Tanah': [100],
    'Luas Bangunan': [100],
    'Carport': [1],
    'Lantai': [2],
    'furnishing': [0],
    'Wilayah_Babelan': [0],
    'Wilayah_Bantar Gebang': [0],
    'Wilayah_Bekasi': [0],
    'Wilayah_Bekasi Barat': [0],
    'Wilayah_Bekasi Kota': [0],
    'Wilayah_Bekasi Timur': [0],
    'Wilayah_Bekasi Utara': [0],
    'Wilayah_Bintara': [0],
    'Wilayah_Caman': [0],
    'Wilayah_Cibitung': [0],
    'Wilayah_Cikarang': [0],
    'Wilayah_Cikarang Selatan': [0],
    'Wilayah_Cikunir': [0],
    'Wilayah_Cimuning': [0],
    'Wilayah_Duren Jaya': [0],
    'Wilayah_Duta Harapan': [0],
    'Wilayah_Galaxy': [0],
    'Wilayah_Golden City': [1],
    'Wilayah_Grand Wisata': [0],
    'Wilayah_Harapan Baru': [0],
    'Wilayah_Harapan Indah': [0],
    'Wilayah_Harapan Jaya': [0],
    'Wilayah_Harapan Mulya': [0],
    'Wilayah_Jababeka': [0],
    'Wilayah_Jaka Sampurna': [0],
    'Wilayah_Jaka Setia': [0],
    'Wilayah_Jati Asih': [0],
    'Wilayah_Jati Cempaka': [0],
    'Wilayah_Jati Luhur': [0],
    'Wilayah_Jati Mekar': [0],
    'Wilayah_Jati Sari': [0],
    'Wilayah_Jatibening': [0],
    'Wilayah_Jatikramat': [0],
    'Wilayah_Jatimakmur': [0],
    'Wilayah_Jatimelati': [0],
    'Wilayah_Jatimurni': [0],
    'Wilayah_Jatiraden': [0],
    'Wilayah_Jatirangga': [0],
    'Wilayah_Jatiranggon': [0],
    'Wilayah_Jatisampurna': [0],
    'Wilayah_Jatiwaringin': [0],
    'Wilayah_Jatiwarna': [0],
    'Wilayah_Kaliabang': [0],
    'Wilayah_Karang Satria': [0],
    'Wilayah_Kayuringin Jaya': [0],
    'Wilayah_Kebalen': [0],
    'Wilayah_Kemang Pratama': [0],
    'Wilayah_Komsen': [0],
    'Wilayah_Kranji': [0],
    'Wilayah_Margahayu': [1],
    'Wilayah_Medan Satria': [0],
    'Wilayah_Mustikajaya': [0],
    'Wilayah_Mustikasari': [0],
    'Wilayah_Narogong': [0],
    'Wilayah_Pedurenan': [0],
    'Wilayah_Pejuang': [0],
    'Wilayah_Pekayon': [0],
    'Wilayah_Perwira': [0],
    'Wilayah_Pondok Gede': [0],
    'Wilayah_Pondok Ungu': [0],
    'Wilayah_Pondokmelati': [0],
    'Wilayah_Rawalumbu': [0],
    'Wilayah_Satriajaya': [0],
    'Wilayah_Serang Baru': [0],
    'Wilayah_Setu': [0],
    'Wilayah_Summarecon Bekasi': [0],
    'Wilayah_Tambun Selatan': [0],
    'Wilayah_Tambun Utara': [0],
    'Wilayah_Tanah Tinggi': [0],
    'Wilayah_Tarumajaya': [0]
})

#prediksi dengan model
y_pred = rd.predict(new_data)

# kembali ke harga semula
y_pred = np.expm1(scaler.inverse_transform(y_pred))

y_pred = y_pred

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modelBekasi.pkl', 'wb') as file:  # open a text file
    pickle.dump(rd, file)