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

df.rename(columns={'Kecamatan': 'Wilayah'}, inplace=True)

# Daftar kota yang ingin disertakan
kota_terpilih = [' Tangerang']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi'], axis =1)

# feature encoding
#df['Kondisi'] = df['Kondisi'].astype('category').cat.codes
#df['furnishing'] = df['furnishing'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Wilayah'])

#df = df.drop(columns = ['Wilayah_Labuan Bajo', 'Wilayah_Batik Danar Hadi'], axis =1)

x = df.drop(columns = ['Harga'], axis =1)
y = df['Harga']

from sklearn.model_selection import train_test_split

# Apply logarithmic transformation to the target variable
y_log = np.log1p(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.3, random_state=0)

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

new_data = pd.DataFrame({
    'K. Tidur': [2],
    'K. Mandi': [2],
    'Luas Tanah': [60],
    'Luas Bangunan': [50],
    'Carport': [2],
    'Lantai': [2],
    'furnishing': [0],
    'TahunRumah': [0],
    'Wilayah_Alam Sutera': [1],
    'Wilayah_BSD': [0],
    'Wilayah_BSD Anggrek Loka': [0],
    'Wilayah_BSD Avani': [0],
    'Wilayah_BSD Bukit Golf': [0],
    'Wilayah_BSD City': [0],
    'Wilayah_BSD De Park': [0],
    'Wilayah_BSD Delatinos': [0],
    'Wilayah_BSD Duta Bintaro': [0],
    'Wilayah_BSD Eminent': [0],
    'Wilayah_BSD Foresta': [0],
    'Wilayah_BSD Giri Loka': [0],
    'Wilayah_BSD Graha Raya': [0],
    'Wilayah_BSD Green Cove': [0],
    'Wilayah_BSD Green Wich': [0],
    'Wilayah_BSD Griya Loka': [0],
    'Wilayah_BSD Kencana Loka': [0],
    'Wilayah_BSD Nusaloka': [0],
    'Wilayah_BSD Provance Parkland': [0],
    'Wilayah_BSD Puspita Loka': [0],
    'Wilayah_BSD Residence One': [0],
    'Wilayah_BSD Sevilla': [0],
    'Wilayah_BSD Taman Giri Loka': [0],
    'Wilayah_BSD Telaga Golf': [0],
    'Wilayah_BSD The Green': [0],
    'Wilayah_BSD The Icon': [0],
    'Wilayah_Babakan': [0],
    'Wilayah_Bakti Jaya': [0],
    'Wilayah_Balaraja': [0],
    'Wilayah_Banjar Wijaya': [0],
    'Wilayah_Batu Ceper': [0],
    'Wilayah_Benda': [0],
    'Wilayah_Bitung': [0],
    'Wilayah_Ciater': [0],
    'Wilayah_Cibodas': [0],
    'Wilayah_Cihuni': [0],
    'Wilayah_Cikokol': [0],
    'Wilayah_Cikupa': [0],
    'Wilayah_Cikupa Citra Raya': [0],
    'Wilayah_Ciledug': [0],
    'Wilayah_Cimone': [0],
    'Wilayah_Cipadu': [0],
    'Wilayah_Cipondoh': [0],
    'Wilayah_Cireundeu': [0],
    'Wilayah_Cisauk': [0],
    'Wilayah_Curug': [0],
    'Wilayah_Duta Garden': [0],
    'Wilayah_Gading Serpong': [0],
    'Wilayah_Gading Serpong Andalucia': [0],
    'Wilayah_Gading Serpong IL Lago': [0],
    'Wilayah_Gading Serpong L Agricola': [0],
    'Wilayah_Gading Serpong Pondok Hijau Golf': [0],
    'Wilayah_Gading Serpong Samara Village': [0],
    'Wilayah_Gading Serpong Scientia Garden': [0],
    'Wilayah_Gading Serpong The Spring': [0],
    'Wilayah_Graha Raya': [0],
    'Wilayah_Jatake': [0],
    'Wilayah_Jelupang': [0],
    'Wilayah_Jombang': [0],
    'Wilayah_Karang Tengah': [0],
    'Wilayah_Karawaci': [0],
    'Wilayah_Kelapa Dua': [0],
    'Wilayah_Kreo': [0],
    'Wilayah_Kresek': [0],
    'Wilayah_Larangan': [0],
    'Wilayah_Lengkong Kulon': [0],
    'Wilayah_Lippo Karawaci': [0],
    'Wilayah_Metro Permata': [0],
    'Wilayah_Modernland': [0],
    'Wilayah_Pagedangan': [0],
    'Wilayah_Panongan': [0],
    'Wilayah_Parigi': [0],
    'Wilayah_Pasar Kemis': [0],
    'Wilayah_Pengasinan': [0],
    'Wilayah_Pesanggrahan': [0],
    'Wilayah_Pinang': [0],
    'Wilayah_Pondok Benda': [0],
    'Wilayah_Pondok Betung': [0],
    'Wilayah_Pondok Cabe': [0],
    'Wilayah_Poris': [0],
    'Wilayah_Rajeg': [0],
    'Wilayah_Rawakalong': [0],
    'Wilayah_Rempoa': [0],
    'Wilayah_Sepatan': [0],
    'Wilayah_Serua': [0],
    'Wilayah_Sindang Jaya': [0],
    'Wilayah_Sudimara': [0],
    'Wilayah_Sutera Onix Alam Sutera': [0],
    'Wilayah_Sutera Sitara Alam Sutera': [0],
    'Wilayah_Tajur': [0],
    'Wilayah_Tanah Tinggi': [0],
    'Wilayah_Tangerang Kota': [0],
    'Wilayah_Tigaraksa': [0]
})

#prediksi dengan model
y_pred = rd.predict(new_data)

# Log transform: mengambil logaritma natural
log_transformed_price = np.log1p(y_pred)

# Kembalikan nilai prediksi ke skala semula
y_pred = np.expm1(scaler.inverse_transform(log_transformed_price))

y_pred = y_pred

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modelTang.pkl', 'wb') as file:  # open a text file
    pickle.dump(rd, file)