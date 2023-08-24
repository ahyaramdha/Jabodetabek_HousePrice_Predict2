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

# Daftar kota yang ingin disertakan
kota_terpilih = [' Jakarta Timur']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df['K. Tidur'] = df['K. Tidur'] + df['K. Tidur Pembantu']
df['K. Mandi'] = df['K. Mandi'] + df['K. Mandi Pembantu']

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi', 'TahunRumah'], axis =1)

df.rename(columns={'Kecamatan': 'Wilayah'}, inplace=True)

# feature encoding
df = pd.get_dummies(df, columns=['Wilayah'])

x = df.drop(columns = ['Harga'], axis =1)
y = df['Harga']

from sklearn.model_selection import train_test_split

# transformasi logaritma agar hasilnya tidak nan karena terlalu besar
y_log = np.log1p(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.3, random_state=0)

# Scaling data
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Linear regression algorithm
lr = LinearRegression()
lr.fit(x_train, y_train_scaled)

# Predict
y_pred_scaled = lr.predict(x_test)

# Inverse transform y_pred_scaled
y_pred = np.expm1(scaler.inverse_transform(y_pred_scaled))

new_data = pd.DataFrame({
    'K. Tidur': [3],
    'K. Mandi': [3],
    'Luas Tanah': [112],
    'Luas Bangunan': [112],
    'Carport': [2],
    'Lantai':[1],
    'furnishing': [0],
    'Wilayah_Cakung': [0],
    'Wilayah_Cawang': [0],
    'Wilayah_Cibubur': [0],
    'Wilayah_Cipinang': [0],
    'Wilayah_Ciracas': [0],
    'Wilayah_Duren Sawit': [0],
    'Wilayah_Jatinegara': [0], 
    'Wilayah_Jatiwaringin': [0],
    'Wilayah_Kayu Putih': [0],
    'Wilayah_Kramat Jati': [0],
    'Wilayah_Lubang Buaya': [0],
    'Wilayah_Matraman': [0], 
    'Wilayah_Pasar Rebo': [0], 
    'Wilayah_Pinang Ranti': [0],
    'Wilayah_Pondok Bambu': [0],  
    'Wilayah_Pondok Kelapa': [0],
    'Wilayah_Pulo Gadung': [0],  
    'Wilayah_Pulomas': [1], 
    'Wilayah_Raffles Hills': [0], 
    'Wilayah_Rawamangun': [0], 
    'Wilayah_Utan Kayu': [0]
  
})

#prediksi dengan model
y_pred = lr.predict(new_data)

# Log transform: mengambil logaritma natural
log_transformed_price = np.log1p(y_pred)

# Kembalikan nilai prediksi ke skala semula
y_pred = np.expm1(scaler.inverse_transform(log_transformed_price))

y_pred = y_pred

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modeljaktim.pkl', 'wb') as file:  # open a text file
    pickle.dump(lr, file)