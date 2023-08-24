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
kota_terpilih = [' Depok']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi', 'TahunRumah'], axis =1)

# feature encoding
#df['Kondisi'] = df['Kondisi'].astype('category').cat.codes
#df['furnishing'] = df['furnishing'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Wilayah'])
df.head(3)

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
    'Carport': [2],
    'Lantai': [2],
    'furnishing': [0],
    'Wilayah_Beji': [0],
    'Wilayah_Bojong Sari': [0],
    'Wilayah_Cilangkap': [0],
    'Wilayah_Cilodong': [0],
    'Wilayah_Cimanggis': [1],
    'Wilayah_Cinangka': [0],
    'Wilayah_Cinere': [0],
    'Wilayah_Cipayung': [0],
    'Wilayah_Cisalak': [0],
    'Wilayah_Citayam': [0],
    'Wilayah_Depok II': [0],
    'Wilayah_Gandul': [0],
    'Wilayah_Harjamukti': [0],
    'Wilayah_Kelapa Dua': [0],
    'Wilayah_Krukut': [0],
    'Wilayah_Kukusan': [0],
    'Wilayah_Limo': [0],
    'Wilayah_Margonda': [0],
    'Wilayah_Pancoran Mas': [0],
    'Wilayah_Pangkalan Jati': [0],
    'Wilayah_Rangkapanjaya': [0],
    'Wilayah_Sawangan': [0],
    'Wilayah_Sukatani': [0],
    'Wilayah_Sukmajaya': [0],
    'Wilayah_Tanah Baru': [0],
    'Wilayah_Tapos': [0],
    'Wilayah_Tirtajaya': [0]
})
#prediksi dengan model
#y_pred = rd.predict(new_data)

# Log transform: mengambil logaritma natural
#log_transformed_price = np.log1p(y_pred)

# Kembalikan nilai prediksi ke skala semula
#y_pred = np.expm1(scaler.inverse_transform(log_transformed_price))

#y_pred = y_pred/10

# Cetak prediksi harga rumah
#print("prediksi harga rumah:", y_pred)

#prediksi dengan model
y_pred = rd.predict(new_data)

# kembali ke harga semula
y_pred = np.expm1(scaler.inverse_transform(y_pred))

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modelDepok.pkl', 'wb') as file:  # open a text file
    pickle.dump(rd, file)