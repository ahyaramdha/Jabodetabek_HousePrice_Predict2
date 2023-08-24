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

df.rename({'Kecamatan' : 'Wilayah'}, inplace = True)

# Daftar kota yang ingin disertakan
kota_terpilih = [' Jakarta Selatan']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi', 'TahunRumah'], axis =1)

# feature encoding
#df['Kondisi'] = df['Kondisi'].astype('category').cat.codes
#df['furnishing'] = df['furnishing'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Wilayah'])

x = df.drop(columns = ['Harga'], axis =1)
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
    'K. Mandi': [2],
    'Luas Tanah': [70],
    'Luas Bangunan': [60],
    'Carport': [1],
    'Lantai':[1],
    'furnishing': [0],
    'Wilayah_Ampera': [0], 
    'Wilayah_Antasari': [0], 
    'Wilayah_Bangka': [0],
    'Wilayah_Bintaro': [0], 
    'Wilayah_Ciganjur': [0], 
    'Wilayah_Cilandak': [0],
    'Wilayah_Cinere': [1], 
    'Wilayah_Cipete': [0], 
    'Wilayah_Duren Tiga': [0],
    'Wilayah_Fatmawati': [0], 
    'Wilayah_Gandaria': [0], 
    'Wilayah_Gatot Subroto': [0],
    'Wilayah_Jagakarsa': [0], 
    'Wilayah_Jati Padang': [0], 
    'Wilayah_Kalibata': [0],
    'Wilayah_Kebagusan': [0], 
    'Wilayah_Kebayoran Baru': [0], 
    'Wilayah_Kebayoran Lama': [0],
    'Wilayah_Kemang': [0], 
    'Wilayah_Lebak Bulus': [0], 
    'Wilayah_Mampang Prapatan': [0],
    'Wilayah_Pancoran': [0], 
    'Wilayah_Pasar Minggu': [0], 
    'Wilayah_Pejaten': [0],
    'Wilayah_Pengadegan': [0], 
    'Wilayah_Permata Hijau': [0], 
    'Wilayah_Pondok Indah': [0],
    'Wilayah_Radio Dalam': [0], 
    'Wilayah_Ragunan': [0], 
    'Wilayah_Sektor 3A-Bintaro': [0],
    'Wilayah_Senayan': [0], 
    'Wilayah_Senopati': [0], 
    'Wilayah_Setiabudi': [0],
    'Wilayah_Simprug': [0], 
    'Wilayah_TB Simatupang': [0], 
    'Wilayah_Tanah Kusir': [0],
    'Wilayah_Tanjung Barat': [0], 
    'Wilayah_Tebet': [0],
    'Wilayah_patra kuningan': [0]
})

#prediksi dengan model
y_pred = rd.predict(new_data)

# kembali ke harga semula
y_pred = np.expm1(scaler.inverse_transform(y_pred))

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modelJaksel.pkl', 'wb') as file:  # open a text file
    pickle.dump(rd, file)