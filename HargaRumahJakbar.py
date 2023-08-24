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
kota_terpilih = [' Jakarta Barat']

# Filter DataFrame berdasarkan kolom "Kota"
df = df[df['Kota'].isin(kota_terpilih)]

df = df.drop(columns = ['K. Tidur Pembantu', 'K. Mandi Pembantu', 'Garasi', 'Listrik', 'Kota', 'Kondisi'], axis =1)

# feature encoding
#df['Kondisi'] = df['Kondisi'].astype('category').cat.codes
#df['furnishing'] = df['furnishing'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Wilayah'])
df.head(3)

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
    'K. Tidur': [3],
    'K. Mandi': [2],
    'Luas Tanah': [60],
    'Luas Bangunan': [100],
    'Carport': [1],
    'Lantai':[2],
    'furnishing': [0],
    'TahunRumah': [0],
    'Wilayah_Angke': [0],
    'Wilayah_Bojong Indah': [0],
    'Wilayah_Cengkareng': [0],
    'Wilayah_Citra Garden': [0],
    'Wilayah_Daan Mogot': [0],
    'Wilayah_Duri Kepa': [0],
    'Wilayah_Duri Kosambi': [0],
    'Wilayah_Green Lake City': [0],
    'Wilayah_Green Ville': [0],
    'Wilayah_Green garden': [0],
    'Wilayah_Grogol': [0],
    'Wilayah_Intercon': [0],
    'Wilayah_Jelambar': [0],
    'Wilayah_Joglo': [0],
    'Wilayah_Kalideres': [0],
    'Wilayah_Kebon Jeruk': [0],
    'Wilayah_Kedoya Utara': [0],
    'Wilayah_Kembangan': [0],
    'Wilayah_Kembangan Selatan': [0],
    'Wilayah_Kota Bambu Utara': [0],
    'Wilayah_Mangga Besar': [0],
    'Wilayah_Meruya': [1],
    'Wilayah_Metland Puri': [0],
    'Wilayah_Metro permata': [0],
    'Wilayah_Palmerah': [0],
    'Wilayah_Permata Buana': [0],
    'Wilayah_Pos Pengumben': [0],
    'Wilayah_Puri Indah': [0],
    'Wilayah_Puri Mansion': [0],
    'Wilayah_Puri Media': [0],
    'Wilayah_Slipi': [0],
    'Wilayah_Sunrise Garden': [0],
    'Wilayah_Taman Meruya': [0],
    'Wilayah_Taman Palem': [0],
    'Wilayah_Taman Ratu': [0],
    'Wilayah_Tamansari': [0],
    'Wilayah_Tambora': [0],
    'Wilayah_Tanjung Duren': [0],
    'Wilayah_Tomang': [0]
})

#prediksi dengan model
y_pred = rd.predict(new_data)

# kembali ke harga semula
y_pred = np.expm1(scaler.inverse_transform(y_pred))

y_pred = y_pred

# Cetak prediksi harga rumah
print("prediksi harga rumah: Rp.", y_pred.astype(int).round(-3))

import pickle

with open('modelJakbar.pkl', 'wb') as file:  # open a text file
    pickle.dump(rd, file)