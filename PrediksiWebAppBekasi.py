import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Load the machine learning model
model = joblib.load('modelBekasi.pkl')

# Function to preprocess user input and make a prediction
def predict_price(input_data):
    return model.predict(input_data)

#furnishing
furnishing_mapping = {'unfurnished': 0, 'semi furnished': 1, 'furnished': 2, 'baru': 0}

# Streamlit web app
def main():
    st.title("Prediksi Harga Rumah Di Bekasi")
    st.write("Sebelum melakukan prediksi, dapat dilihat dahulu gambar distribusi selisih nilai\nprediksi dengan nilai sebenarnya. Hal ini ditujukan untuk pengguna dapat\nmemperkirakan besar kesalahan prediksi dari model ini.")
    
    #masukin gambar
    st.image("Distribution_of_Residuals_by_Ridge_Regression_in_Bekasi.jpg", use_column_width=True)
    st.caption('Selisih harga prediksi dengan harga aktual dari model (Regresi Ridge)')
    st.write("Jika tidak terdapat wilayah yang Anda mau, mohon dimaafkan karena data kurang lengkap. Anda dapat memilih daerah terdekat.")

    # Get user inputs
    kamar_tidur = st.number_input("Jumlah Kamar Tidur")
    kamar_mandi = st.number_input("Jumlah Kamar Mandi")
    luas_tanah = st.number_input("Luas Tanah (m^2)")
    luas_bangunan = st.number_input("Luas Bangunan (m^2)")
    carport = st.number_input("Jumlah Carport")
    lantai = st.number_input("Jumlah Lantai")
    furnishing = st.selectbox("Furnishing", list(furnishing_mapping.keys()))
    wilayah = st.selectbox("Wilayah", ['Babelan', 'Bantar Gebang', 'Bekasi', 'Bekasi Barat', 'Bekasi Kota', 'Bekasi Timur', 'Bekasi Utara', 'Bintara', 'Caman', 'Cibitung', 'Cikarang', 'Cikarang Selatan', 'Cikunir', 'Cimuning', 'Duren Jaya', 'Duta Harapan', 'Galaxy', 'Golden City', 'Grand Wisata', 'Harapan Baru', 'Harapan Indah', 'Harapan Jaya', 'Harapan Mulya', 'Jababeka', 'Jaka Sampurna', 'Jaka Setia', 'Jati Asih', 'Jati Cempaka', 'Jati Luhur', 'Jati Mekar', 'Jati Sari', 'Jatibening', 'Jatikramat', 'Jatimakmur', 'Jatimelati', 'Jatimurni', 'Jatiraden', 'Jatirangga', 'Jatiranggon', 'Jatisampurna', 'Jatiwaringin', 'Jatiwarna', 'Kaliabang', 'Karang Satria', 'Kayuringin Jaya', 'Kebalen', 'Kemang Pratama', 'Komsen', 'Kranji', 'Margahayu', 'Medan Satria', 'Mustikajaya', 'Mustikasari', 'Narogong', 'Pedurenan', 'Pejuang', 'Pekayon', 'Perwira', 'Pondok Gede', 'Pondok Ungu', 'Pondokmelati', 'Rawalumbu', 'Satriajaya', 'Serang Baru', 'Setu', 'Summarecon Bekasi', 'Tambun Selatan', 'Tambun Utara', 'Tanah Tinggi', 'Tarumajaya'])

    # Map furnishing to its corresponding value
    furnishing_value = furnishing_mapping.get(furnishing, 0)

    # Create a dictionary to hold user input data
    input_dict = {
        'Luas Tanah': [luas_tanah],
        'Luas Bangunan': [luas_bangunan],
        'K. Tidur': [kamar_tidur],
        'K. Mandi': [kamar_mandi],
        'furnishing': [furnishing_value],
        'Carport': [carport],
        'Lantai': [lantai]
    }

    # Add wilayah inputs
    wilayah_values = ['Babelan', 'Bantar Gebang', 'Bekasi', 'Bekasi Barat', 'Bekasi Kota', 'Bekasi Timur', 'Bekasi Utara', 'Bintara', 'Caman', 'Cibitung', 'Cikarang', 'Cikarang Selatan', 'Cikunir', 'Cimuning', 'Duren Jaya', 'Duta Harapan', 'Galaxy', 'Golden City', 'Grand Wisata', 'Harapan Baru', 'Harapan Indah', 'Harapan Jaya', 'Harapan Mulya', 'Jababeka', 'Jaka Sampurna', 'Jaka Setia', 'Jati Asih', 'Jati Cempaka', 'Jati Luhur', 'Jati Mekar', 'Jati Sari', 'Jatibening', 'Jatikramat', 'Jatimakmur', 'Jatimelati', 'Jatimurni', 'Jatiraden', 'Jatirangga', 'Jatiranggon', 'Jatisampurna', 'Jatiwaringin', 'Jatiwarna', 'Kaliabang', 'Karang Satria', 'Kayuringin Jaya', 'Kebalen', 'Kemang Pratama', 'Komsen', 'Kranji', 'Margahayu', 'Medan Satria', 'Mustikajaya', 'Mustikasari', 'Narogong', 'Pedurenan', 'Pejuang', 'Pekayon', 'Perwira', 'Pondok Gede', 'Pondok Ungu', 'Pondokmelati', 'Rawalumbu', 'Satriajaya', 'Serang Baru', 'Setu', 'Summarecon Bekasi', 'Tambun Selatan', 'Tambun Utara', 'Tanah Tinggi', 'Tarumajaya']
    for w in wilayah_values:
        input_dict[f'Wilayah_{w}'] = [1 if w == wilayah else 0]

    if st.button('Predict Here!'):
        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_dict)

        # Make prediction
        predicted_price = predict_price(input_df)

        # Kembalikan nilai prediksi ke skala semula
        # Scaling
        scaler = StandardScaler()

        # Read the dataset
        df = pd.read_csv('HargaRumahJabodetabek_filter.csv')

        # Daftar kota yang ingin disertakan
        kota_terpilih = [' Bekasi']
        # Filter DataFrame berdasarkan kolom "Kota"
        df = df[df['Kota'].isin(kota_terpilih)]

        # Scaling
        y_log = np.log1p(df['Harga'])
        y_test = y_log.values.reshape(-1, 1)
        train = scaler.fit_transform(y_test)

        # Reverse scaling and inverse log transformation
        # Log transform: mengambil logaritma natural
        log_transformed_price = np.log1p(predicted_price)
        # Inverse scale
        # Inverse log transformation
        predicted_price_original = np.expm1(scaler.inverse_transform(log_transformed_price))#_scaled)
        predicted_price_original = predicted_price_original/10

        # Display prediction result
        #st.write("Hasil Prediksi Harga Rumah:")
        st.write("Prediksi harga rumah: Rp.", predicted_price_original.astype(int).round(-3))
        st.write("Terimakasih sudah mencoba web app prediksi ini, jika kurang sesuai, harap maklum.")
        st.write("Sumber: Kaggle & Rumah123.com")

if __name__ == "__main__":
    main()
