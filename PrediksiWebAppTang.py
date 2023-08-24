import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Load the machine learning model
model = joblib.load('modelTang.pkl')

# Function to preprocess user input and make a prediction
def predict_price(input_data):
    return model.predict(input_data)

#furnishing
furnishing_mapping = {'unfurnished': 0, 'semi furnished': 1, 'furnished': 2, 'baru': 0}

# Streamlit web app
def main():
    st.title("Prediksi Harga Rumah Di Tangerang")
    st.write("Sebelum melakukan prediksi, dapat dilihat dahulu gambar distribusi selisih nilai\nprediksi dengan nilai sebenarnya. Hal ini ditujukan untuk pengguna dapat\nmemperkirakan besar kesalahan prediksi dari model ini.")
    st.write("Jika tidak terdapat wilayah yang Anda mau, mohon dimaafkan karena data kurang lengkap. Anda dapat memilih daerah terdekat.")

    #masukin gambar
    st.image("Distribution_of_Residuals_by_Ridge_Regression_in_Tangerang.jpg", use_column_width=True)
    st.caption('Selisih harga prediksi dengan harga aktual dari model (Regresi Linear)')

    # Get user inputs
    kamar_tidur = st.number_input("Jumlah Kamar Tidur")
    kamar_mandi = st.number_input("Jumlah Kamar Mandi")
    luas_tanah = st.number_input("Luas Tanah (m^2)")
    luas_bangunan = st.number_input("Luas Bangunan (m^2)")
    carport = st.number_input("Jumlah Carport")
    lantai = st.number_input("Jumlah Lantai")
    Tahun_rumah = st.number_input("Umur Rumah (Tahun)")
    furnishing = st.selectbox("Furnishing", list(furnishing_mapping.keys()))
    wilayah = st.selectbox("Wilayah", ['Alam Sutera', 'BSD', 'BSD Anggrek Loka', 'BSD Avani', 'BSD Bukit Golf', 'BSD City', 'BSD De Park', 'BSD Delatinos', 'BSD Duta Bintaro', 'BSD Eminent', 'BSD Foresta', 'BSD Giri Loka', 'BSD Graha Raya', 'BSD Green Cove', 'BSD Green Wich', 'BSD Griya Loka', 'BSD Kencana Loka', 'BSD Nusaloka', 'BSD Provance Parkland', 'BSD Puspita Loka', 'BSD Residence One', 'BSD Sevilla', 'BSD Taman Giri Loka', 'BSD Telaga Golf', 'BSD The Green', 'BSD The Icon', 'Babakan', 'Bakti Jaya', 'Balaraja', 'Banjar Wijaya', 'Batu Ceper', 'Benda', 'Bitung', 'Ciater', 'Cibodas', 'Cihuni', 'Cikokol', 'Cikupa', 'Cikupa Citra Raya', 'Ciledug', 'Cimone', 'Cipadu', 'Cipondoh', 'Cireundeu', 'Cisauk', 'Curug', 'Duta Garden', 'Gading Serpong', 'Gading Serpong Andalucia', 'Gading Serpong IL Lago', 'Gading Serpong L Agricola', 'Gading Serpong Pondok Hijau Golf', 'Gading Serpong Samara Village', 'Gading Serpong Scientia Garden', 'Gading Serpong The Spring', 'Graha Raya', 'Jatake', 'Jelupang', 'Jombang', 'Karang Tengah', 'Karawaci', 'Kelapa Dua', 'Kreo', 'Kresek', 'Larangan', 'Lengkong Kulon', 'Lippo Karawaci', 'Metro Permata', 'Modernland', 'Pagedangan', 'Panongan', 'Parigi', 'Pasar Kemis', 'Pengasinan', 'Pesanggrahan', 'Pinang', 'Pondok Benda', 'Pondok Betung', 'Pondok Cabe', 'Poris', 'Rajeg', 'Rawakalong', 'Rempoa', 'Sepatan', 'Serua', 'Sindang Jaya', 'Sudimara', 'Sutera Onix Alam Sutera', 'Sutera Sitara Alam Sutera', 'Tajur', 'Tanah Tinggi', 'Tangerang Kota', 'Tigaraksa'])

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
        'Lantai': [lantai],
        'TahunRumah': [Tahun_rumah]
        }

    # Add wilayah inputs
    wilayah_values = ['Alam Sutera', 'BSD', 'BSD Anggrek Loka', 'BSD Avani', 'BSD Bukit Golf', 'BSD City', 'BSD De Park', 'BSD Delatinos', 'BSD Duta Bintaro', 'BSD Eminent', 'BSD Foresta', 'BSD Giri Loka', 'BSD Graha Raya', 'BSD Green Cove', 'BSD Green Wich', 'BSD Griya Loka', 'BSD Kencana Loka', 'BSD Nusaloka', 'BSD Provance Parkland', 'BSD Puspita Loka', 'BSD Residence One', 'BSD Sevilla', 'BSD Taman Giri Loka', 'BSD Telaga Golf', 'BSD The Green', 'BSD The Icon', 'Babakan', 'Bakti Jaya', 'Balaraja', 'Banjar Wijaya', 'Batu Ceper', 'Benda', 'Bitung', 'Ciater', 'Cibodas', 'Cihuni', 'Cikokol', 'Cikupa', 'Cikupa Citra Raya', 'Ciledug', 'Cimone', 'Cipadu', 'Cipondoh', 'Cireundeu', 'Cisauk', 'Curug', 'Duta Garden', 'Gading Serpong', 'Gading Serpong Andalucia', 'Gading Serpong IL Lago', 'Gading Serpong L Agricola', 'Gading Serpong Pondok Hijau Golf', 'Gading Serpong Samara Village', 'Gading Serpong Scientia Garden', 'Gading Serpong The Spring', 'Graha Raya', 'Jatake', 'Jelupang', 'Jombang', 'Karang Tengah', 'Karawaci', 'Kelapa Dua', 'Kreo', 'Kresek', 'Larangan', 'Lengkong Kulon', 'Lippo Karawaci', 'Metro Permata', 'Modernland', 'Pagedangan', 'Panongan', 'Parigi', 'Pasar Kemis', 'Pengasinan', 'Pesanggrahan', 'Pinang', 'Pondok Benda', 'Pondok Betung', 'Pondok Cabe', 'Poris', 'Rajeg', 'Rawakalong', 'Rempoa', 'Sepatan', 'Serua', 'Sindang Jaya', 'Sudimara', 'Sutera Onix Alam Sutera', 'Sutera Sitara Alam Sutera', 'Tajur', 'Tanah Tinggi', 'Tangerang Kota', 'Tigaraksa']
    for w in wilayah_values:
        input_dict[f'Wilayah_{w}'] = [1 if w == wilayah else 0]

    # Add a "Predict" button
    if st.button("Predict"):
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
        kota_terpilih = [' Tangerang']
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
        st.write("Prediksi harga rumah: Rp.", predicted_price_original)
        st.write("Terimakasih sudah mencoba web app prediksi ini, jika kurang sesuai, harap maklum.")
        st.write("Sumber: Kaggle & Rumah123.com")

if __name__ == "__main__":
    main()

