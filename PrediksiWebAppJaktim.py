import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Load the machine learning model
model = joblib.load('modeljaktim.pkl')

# Function to preprocess user input and make a prediction
def predict_price(input_data):
    return model.predict(input_data)

#furnishing
furnishing_mapping = {'unfurnished': 0, 'semi furnished': 1, 'furnished': 2, 'baru': 0}

# Streamlit web app
def main():
    st.title("Prediksi Harga Rumah Di Jakarta Timur")
    st.write("Sebelum melakukan prediksi, dapat dilihat dahulu gambar distribusi selisih nilai\nprediksi dengan nilai sebenarnya. Hal ini ditujukan untuk pengguna dapat\nmemperkirakan besar kesalahan prediksi dari model ini.")

    #masukin gambar
    st.image("Distribution_of_Residuals_by_Linear_Regression_Model_in_East_Jakarta.jpg", use_column_width=True)
    st.caption('Selisih harga prediksi dengan harga aktual dari model (Regresi Linear)')
    st.write("Jika tidak terdapat wilayah yang Anda mau, mohon dimaafkan karena data kurang lengkap. Anda dapat memilih daerah terdekat.")

    # Get user inputs
    kamar_tidur = st.number_input("Jumlah Kamar Tidur")
    kamar_mandi = st.number_input("Jumlah Kamar Mandi")
    luas_tanah = st.number_input("Luas Tanah (m^2)")
    luas_bangunan = st.number_input("Luas Bangunan (m^2)")
    carport = st.number_input("Jumlah Carport")
    lantai = st.number_input("Jumlah Lantai")
    furnishing = st.selectbox("Furnishing", list(furnishing_mapping.keys()))
    wilayah = st.selectbox("Wilayah", ['Cakung', 'Cawang', 'Cibubur', 'Cipinang', 'Ciracas', 'Duren Sawit', 'Jatinegara', 'Jatiwaringin', 'Kayu Putih', 'Kramat Jati', 'Lubang Buaya', 'Matraman', 'Pasar Rebo', 'Pinang Ranti', 'Pondok Bambu', 'Pondok Kelapa', 'Pulo Gadung', 'Pulomas', 'Raffles Hills', 'Rawamangun', 'Utan Kayu'])

    # Map furnishing to its corresponding value
    furnishing_value = furnishing_mapping.get(furnishing, 0)

    # Create a dictionary to hold user input data
    input_dict = {
        'K. Tidur': [kamar_tidur],
        'K. Mandi': [kamar_mandi],
        'Luas Tanah': [luas_tanah],
        'Luas Bangunan': [luas_bangunan],
        'Carport': [carport],
        'Lantai': [lantai],
        'Furnishing': [furnishing_value]
    }

    # Add wilayah inputs
    wilayah_values = ['Cakung', 'Cawang', 'Cibubur', 'Cipinang', 'Ciracas', 'Duren Sawit', 'Jatinegara', 'Jatiwaringin', 'Kayu Putih', 'Kramat Jati', 'Lubang Buaya', 'Matraman', 'Pasar Rebo', 'Pinang Ranti', 'Pondok Bambu', 'Pondok Kelapa', 'Pulo Gadung', 'Pulomas', 'Raffles Hills', 'Rawamangun', 'Utan Kayu']
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
        kota_terpilih = [' Jakarta Timur']
        # Filter DataFrame berdasarkan kolom "Kota"
        df = df[df['Kota'].isin(kota_terpilih)]

        # Scaling
        y_log = np.log1p(df['Harga'])
        y_test = y_log.values.reshape(-1, 1)
        train = scaler.fit_transform(y_test)

        # Reverse scaling and inverse log transformation
        # Inverse scale
        predicted_price_scaled = scaler.inverse_transform(predicted_price.reshape(-1, 1))
        # Inverse log transformation
        predicted_price_original = np.expm1(predicted_price_scaled)

        # Display prediction result
        #st.write("Hasil Prediksi Harga Rumah:")
        st.write("Prediksi harga rumah: Rp.", predicted_price_original.astype(int).round(-3))
        st.write("Terimakasih sudah mencoba web app prediksi ini. Jika kurang sesuai, harap maklum.")

if __name__ == "__main__":
    main()
