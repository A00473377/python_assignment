import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
from pathlib import Path

#Making the API call and saving the data in cache.
@st.cache_data
def get_all_coin_ids(csv_path="CoinsData.csv"):
    if os.path.exists(csv_path):
        coins_data = pd.read_csv(csv_path)
        return coins_data

    else:
        api_url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(api_url)
        coins_list = response.json()
        coins_df = pd.DataFrame(coins_list)
        coins_df.to_csv(csv_path, index=False)
        return coins_df


#Retreiving historical price data
@st.cache_data
def get_historical_data(coin_id, start_date, end_date):
    s_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    e_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    headers = {"x-cg-demo-api-key": "CG-9faEcCfPn1E92gTP4GV7kgCQ"}
    params = {
        "vs_currency": "cad",
        "from": s_timestamp,
        "to": e_timestamp
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.set_index('date', inplace=True)
        return df


loaded_model = None


#Loading/Trainning model
def load_or_train_model():
    global loaded_model

    if loaded_model is None:
        model_path = Path("model.keras")

        if model_path.exists():
            print("Loading model...")
            loaded_model = tf.keras.models.load_model(model_path)

        else:
            print("Training model...")
            mnist = tf.keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0

            train_images = train_images[..., tf.newaxis].astype("float32")
            test_images = test_images[..., tf.newaxis].astype("float32")

            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10)
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            st.info("Training...")
            model.fit(train_images, train_labels, epochs=2, validation_split=0.1)
            model.save(model_path)
            loaded_model = model

    return loaded_model



#Converts image to the appropriate format and predicts the digit using the provided model.
def predict_digit(image, model):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.split()[-1] 
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predictions = tf.nn.softmax(predictions).numpy()
    label = np.argmax(predictions)
    confidence = np.max(predictions)
    return label, confidence



# Displaying analysis of historical price data for a single cryptocurrency.
def show_single_crypto_analysis():
    st.header("Single Stock Analysis")

    coins_list = get_all_coin_ids()
    if not coins_list.empty:
        coin_names = coins_list['id'].sort_values().tolist()
        selected_coin_name = st.selectbox("Select Stock", [""] + coin_names)
        
        if selected_coin_name:
            one_year_ago_date = datetime.now() - relativedelta(years=1)
            start_date_str = one_year_ago_date.strftime("%Y-%m-%d")
            end_date_str = datetime.now().strftime("%Y-%m-%d")
            df = get_historical_data(selected_coin_name, start_date_str, end_date_str)
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                max_price = df["price"].max()
                min_price = df["price"].min()
                max_date = df[df["price"] == max_price].index[0].strftime('%Y-%m-%d')
                min_date = df[df["price"] == min_price].index[0].strftime('%Y-%m-%d')


                df.index = df.index.strftime('%Y-%m-%d')
                st.line_chart(df["price"])
                st.write(f"The highest trading price was on {max_date} with a price of CAD {max_price}")
                st.write(f"The lowest trading price was on {min_date} with a price of CAD {min_price}")
    
    else:
        st.error("Could not load the coins list")



#Compares historical price data of two selected cryptocurrencies over a specified period.
def show_crypto_comparison():
    st.header("Stock Comparison")
    coins_list = get_all_coin_ids()
    if not coins_list.empty:
        coin_names = coins_list['id'].sort_values().tolist()
        
        selected_coin1 = st.selectbox("Select Stock 1", [""] + coin_names)
        selected_coin2 = st.selectbox("Second Stock 2", [""] + coin_names)
        start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End date", value=datetime.now())

        if selected_coin1 and selected_coin2 and start_date < end_date:

            if selected_coin1 != selected_coin2:

                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                df1 = get_historical_data(selected_coin1, start_date_str, end_date_str)
                df2 = get_historical_data(selected_coin2, start_date_str, end_date_str)

                df1.rename(columns={'price': selected_coin1}, inplace=True)
                df2.rename(columns={'price': selected_coin2}, inplace=True)
                df = df1[[selected_coin1]].join(df2[[selected_coin2]], how='outer')
                df.index = pd.to_datetime(df.index)
                df.index = df.index.strftime('%Y-%m-%d')

                if not df.empty:
                    st.line_chart(df)

                else:
                    st.write("No data available for the selected coins or date range")

            else:
                st.error("Ensure that both cryptocurrencies are not same")
        
        else:
            st.write("")
            st.write("")
            st.write("")
            st.info("Ensure that both cryptocurrencies are selected and the start date is before the end date")
    
    else:
        st.error("Could not load the coins list")



# Provides a user interface for digit recognition. Allow users to upload and classify images of digits
def show_digit_model():
    st.header("Digit Classifier")
    st.write("This app predicts digits from 0 to 9")

    model = load_or_train_model()
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        label, confidence = predict_digit(image, model)
        st.success(f'Prediction: {label} with confidence {confidence:.2f}')
        st.image(image, caption='Uploaded Digit', use_column_width=True)
        

    else:
        st.write("Please upload an image file to predict the digit.")





st.sidebar.title("Data Mining - Python Assignment")
page = st.sidebar.radio("Please navigate", ["Stock Details", "Stock Comparison", "Digit Classifier"])

if page == "Stock Details":
    show_single_crypto_analysis()

elif page == "Stock Comparison":
    show_crypto_comparison()

else:
    show_digit_model()

