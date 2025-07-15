import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("stock_price_model.pkl")

# Title
st.title("📈 Stock Price Predictor")

# Load sample data for feature statistics
df = pd.read_csv("TSLA.csv")
features = ['Open', 'High', 'Low', 'Volume']
st.sidebar.title("Feature Ranges")
st.sidebar.write(df[features].describe())


# Input fields
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0.0)

# Predict button
if st.button("Predict Closing Price"):
    input_df = pd.DataFrame([[open_price, high_price, low_price, volume]],
                            columns=['Open', 'High', 'Low', 'Volume'])
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Closing Price: ₹{prediction:.2f}")


