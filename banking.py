import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load saved artifacts
with open("banking_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Streamlit app configuration
st.set_page_config(page_title="💼 Bank Term Deposit Predictor", layout="centered")

# Custom CSS for a classy theme
st.markdown(
    """
    <style>
    body {
        background-color: #f4f6f9;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    <div class="main">
    <div class="title">💼 Bank Term Deposit Predictor</div>
    <div class="subtitle">Analyze client data to predict subscription to term deposits</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user inputs
st.sidebar.header("📋 Client Information")

# Collect user input
def user_input():
    input_dict = {}

    input_dict['age'] = st.sidebar.slider('Age', 18, 100, 30)
    input_dict['balance'] = st.sidebar.number_input('Balance', value=1000)
    input_dict['day'] = st.sidebar.slider('Day Contacted', 1, 31, 15)
    input_dict['duration'] = st.sidebar.number_input('Call Duration (sec)', value=150)
    input_dict['campaign'] = st.sidebar.slider('Campaign Contacts', 1, 50, 1)
    input_dict['pdays'] = st.sidebar.number_input('Days Since Last Contact', value=999)
    input_dict['previous'] = st.sidebar.number_input('Previous Contacts', value=0)

    for col in label_encoders:
        if col != 'target_y':
            classes = label_encoders[col].classes_
            selected = st.sidebar.selectbox(f"{col.capitalize()}", classes)
            input_dict[col] = label_encoders[col].transform([selected])[0]

    return pd.DataFrame([input_dict])

input_df = user_input()
input_df = input_df.reindex(columns=feature_columns)
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(input_scaled)
    prediction_label = label_encoders['target_y'].inverse_transform(prediction)[0]

    st.markdown("---")
    if prediction_label == 'yes':
        st.success("✅ The client is likely to subscribe to a term deposit.")
        st.balloons()
    else:
        st.error("❌ The client is unlikely to subscribe to a term deposit.")

# Closing container
st.markdown("</div>", unsafe_allow_html=True)
#st.markdown("<center>Made with ❤️ using Streamlit | Developed by Shamna Nourin</center>", unsafe_allow_html=True)

